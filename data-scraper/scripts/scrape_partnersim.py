# scripts/scrape_partnersim.py
import csv
import re
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

START_URL = "https://www.ycarena.com/games/partnersim"

SLEEP_SEC = 0.6
MAX_ITEMS = 5000

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "raw" / "partnersim.csv"
DEBUG_DIR = ROOT / "data" / "raw" / "debug" / "screenshots"
LOGS_DIR = ROOT / "logs"
LOG_FILE = LOGS_DIR / "scrape.log"

ACCEPT_BTN = "button:has-text('Accept')"
REJECT_BTN = "button:has-text('Reject')"

# IMPORTANT: use ONLY the explicit "Next pitch" control(s)
NEXT_PITCH_SELECTORS = [
    "button:has-text('Next pitch')",
    "button:has-text('Next Pitch')",
    "a:has-text('Next pitch')",
    "a:has-text('Next Pitch')",
    "[role='button']:has-text('Next pitch')",
    "[role='button']:has-text('Next Pitch')",
    "text=/^\\s*Next pitch\\s*$/i",
]

# Sometimes there's a start/consent overlay
START_SELECTORS = [
    "button:has-text('Start')",
    "button:has-text('Play')",
    "button:has-text('Begin')",
    "button:has-text('Continue')",
    "button:has-text('I Agree')",
    "button:has-text('Accept all')",
    "button:has-text('OK')",
    "text=/^\\s*Start\\s*$/i",
]

VIDEO_IFRAME = (
    "iframe[src*='youtube.com'], "
    "iframe[src*='youtube-nocookie.com'], "
    "iframe[src*='vimeo.com']"
)

ACCEPTED_KEYWORDS = ["accepted", "got in", "admitted"]
REJECTED_KEYWORDS = ["rejected", "didn't get in", "did not get in", "not accepted", "declined"]


def ensure_dirs() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{ts} {msg}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def normalize_video(iframe_src: str) -> Tuple[str, str, str]:
    if not iframe_src:
        return ("", "", "")

    m = re.search(r"(?:youtube\.com|youtube-nocookie\.com)/embed/([a-zA-Z0-9_-]{11})", iframe_src)
    if m:
        vid = m.group(1)
        return ("youtube", vid, f"https://www.youtube.com/watch?v={vid}")

    m2 = re.search(r"vimeo\.com/(?:video/)?(\d+)", iframe_src)
    if m2:
        vid = m2.group(1)
        return ("vimeo", vid, f"https://vimeo.com/{vid}")

    return ("unknown", "", iframe_src)


def click_first_visible(page, selectors: list[str], timeout_ms: int = 2000) -> bool:
    for sel in selectors:
        loc = page.locator(sel).first
        try:
            if loc.count() == 0:
                continue
            if not loc.is_visible():
                continue
            loc.scroll_into_view_if_needed(timeout=1000)
            loc.click(timeout=timeout_ms)
            return True
        except Exception:
            continue
    return False


def parse_label(text: str) -> Optional[int]:
    t = (text or "").lower()
    if any(k in t for k in ACCEPTED_KEYWORDS):
        return 1
    if any(k in t for k in REJECTED_KEYWORDS):
        return 0
    if "accepted" in t:
        return 1
    if "rejected" in t:
        return 0
    return None


def get_iframe_src(page) -> str:
    return page.locator(VIDEO_IFRAME).first.get_attribute("src") or ""


def wait_for_video_change(page, old_src: str, timeout_ms: int = 8000) -> bool:
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        try:
            if page.locator(VIDEO_IFRAME).count() == 0:
                time.sleep(0.1)
                continue
            new_src = get_iframe_src(page)
            if new_src and new_src != old_src:
                return True
        except Exception:
            pass
        time.sleep(0.15)
    return False


def load_seen_video_urls() -> set[str]:
    seen = set()
    if not OUT_CSV.exists():
        return seen
    try:
        with OUT_CSV.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                url = (row.get("video_url") or "").strip()
                if url:
                    seen.add(url)
    except Exception:
        pass
    return seen


def main() -> None:
    ensure_dirs()
    new_file = not OUT_CSV.exists()
    seen_urls = load_seen_video_urls()

    HEADFUL = os.getenv("HEADFUL", "").strip() == "1"

    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "source_page_url", "platform", "video_id", "video_url", "iframe_src", "label", "scraped_at_utc"],
        )
        if new_file:
            writer.writeheader()

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not HEADFUL, slow_mo=150 if HEADFUL else 0)
            page = browser.new_page()

            log(f"Opening {START_URL}")
            page.goto(START_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(1200)
            click_first_visible(page, START_SELECTORS)

            for idx in range(MAX_ITEMS):
                # Ensure iframe exists
                try:
                    page.wait_for_selector(VIDEO_IFRAME, timeout=8000)
                except PWTimeoutError:
                    snap = DEBUG_DIR / f"no_video_{idx}.png"
                    page.screenshot(path=str(snap), full_page=True)
                    log(f"[WARN] idx={idx} missing video iframe. Saved {snap.name}")
                    click_first_visible(page, START_SELECTORS)
                    page.wait_for_timeout(600)
                    continue

                iframe_src_before = get_iframe_src(page)
                platform, video_id, video_url = normalize_video(iframe_src_before)

                # Reveal result
                if page.locator(ACCEPT_BTN).count() > 0 and page.locator(ACCEPT_BTN).first.is_visible():
                    page.locator(ACCEPT_BTN).first.click(timeout=3000)
                elif page.locator(REJECT_BTN).count() > 0 and page.locator(REJECT_BTN).first.is_visible():
                    page.locator(REJECT_BTN).first.click(timeout=3000)
                else:
                    snap = DEBUG_DIR / f"no_choice_{idx}.png"
                    page.screenshot(path=str(snap), full_page=True)
                    log(f"[ERROR] idx={idx} Accept/Reject not visible. Saved {snap.name}")
                    break

                page.wait_for_timeout(400)

                # Parse label from page text (good enough here)
                body_text = page.inner_text("body")
                label = parse_label(body_text)
                if label is None:
                    snap = DEBUG_DIR / f"no_label_{idx}.png"
                    page.screenshot(path=str(snap), full_page=True)
                    log(f"[WARN] idx={idx} could not parse label. Saved {snap.name}")
                else:
                    if video_url and video_url not in seen_urls:
                        writer.writerow(
                            {
                                "idx": idx,
                                "source_page_url": page.url,
                                "platform": platform,
                                "video_id": video_id,
                                "video_url": video_url,
                                "iframe_src": iframe_src_before,
                                "label": label,
                                "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        f.flush()
                        seen_urls.add(video_url)
                        log(f"[OK] idx={idx} label={label} platform={platform} video={video_url}")
                    else:
                        log(f"[SKIP] idx={idx} duplicate or missing video_url={video_url}")

                # Click explicit "Next pitch"
                if not click_first_visible(page, NEXT_PITCH_SELECTORS, timeout_ms=2500):
                    snap = DEBUG_DIR / f"no_next_{idx}.png"
                    page.screenshot(path=str(snap), full_page=True)
                    log(f"[ERROR] idx={idx} could not find visible 'Next pitch'. Saved {snap.name}")
                    break

                # Wait until iframe src changes (prevents duplicates)
                if not wait_for_video_change(page, iframe_src_before, timeout_ms=9000):
                    snap = DEBUG_DIR / f"next_no_change_{idx}.png"
                    page.screenshot(path=str(snap), full_page=True)
                    log(f"[WARN] idx={idx} clicked Next pitch but video did not change. Saved {snap.name}")
                    # Try one more click
                    click_first_visible(page, NEXT_PITCH_SELECTORS, timeout_ms=2500)
                    wait_for_video_change(page, iframe_src_before, timeout_ms=9000)

                time.sleep(SLEEP_SEC)

            browser.close()
            log("Done.")


if __name__ == "__main__":
    main()
