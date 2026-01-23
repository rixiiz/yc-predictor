import subprocess
from pathlib import Path

def youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"

def download_youtube(video_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = (out_dir / f"{video_id}.%(ext)s").resolve()
    url = youtube_url(video_id)

    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "-o", str(out_template),

        "--retries", "10",
        "--fragment-retries", "10",
        "--retry-sleep", "fragment:2",
        "--socket-timeout", "30",

        "--add-header", "User-Agent:Mozilla/5.0",
        "--add-header", f"Referer:{url}",

        # IMPORTANT: give yt-dlp a JS runtime
        "--js-runtimes", "node",

        "--no-cache-dir",
        url,
    ]

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        pretty = subprocess.list2cmdline(cmd)
        msg = (
            f"yt-dlp failed (exit {proc.returncode}).\n\n"
            f"CMD: {pretty}\n\n"
            f"STDOUT:\n{proc.stdout}\n\n"
            f"STDERR:\n{proc.stderr}\n"
        )
        raise RuntimeError(msg)

    matches = list(out_dir.glob(f"{video_id}.*"))
    if not matches:
        raise FileNotFoundError(f"Download succeeded but file not found for {video_id}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]
