import csv
import json
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from src.config import (
    DATASET_CSV,
    TRANSCRIPTS_CSV,
    TMP_VIDEOS,
    TMP_AUDIO,
    TMP_FRAMES,
    ensure_dirs,
)
from src.media.youtube import download_youtube
from src.media.audio import extract_wav
from src.media.frames import extract_first_n_frames
from src.features.frame_feats import summarize_first_frames

@dataclass
class Item:
    youtube_id: str
    label: int
    year: int

def read_dataset(path: Path) -> list[Item]:
    items: list[Item] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            yt = (r.get("youtube_id") or "").strip()
            if not yt:
                continue
            items.append(
                Item(
                    youtube_id=yt,
                    label=int(r["label"]),
                    year=int(r["year"]),
                )
            )
    # dedupe by youtube_id
    seen = set()
    out = []
    for it in items:
        if it.youtube_id in seen:
            continue
        seen.add(it.youtube_id)
        out.append(it)
    return out

def transcribe_wav(model: WhisperModel, wav_path: Path) -> tuple[str, dict]:
    segments, info = model.transcribe(str(wav_path), vad_filter=True)
    texts = []
    seg_count = 0
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            texts.append(t)
        seg_count += 1
    transcript = " ".join(texts)
    stats = {
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "segments": seg_count,
        "num_words": len(transcript.split()),
    }
    return transcript, stats

def main():
    ensure_dirs()
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing {DATASET_CSV}")

    items = read_dataset(DATASET_CSV)
    print(f"Dataset items: {len(items)}")

    # Whisper (CPU-friendly)
    model = WhisperModel("base", device="cpu", compute_type="int8")

    TRANSCRIPTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    failed_path = TRANSCRIPTS_CSV.parent / "transcribe_failed.jsonl"
    with TRANSCRIPTS_CSV.open("w", encoding="utf-8", newline="") as out_f, \
         failed_path.open("w", encoding="utf-8") as fail_f:

        writer = csv.DictWriter(out_f, fieldnames=[
            "youtube_id", "label", "year", "transcript",
            "num_words", "duration_sec", "language",
            "n_frames", "mean_brightness", "std_brightness", "mean_edge_density",
        ])
        writer.writeheader()

        for i, it in enumerate(items, start=1):
            print(f"[{i}/{len(items)}] {it.youtube_id}")
            try:
                video_path = download_youtube(it.youtube_id, TMP_VIDEOS)

                # 1) Extract first 10 frames and compute frame features
                frame_dir = TMP_FRAMES / it.youtube_id
                frames = extract_first_n_frames(video_path, frame_dir, n=10, fps=1)
                frame_summary = summarize_first_frames(frames)

                # 2) Audio + transcript
                wav_path = extract_wav(video_path, TMP_AUDIO)
                transcript, stats = transcribe_wav(model, wav_path)

                writer.writerow({
                    "youtube_id": it.youtube_id,
                    "label": it.label,
                    "year": it.year,
                    "transcript": transcript,
                    "num_words": stats["num_words"],
                    "duration_sec": stats["duration"],
                    "language": stats["language"],
                    "n_frames": frame_summary.n_frames,
                    "mean_brightness": frame_summary.mean_brightness,
                    "std_brightness": frame_summary.std_brightness,
                    "mean_edge_density": frame_summary.mean_edge_density,
                })

            except Exception as e:
                fail_f.write(json.dumps({
                    "youtube_id": it.youtube_id,
                    "error": str(e),
                }) + "\n")
                print("  FAILED:", e)

    print(f"\nWrote transcripts+frame feats to: {TRANSCRIPTS_CSV}")
    print(f"Wrote failures to:            {failed_path}")

if __name__ == "__main__":
    main()
