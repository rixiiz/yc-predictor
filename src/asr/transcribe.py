# src/asr/transcribe.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from faster_whisper import WhisperModel

from src.config import DATASET_CSV, TRANSCRIPTS_CSV, ensure_dirs, TMP_VIDEOS, TMP_AUDIO, TMP_FRAMES
from src.media.youtube import download_youtube
from src.media.audio import extract_wav
from src.media.frames import extract_first_n_frames
from src.features.frame_feats import summarize_first_frames


def _read_dataset(csv_path: Path) -> List[Dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "youtube_id": r["youtube_id"].strip(),
                "label": int(r["label"]),
                "year": int(r["year"]),
            })
    return rows


def _transcribe_whisper(model: WhisperModel, wav_path: Path) -> str:
    segments, _info = model.transcribe(str(wav_path), vad_filter=True)
    parts = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts)


def main():
    ensure_dirs()
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATASET_CSV}")

    dataset = _read_dataset(DATASET_CSV)
    PROCESSED_DIR = TRANSCRIPTS_CSV.parent
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    failed_path = PROCESSED_DIR / "transcribe_failed.jsonl"

    asr = WhisperModel("base", device="cpu", compute_type="int8")

    fieldnames = [
        "youtube_id", "label", "year",
        "transcript",
        "mean_brightness", "std_brightness", "mean_edge_density", "mean_motion_energy", "n_frames",
    ]

    # Write fresh file each run
    with TRANSCRIPTS_CSV.open("w", encoding="utf-8", newline="") as out_f, \
         failed_path.open("w", encoding="utf-8") as fail_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for item in dataset:
            yt = item["youtube_id"]
            try:
                video_path = download_youtube(yt, TMP_VIDEOS)

                frame_dir = TMP_FRAMES / yt
                frames = extract_first_n_frames(video_path, frame_dir, n=10, fps=1)
                frame_summary = summarize_first_frames(frames)

                wav_path = extract_wav(video_path, TMP_AUDIO)
                transcript = _transcribe_whisper(asr, wav_path)

                writer.writerow({
                    "youtube_id": yt,
                    "label": item["label"],
                    "year": item["year"],
                    "transcript": transcript,
                    "mean_brightness": frame_summary.mean_brightness,
                    "std_brightness": frame_summary.std_brightness,
                    "mean_edge_density": frame_summary.mean_edge_density,
                    "mean_motion_energy": frame_summary.mean_motion_energy,
                    "n_frames": frame_summary.n_frames,
                })

            except Exception as e:
                fail_f.write(json.dumps({"youtube_id": yt, "error": str(e)}, ensure_ascii=False) + "\n")

    print(f"Wrote: {TRANSCRIPTS_CSV}")
    print(f"Failures: {failed_path}")


if __name__ == "__main__":
    main()