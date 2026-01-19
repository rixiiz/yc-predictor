# src/train/train_text.py
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

from src.config import TRANSCRIPTS_CSV, CLF_PATH, META_PATH, ensure_dirs
from src.features.text_embed import TextEmbedder
from src.train.split import choose_year_holdout
from src.train.eval import compute_metrics, metrics_to_dict

def load_transcripts(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            t = (r.get("transcript") or "").strip()
            if not t:
                continue
            rows.append({
                "youtube_id": r["youtube_id"],
                "label": int(r["label"]),
                "year": int(r["year"]),
                "transcript": t,
                "n_frames": int(float(r.get("n_frames", 0) or 0)),
                "mean_brightness": float(r.get("mean_brightness", 0.0) or 0.0),
                "std_brightness": float(r.get("std_brightness", 0.0) or 0.0),
                "mean_edge_density": float(r.get("mean_edge_density", 0.0) or 0.0),
            })
    return rows

def frame_vec(r) -> np.ndarray:
    return np.array([
        float(r.get("mean_brightness", 0.0)),
        float(r.get("std_brightness", 0.0)),
        float(r.get("mean_edge_density", 0.0)),
        float(r.get("n_frames", 0.0)),
    ], dtype=np.float32)

def main():
    ensure_dirs()
    if not TRANSCRIPTS_CSV.exists():
        raise FileNotFoundError(f"Missing {TRANSCRIPTS_CSV}. Run: python -m src.asr.transcribe")

    rows = load_transcripts(TRANSCRIPTS_CSV)
    if len(rows) < 20:
        raise ValueError(f"Too few transcripts ({len(rows)}).")

    holdout_year = choose_year_holdout(TRANSCRIPTS_CSV)
    train_rows = [r for r in rows if r["year"] != holdout_year]
    test_rows  = [r for r in rows if r["year"] == holdout_year]

    if len(test_rows) < 5 or len(train_rows) < 10:
        # fallback: last 20% as test
        rows_sorted = sorted(rows, key=lambda x: (x["year"], x["youtube_id"]))
        cut = int(len(rows_sorted) * 0.8)
        train_rows, test_rows = rows_sorted[:cut], rows_sorted[cut:]
        holdout_year = -1

    print(f"Train: {len(train_rows)}  Test: {len(test_rows)}  (holdout_year={holdout_year})")

    embedder = TextEmbedder(device="cpu")

    # Text embeddings
    X_train_text = embedder.encode([r["transcript"] for r in train_rows])
    X_test_text  = embedder.encode([r["transcript"] for r in test_rows])

    # Frame features
    X_train_frames = np.vstack([frame_vec(r) for r in train_rows])
    X_test_frames  = np.vstack([frame_vec(r) for r in test_rows])

    # Concatenate
    X_train = np.hstack([X_train_text, X_train_frames])
    X_test  = np.hstack([X_test_text, X_test_frames])

    y_train = np.array([r["label"] for r in train_rows], dtype=np.int64)
    y_test  = np.array([r["label"] for r in test_rows], dtype=np.int64)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, proba)

    print("Metrics:", metrics_to_dict(m))

    CLF_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLF_PATH)

    meta = {
        "model": "LogReg + all-MiniLM-L6-v2 embeddings + first-10-frame features",
        "frame_features": ["mean_brightness", "std_brightness", "mean_edge_density", "n_frames"],
        "split": {"holdout_year": holdout_year, "train_n": len(train_rows), "test_n": len(test_rows)},
        "metrics": metrics_to_dict(m),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {CLF_PATH}")
    print(f"Saved meta:  {META_PATH}")

if __name__ == "__main__":
    main()
