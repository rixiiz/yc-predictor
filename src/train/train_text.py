# src/train/train_text.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import TRANSCRIPTS_CSV, NN_PATH, META_PATH, ensure_dirs
from src.features.text_embed import TextEmbedder
from src.train.nn_model import YCMLP

# numeric features from first 10 frames (must exist as columns in transcripts.csv; missing => 0.0)
NUM_KEYS = ["mean_brightness", "std_brightness", "mean_edge_density", "mean_motion_energy", "n_frames"]


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            t = (r.get("transcript") or "").strip()
            if not t:
                continue
            rows.append(
                {
                    "youtube_id": r.get("youtube_id", ""),
                    "label": int(r.get("label", 0)),
                    "year": int(float(r.get("year", 0) or 0)),
                    "transcript": t,
                    **{k: float(r.get(k, 0.0) or 0.0) for k in NUM_KEYS},
                }
            )
    return rows


def zfit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return mu, sd


def zapply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def main():
    ensure_dirs()

    if not TRANSCRIPTS_CSV.exists():
        raise FileNotFoundError(f"Missing {TRANSCRIPTS_CSV}. Run: python3 -m src.asr.transcribe")

    rows = load_rows(TRANSCRIPTS_CSV)
    if len(rows) < 10:
        raise ValueError(f"Too few transcripts ({len(rows)}).")

    y = np.array([r["label"] for r in rows], dtype=int)

    # Text embeddings
    embedder = TextEmbedder(device="cpu")
    X_text = embedder.encode([r["transcript"] for r in rows]).astype(np.float32)

    # Numeric features + zscore
    X_num_raw = np.array([[r.get(k, 0.0) for k in NUM_KEYS] for r in rows], dtype=np.float32)
    mu, sd = zfit(X_num_raw)
    X_num = zapply(X_num_raw, mu, sd)

    # Final features
    X = np.hstack([X_text, X_num]).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simple MLP
    model = YCMLP(input_dim=X.shape[1], hidden_dims=[128, 32], dropout=0.5).to(device)

    # Handle imbalance with pos_weight only
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    model.train()
    for _epoch in range(40):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Save model + meta (predict.py does threshold selection on test slice itself)
    NN_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "YCMLP",
            "input_dim": int(X.shape[1]),
            "hidden_dims": [128, 32],
            "dropout": 0.5,
            "state_dict": model.state_dict(),
        },
        NN_PATH,
    )

    meta = {
        "numeric_features": NUM_KEYS,
        "numeric_scaler": {"mean": mu.tolist(), "std": sd.tolist()},
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {NN_PATH}")
    print(f"Saved meta:  {META_PATH}")


if __name__ == "__main__":
    main()