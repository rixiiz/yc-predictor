# src/train/predict.py
import csv
import json
import numpy as np
import torch

from src.config import TRANSCRIPTS_CSV, NN_PATH, META_PATH
from src.features.text_embed import TextEmbedder
from src.train.nn_model import YCMLP


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, thr: float):
    pred = (p >= thr).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))

    tnr = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (rec + tnr)

    return {
        "thr": float(thr),
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "bal_acc": float(bal_acc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_threshold_by_f1(y: np.ndarray, p: np.ndarray):
    best = None
    for thr in np.linspace(0.05, 0.95, 19):
        m = metrics_at_threshold(y, p, float(thr))
        if best is None or m["f1"] > best["f1"]:
            best = m
    return best


def main():
    if not TRANSCRIPTS_CSV.exists():
        raise FileNotFoundError(f"Missing {TRANSCRIPTS_CSV}. Run: python3 -m src.asr.transcribe")
    if not NN_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Missing trained model. Run: python3 -m src.train.train_text")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    num_keys = meta["numeric_features"]
    mu = np.array(meta["numeric_scaler"]["mean"], dtype=np.float32)
    sd = np.array(meta["numeric_scaler"]["std"], dtype=np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd)

    ckpt = torch.load(NN_PATH, map_location="cpu")
    model = YCMLP(
        input_dim=int(ckpt["input_dim"]),
        hidden_dims=list(ckpt.get("hidden_dims", [128, 32])),
        dropout=float(ckpt.get("dropout", 0.5)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rows = []
    with TRANSCRIPTS_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("transcript") or "").strip():
                rows.append(r)

    if len(rows) < 10:
        raise ValueError("Too few rows to test.")

    # Quick test = last 20% (sorted by year, youtube_id)
    rows_sorted = sorted(rows, key=lambda x: (int(float(x.get("year", 0) or 0)), x.get("youtube_id", "")))
    cut = int(len(rows_sorted) * 0.8)
    test_rows = rows_sorted[cut:]

    embedder = TextEmbedder(device="cpu")
    X_text = embedder.encode([r["transcript"] for r in test_rows]).astype(np.float32)

    X_num_raw = np.array(
        [[float(r.get(k, 0.0) or 0.0) for k in num_keys] for r in test_rows],
        dtype=np.float32,
    )
    X_num = (X_num_raw - mu) / sd

    X = np.hstack([X_text, X_num]).astype(np.float32)
    y = np.array([int(r["label"]) for r in test_rows], dtype=int)

    with torch.no_grad():
        logits = model(torch.from_numpy(X)).numpy()
        p = sigmoid(logits)

    best = best_threshold_by_f1(y, p)

    print(f"Quick test on last 20%: {len(y)} samples")
    print(f"Class balance: positives={int((y==1).sum())} negatives={int((y==0).sum())}\n")

    print("Best threshold by F1:")
    print(
        f"  thr={best['thr']:.2f}  acc={best['acc']:.3f}  bal_acc={best['bal_acc']:.3f}  "
        f"prec={best['prec']:.3f}  rec={best['rec']:.3f}  f1={best['f1']:.3f}"
    )
    print(f"  Confusion: TP={best['tp']} TN={best['tn']} FP={best['fp']} FN={best['fn']}")

    out = []
    for r, pi in zip(test_rows, p):
        yi = int(r["label"])
        out.append((abs(yi - float(pi)), r.get("youtube_id", ""), int(float(r.get("year", 0) or 0)), yi, float(pi)))
    out.sort(reverse=True)

    print("\nWorst 10 errors (biggest |y - p|):")
    for e, vid, year, yi, pi in out[:10]:
        print(f"{vid}  year={year}  label={yi}  prob={pi:.3f}")


if __name__ == "__main__":
    main()