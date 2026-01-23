from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import torch
from faster_whisper import WhisperModel

from src.config import NN_PATH, META_PATH, ensure_dirs, TMP_VIDEOS, TMP_AUDIO, TMP_FRAMES
from src.media.youtube import download_youtube
from src.media.audio import extract_wav
from src.media.frames import extract_first_n_frames
from src.features.frame_feats import summarize_first_frames
from src.features.text_embed import TextEmbedder
from src.train.nn_model import YCMLP

app = FastAPI(title="YC Predictor (NN: Transcript + First-10-Frames)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScoreReq(BaseModel):
    youtube_id: str

class Contrib(BaseModel):
    intercept: float
    text_logit: float
    frames_logit: float
    total_logit: float
    p_base: float
    p_text_only: float
    p_frames_only: float
    p_full: float

class ScoreResp(BaseModel):
    youtube_id: str
    yc_like_probability: float
    confidence_label: str
    label: str
    transcript: str
    frame_features: dict
    contrib: Contrib

_model = None
_asr = None
_embed = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

_num_keys = None
_mu = None
_sd = None

def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))

def confidence_label(p: float) -> str:
    if p >= 0.85:
        return "Very YC-like"
    if p >= 0.70:
        return "Strong signal"
    if p >= 0.55:
        return "Borderline"
    return "Unlikely"

@app.on_event("startup")
def startup():
    global _model, _asr, _embed, _num_keys, _mu, _sd

    ensure_dirs()

    if not NN_PATH.exists() or not META_PATH.exists():
        print("WARNING: Model/meta missing. Train with: python3 -m src.train.train_text")
        _model = None
    else:
        ckpt = torch.load(NN_PATH, map_location="cpu")
        _model = YCMLP(
            input_dim=int(ckpt["input_dim"]),
            hidden_dims=list(ckpt.get("hidden_dims", [128, 32])),
            dropout=float(ckpt.get("dropout", 0.5)),
        )
        _model.load_state_dict(ckpt["state_dict"])
        _model.to(_device).eval()

        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        _num_keys = list(meta["numeric_features"])
        _mu = np.array(meta["numeric_scaler"]["mean"], dtype=np.float32)
        _sd = np.array(meta["numeric_scaler"]["std"], dtype=np.float32)
        _sd = np.where(_sd < 1e-6, 1.0, _sd)

    _asr = WhisperModel("base", device="cpu", compute_type="int8")
    _embed = TextEmbedder(device="cpu")

def transcribe_wav(wav_path: Path) -> str:
    segments, _ = _asr.transcribe(str(wav_path), vad_filter=True)
    parts = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts)

def _predict_logit(X: np.ndarray) -> float:
    with torch.no_grad():
        return float(_model(torch.from_numpy(X).to(_device)).item())

@app.post("/score", response_model=ScoreResp)
def score(req: ScoreReq):
    if _model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Run: python3 -m src.train.train_text")

    yt = (req.youtube_id or "").strip()
    if not yt:
        raise HTTPException(status_code=400, detail="youtube_id is required.")

    # Download + preprocess
    video_path = download_youtube(yt, TMP_VIDEOS)

    frame_dir = TMP_FRAMES / yt
    frames = extract_first_n_frames(video_path, frame_dir, n=10, fps=1)
    frame_summary = summarize_first_frames(frames)

    wav_path = extract_wav(video_path, TMP_AUDIO)
    transcript = transcribe_wav(wav_path)
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe audio for this video.")

    # Build features
    X_text = _embed.encode([transcript]).astype(np.float32)  # (1, 384)

    feats = {
        "mean_brightness": float(getattr(frame_summary, "mean_brightness", 0.0)),
        "std_brightness": float(getattr(frame_summary, "std_brightness", 0.0)),
        "mean_edge_density": float(getattr(frame_summary, "mean_edge_density", 0.0)),
        "mean_motion_energy": float(getattr(frame_summary, "mean_motion_energy", 0.0)),
        "n_frames": float(getattr(frame_summary, "n_frames", 0.0)),
    }

    x_num_raw = np.array([[float(feats.get(k, 0.0)) for k in _num_keys]], dtype=np.float32)
    x_num = (x_num_raw - _mu) / _sd  # (1, k)

    # Full input
    X_full = np.hstack([X_text, x_num]).astype(np.float32)

    # Ablations for contribution summary
    X_base = np.zeros_like(X_full, dtype=np.float32)
    X_text_only = X_base.copy()
    X_text_only[:, : X_text.shape[1]] = X_text
    X_frames_only = X_base.copy()
    X_frames_only[:, X_text.shape[1] :] = x_num

    # Predict
    logit_full = _predict_logit(X_full)
    p_full = sigmoid(logit_full)

    logit_base = _predict_logit(X_base)
    p_base = sigmoid(logit_base)

    logit_text_only = _predict_logit(X_text_only)
    p_text_only = sigmoid(logit_text_only)

    logit_frames_only = _predict_logit(X_frames_only)
    p_frames_only = sigmoid(logit_frames_only)

    # Logit contributions relative to baseline
    intercept = logit(p_base)
    text_logit = logit(p_text_only) - intercept
    frames_logit = logit(p_frames_only) - intercept
    total_logit = logit(p_full) - intercept

    label = "YC-like" if p_full >= 0.50 else "Not YC-like"

    return ScoreResp(
        youtube_id=yt,
        yc_like_probability=p_full,
        confidence_label=confidence_label(p_full),
        label=label,
        transcript=transcript,
        frame_features=feats,
        contrib=Contrib(
            intercept=float(intercept),
            text_logit=float(text_logit),
            frames_logit=float(frames_logit),
            total_logit=float(total_logit),
            p_base=float(p_base),
            p_text_only=float(p_text_only),
            p_frames_only=float(p_frames_only),
            p_full=float(p_full),
        ),
    )
