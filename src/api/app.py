from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import numpy as np
from faster_whisper import WhisperModel

from src.config import CLF_PATH, ensure_dirs, TMP_VIDEOS, TMP_AUDIO, TMP_FRAMES
from src.media.youtube import download_youtube
from src.media.audio import extract_wav
from src.media.frames import extract_first_n_frames
from src.features.frame_feats import summarize_first_frames
from src.features.text_embed import TextEmbedder

app = FastAPI(title="YC Predictor API")

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

class ScoreResp(BaseModel):
    youtube_id: str
    yc_like_probability: float
    confidence_label: str
    contrib: Contrib

_clf = None
_asr = None
_embed = None

@app.on_event("startup")
def startup():
    global _clf, _asr, _embed
    ensure_dirs()
    if not CLF_PATH.exists():
        print("WARNING: No trained model found. Train with: python -m src.train.train_text")
        _clf = None
    else:
        _clf = joblib.load(CLF_PATH)
    _asr = WhisperModel("base", device="cpu", compute_type="int8")
    _embed = TextEmbedder(device="cpu")

def _transcribe_wav(wav_path: Path) -> str:
    segments, _info = _asr.transcribe(str(wav_path), vad_filter=True)
    parts = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts)

def _frame_vec(summary) -> np.ndarray:
    return np.array([
        float(summary.mean_brightness),
        float(summary.std_brightness),
        float(summary.mean_edge_density),
        float(summary.n_frames),
    ], dtype=np.float32)[None, :]

def _confidence_label(p: float) -> str:
    if p >= 0.85:
        return "Very YC-like"
    if p >= 0.70:
        return "Strong signal"
    if p >= 0.55:
        return "Borderline"
    return "Unlikely"

@app.post("/score", response_model=ScoreResp)
def score(req: ScoreReq):
    if _clf is None:
        raise HTTPException(status_code=400, detail="Model not trained. Run: python -m src.train.train_text")

    yt = (req.youtube_id or "").strip()
    if not yt:
        raise HTTPException(status_code=400, detail="youtube_id is required")

    video_path = download_youtube(yt, TMP_VIDEOS)

    frame_dir = TMP_FRAMES / yt
    frames = extract_first_n_frames(video_path, frame_dir, n=10, fps=1)
    frame_summary = summarize_first_frames(frames)

    wav_path = extract_wav(video_path, TMP_AUDIO)
    transcript = _transcribe_wav(wav_path)
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe audio for this video")

    X_text = _embed.encode([transcript])          # (1, 384)
    X_frames = _frame_vec(frame_summary)          # (1, 4)
    X = np.hstack([X_text, X_frames])             # (1, 388)

    proba = float(_clf.predict_proba(X)[0, 1])

    # Logistic regression contributions (logit space)
    # total_logit = intercept + sum_i coef_i * x_i
    coef = _clf.coef_.reshape(-1)                 # (388,)
    intercept = float(_clf.intercept_.reshape(-1)[0])

    x = X.reshape(-1)                             # (388,)
    text_logit = float(np.dot(coef[:384], x[:384]))
    frames_logit = float(np.dot(cof := coef[384:], x[384:]))
    total_logit = intercept + text_logit + frames_logit

    return ScoreResp(
        youtube_id=yt,
        yc_like_probability=proba,
        confidence_label=_confidence_label(proba),
        contrib=Contrib(
            intercept=intercept,
            text_logit=text_logit,
            frames_logit=frames_logit,
            total_logit=total_logit,
        )
    )
