# src/api/app.py
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import joblib
import numpy as np

from src.config import CLF_PATH, ensure_dirs, TMP_VIDEOS, TMP_AUDIO, TMP_FRAMES
from src.media.youtube import download_youtube
from src.media.audio import extract_wav
from src.media.frames import extract_first_n_frames
from src.features.frame_feats import summarize_first_frames
from src.features.text_embed import TextEmbedder
from faster_whisper import WhisperModel

app = FastAPI(title="YC Predictor (Transcript + First-10-Frames)")

class ScoreReq(BaseModel):
    youtube_id: str

class ScoreResp(BaseModel):
    youtube_id: str
    yc_like_probability: float
    label: str
    transcript: str
    frame_features: dict

_clf = None
_asr = None
_embed = None

@app.on_event("startup")
def startup():
    global _clf, _asr, _embed
    ensure_dirs()
    if not CLF_PATH.exists():
        print("WARNING: No trained model found. Train with: python -m src.train.train_text")
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
    ], dtype=np.float32)

@app.post("/score", response_model=ScoreResp)
def score(req: ScoreReq):
    if _clf is None:
        raise HTTPException(status_code=400, detail="Model not trained. Run: python -m src.train.train_text")

    yt = req.youtube_id.strip()
    if not yt:
        raise HTTPException(status_code=400, detail="youtube_id is required.")

    video_path = download_youtube(yt, TMP_VIDEOS)

    # Frames first (matches training)
    frame_dir = TMP_FRAMES / yt
    frames = extract_first_n_frames(video_path, frame_dir, n=10, fps=1)
    frame_summary = summarize_first_frames(frames)

    # Transcript
    wav_path = extract_wav(video_path, TMP_AUDIO)
    transcript = _transcribe_wav(wav_path)
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe audio for this video.")

    X_text = _embed.encode([transcript])
    X_frames = _frame_vec(frame_summary)[None, :]
    X = np.hstack([X_text, X_frames])

    proba = float(_clf.predict_proba(X)[0, 1])
    label = "YC-like" if proba >= 0.5 else "Not YC-like"

    return ScoreResp(
        youtube_id=yt,
        yc_like_probability=proba,
        label=label,
        transcript=transcript,
        frame_features={
            "n_frames": frame_summary.n_frames,
            "mean_brightness": frame_summary.mean_brightness,
            "std_brightness": frame_summary.std_brightness,
            "mean_edge_density": frame_summary.mean_edge_density,
        },
    )
