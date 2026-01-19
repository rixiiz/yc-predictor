# streamlit_app.py
import re
from pathlib import Path
import numpy as np
import streamlit as st
import joblib
from faster_whisper import WhisperModel

from src.config import ensure_dirs, CLF_PATH, TMP_VIDEOS, TMP_AUDIO, TMP_FRAMES
from src.media.youtube import download_youtube
from src.media.audio import extract_wav
from src.media.frames import extract_first_n_frames
from src.features.frame_feats import summarize_first_frames
from src.features.text_embed import TextEmbedder


YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")

def parse_youtube_id(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return text
    m = YOUTUBE_ID_RE.search(text)
    return m.group(1) if m else None

def transcribe(asr: WhisperModel, wav_path: Path) -> str:
    segments, _ = asr.transcribe(str(wav_path), vad_filter=True)
    parts = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts)

def frame_vec(summary) -> np.ndarray:
    return np.array([
        float(summary.mean_brightness),
        float(summary.std_brightness),
        float(summary.mean_edge_density),
        float(summary.n_frames),
    ], dtype=np.float32)[None, :]


# ---------------- UI ----------------
st.set_page_config(page_title="YC Predictor", page_icon="📈", layout="centered")
st.title("📈 YC Predictor")
st.caption("Input: YouTube ID or URL → Download → First 10 frames + transcript → Model → Probability")

ensure_dirs()

@st.cache_resource
def load_runtime():
    if not CLF_PATH.exists():
        raise FileNotFoundError(f"Model not found at {CLF_PATH}. Train it first: python -m src.train.train_text")
    clf = joblib.load(CLF_PATH)
    embedder = TextEmbedder(device="cpu")
    asr = WhisperModel("base", device="cpu", compute_type="int8")
    return clf, embedder, asr

try:
    clf, embedder, asr = load_runtime()
except Exception as e:
    st.error(str(e))
    st.stop()

user_input = st.text_input(
    "YouTube video ID or URL",
    placeholder="e.g. vtdm40KJyO4 or https://www.youtube.com/watch?v=vtdm40KJyO4",
)

colA, colB = st.columns([1, 1])
with colA:
    cleanup_tmp = st.checkbox("Delete temp files after scoring", value=False)
with colB:
    score_btn = st.button("Score", type="primary")

if score_btn:
    yt_id = parse_youtube_id(user_input)
    if not yt_id:
        st.error("Could not parse a valid YouTube ID. Paste an 11-char ID or a full YouTube URL.")
        st.stop()

    video_path = None
    wav_path = None
    frame_dir = TMP_FRAMES / yt_id

    progress = st.progress(0)
    status = st.empty()

    try:
        status.write("Downloading video…")
        progress.progress(15)
        video_path = download_youtube(yt_id, TMP_VIDEOS)

        status.write("Extracting first 10 frames…")
        progress.progress(35)
        frame_paths = extract_first_n_frames(video_path, frame_dir, n=10, fps=1)
        frame_summary = summarize_first_frames(frame_paths)

        status.write("Extracting audio…")
        progress.progress(50)
        wav_path = extract_wav(video_path, TMP_AUDIO)

        status.write("Transcribing…")
        progress.progress(70)
        transcript = transcribe(asr, wav_path)
        if not transcript.strip():
            raise RuntimeError("Transcript is empty. Try another video.")

        status.write("Running model…")
        progress.progress(90)
        X_text = embedder.encode([transcript])
        X_frames = frame_vec(frame_summary)
        X = np.hstack([X_text, X_frames])

        proba = float(clf.predict_proba(X)[0, 1])
        progress.progress(100)
        status.write("Done.")

        st.metric("YC-like probability", f"{proba:.3f}")

    except Exception as e:
        st.error(f"Failed to score `{yt_id}`: {e}")

    finally:
        if cleanup_tmp:
            try:
                if wav_path and wav_path.exists():
                    wav_path.unlink()
            except Exception:
                pass

            try:
                if video_path and video_path.exists():
                    video_path.unlink()
            except Exception:
                pass

            try:
                if frame_dir.exists():
                    for p in frame_dir.glob("*.png"):
                        try:
                            p.unlink()
                        except Exception:
                            pass
                    frame_dir.rmdir()
            except Exception:
                pass
