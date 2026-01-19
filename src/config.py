from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

TMP_DIR = ROOT / "tmp"
TMP_VIDEOS = TMP_DIR / "videos"
TMP_AUDIO = TMP_DIR / "audio"
TMP_FRAMES = TMP_DIR / "frames"

DATASET_CSV = RAW_DIR / "dataset.csv"
TRANSCRIPTS_CSV = PROCESSED_DIR / "transcripts.csv"

CLF_PATH = MODELS_DIR / "text_clf.joblib"
META_PATH = MODELS_DIR / "meta.json"

def ensure_dirs() -> None:
    for p in [PROCESSED_DIR, MODELS_DIR, TMP_VIDEOS, TMP_AUDIO, TMP_FRAMES]:
        p.mkdir(parents=True, exist_ok=True)
