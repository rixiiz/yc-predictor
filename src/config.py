# src/config.py
from pathlib import Path

# project/src/config.py  -> parents[1] = project/
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
DATASET_TEST_CSV = RAW_DIR / "dataset_test.csv"
TRANSCRIPTS_CSV = PROCESSED_DIR / "transcripts.csv"

# Neural net artifacts
NN_PATH = MODELS_DIR / "text_mlp.pt"
META_PATH = MODELS_DIR / "meta.json"

def ensure_dirs() -> None:
    for p in [
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        TMP_VIDEOS,
        TMP_AUDIO,
        TMP_FRAMES,
    ]:
        p.mkdir(parents=True, exist_ok=True)