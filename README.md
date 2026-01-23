# YC Predictor

YC Predictor is an end-to-end **multimodal machine learning system** that predicts the likelihood a startup pitch video would be accepted by Y Combinator evaluators.  
The system analyzes **what founders say** (speech transcripts) and **what the video looks like** (first 10 visual frames), combines these signals into a single model, and returns a probability score via a FastAPI API.

---

## Key Features

- **Multimodal ML pipeline** combining speech and visual features
- **Automated video ingestion & processing** using `yt-dlp` + `ffmpeg`
- **Whisper-based transcription** (`faster-whisper`) for high-quality speech-to-text
- **Sentence-transformer embeddings** for semantic understanding of transcripts
- **PyTorch neural network (MLP)** trained on fused text + frame features
- **FastAPI inference backend** returning:
  - Probability score (`yc_like_probability`)
  - Predicted label (`YC-like` / `Not YC-like`)
  - Transcript + frame features for debugging

---

## Model Overview

**Inputs**
- Startup pitch YouTube video (ID)

**Features**
- **Text**: Transcript embeddings generated using sentence transformers  
- **Visual**: Statistics from the first 10 video frames (brightness, edge density, motion)

**Model**
- PyTorch MLP classifier trained on labeled pitch outcomes  
- Outputs a probability score

---

## Example Evaluation Result

Quick test on the last 20% split:

- **Threshold**: `0.50`
- **Accuracy**: `0.905`
- **Balanced Accuracy**: `0.869`

---

## 🗂️ Project Structure

yc-predictor/<br>
├── src/ # Python ML + backend code<br>
│ ├── api/ # FastAPI inference service<br>
│ ├── asr/ # Transcription pipeline (Whisper)<br>
│ ├── features/ # Text & frame feature extraction<br>
│ ├── media/ # YouTube download, audio, frames<br>
│ └── train/ # Model training + evaluation<br>
├── data/<br>
│ ├── raw/ # Original dataset (CSV)<br>
│ ├── processed/ # Transcripts + features<br>
│ └── models/ # Trained model artifacts<br>
├── tmp/ # Temporary video/audio/frame files (ignored)<br>
├── requirements.txt<br>
├── .gitignore<br>
├── data-scraper/<br>
└── README.md<br>

---

## Dataset

- Pitch videos collected from YouTube using `yt-dlp`
- Labeled as **1(accepted) / 0(rejected)**
- Stored as a CSV with:

csv - `youtube_id,label,year`

## Running the Project

### System Requirements

- Python 3.10+
- Node.js 18+
- **ffmpeg** installed and on PATH
- Git

### Backend Setup (Python)

`git clone https://github.com/rixiiz/yc-predictor`<br>
`cd YCpredictor`<br>
`python -m venv .venv`<br>
`source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1`<br>
`pip install -r requirements.txt`<br>

### Process Data & Train Model

`python -m src.asr.transcribe`<br>
`python -m src.train.train_text`<br>

This will generate:
- `data/processed/transcripts.csv`<br>
- `data/models/text_clf.joblib`<br>

### Evaluation
`python -m src.train.predict`<br>

### Start Backend API

`python -m uvicorn src.api.app:app --reload --port 8000`

### Frontend Setup (TypeScript UI)
`cd yc-ui`<br>
`npm install`<br>

Create `yc-ui/.env.local`:<br> 
`NEXT_PUBLIC_API_BASE=http://localhost:8000`

Start UI:<br>
`npm run dev`

## Notes

- Temporary files (`tmp/`) are not committed
- Model artifacts are reproducible and not versioned
- Some YouTube videos may fail due to availability or rate limits
