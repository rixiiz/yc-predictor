# YC Predictor

YC Predictor is an end-to-end **multimodal machine learning system** that predicts the likelihood a startup pitch video would be accepted by Y Combinator evaluators.
The system analyzes **what founders say** (speech transcripts) and **what the video looks like** (first 10 visual frames), combines these signals into a single model, and returns an interpretable probability score via a clean TypeScript UI.

---

## Key Features

- **Multimodal ML pipeline** combining speech and visual features
- **Automated data scraping & processing** from https://ycarena.com
- **Whisper-based transcription** for high-quality speech-to-text
- **Sentence-transformer embeddings** for semantic understanding
- **Interpretable logistic regression model** with feature contribution breakdowns
- **FastAPI inference backend**
- **TypeScript (Next.js) frontend** with:
  - Progress timeline
  - Confidence interpretation
  - Feature contribution summary
  - Local prediction history

---

## Model Overview

**Inputs**
- Startup pitch YouTube video (ID or URL)

**Features**
- **Text**: Transcript embeddings generated using sentence transformers
- **Visual**: Statistics from the first 10 video frames (brightness, contrast, edge density)

**Model**
- Logistic Regression classifier trained on labeled pitch outcomes
- Outputs a probability score and interpretable logit-space contributions:
  - Text contribution
  - Visual contribution
  - Intercept (base rate)

---

## Example Output

- **YC-like probability**: `78.3%`
- **Confidence label**: `Strong signal`
- **Feature contribution summary**:
  - Text embedding: `+0.31`
  - Frame features: `+0.07`
  - Intercept: `+0.12`

---

## 🗂️ Project Structure

YCpredictor/
├── src/ # Python ML + backend code
│ ├── api/ # FastAPI inference service
│ ├── asr/ # Transcription pipeline (Whisper)
│ ├── features/ # Text & frame feature extraction
│ ├── media/ # YouTube download, audio, frames
│ └── train/ # Model training
├── data/
│ ├── raw/ # Original dataset (CSV)
│ ├── processed/ # Transcripts + features
│ └── models/ # Trained model artifacts
├── tmp/ # Temporary video/audio/frame files (ignored)
├── yc-ui/ # TypeScript UI (Next.js)
│ ├── app/ # App Router pages/layout
│ ├── components/ # UI components
│ └── lib/ # API helpers
├── requirements.txt
├── .gitignore
└── README.md

---

## Dataset

- Pitch videos scraped from YouTube using `yt-dlp`
- Labeled as **1(accepted) / 0(rejected)**
- Stored as a CSV with:
  ```csv
  youtube_id,label,year

The dataset is processed automatically into transcripts and feature vectors before training.

## Running the Project

### System Requirements

- Python 3.10+
- Node.js 18+
- **ffmpeg** installed and on PATH
- Git

### Backend Setup (Python)

`git clone https://github.com/rixiiz/yc-predictor`
`cd YCpredictor`
`python -m venv .venv`
`source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1`
`pip install -r requirements.txt`

### Process Data & Train Model

`python -m src.asr.transcribe`
`python -m src.train.train_text`

This will generate:
- `data/processed/transcripts.csv`
- `data/models/text_clf.joblib`

### Start Backend API

`uvicorn src.api.app:app --reload --port 8000`

### Frontend Setup (TypeScript UI)
`cd yc-ui`
`npm install`

Create `yc-ui/.env.local`: 
`NEXT_PUBLIC_API_BASE=http://localhost:8000`

Start UI:
`npm run dev`

## Notes

- Temporary files (`tmp/`) are not committed
- Model artifacts are reproducible and not versioned
- Some YouTube videos may fail due to availability or rate limits
