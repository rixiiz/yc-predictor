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

YCpredictor/<br>
├── src/ # Python ML + backend code<br>
│ ├── api/ # FastAPI inference service<br>
│ ├── asr/ # Transcription pipeline (Whisper)<br>
│ ├── features/ # Text & frame feature extraction<br>
│ ├── media/ # YouTube download, audio, frames<br>
│ └── train/ # Model training<br>
├── data/<br>
│ ├── raw/ # Original dataset (CSV)<br>
│ ├── processed/ # Transcripts + features<br>
│ └── models/ # Trained model artifacts<br>
├── tmp/ # Temporary video/audio/frame files (ignored)<br>
├── yc-ui/ # TypeScript UI (Next.js)<br>
│ ├── app/ # App Router pages/layout<br>
│ ├── components/ # UI components<br>
│ └── lib/ # API helpers<br>
├── requirements.txt<br>
├── .gitignore<br>
└── README.md<br>

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

### Start Backend API

`uvicorn src.api.app:app --reload --port 8000`

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

