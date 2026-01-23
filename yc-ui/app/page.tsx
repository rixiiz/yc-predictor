import ScoreForm from "../components/ScoreForm";

export default function Page() {
  return (
    <main className="card">
      <div className="badge">TypeScript UI + FastAPI backend</div>
      <h1 style={{ marginTop: 14, marginBottom: 8 }}>YC Predictor</h1>
      <p className="small" style={{ marginTop: 0 }}>
        Input a YouTube video's <b>ID</b> or <b>URL</b>. The backend downloads the video, extracts the first 10 frames,
        transcribes audio, embeds text + frame features, and returns a probability.
      </p>

      <ScoreForm />
    </main>
  );
}
