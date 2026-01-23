"use client";

import { useState } from "react";
import { parseYouTubeId } from "../lib/parseYoutube";
import { scoreVideo } from "../lib/api";
import ProbabilityCard from "./ProbabilityCard";

export default function ScoreForm() {
  const [input, setInput] = useState("");
  const [youtubeId, setYoutubeId] = useState<string | null>(null);
  const [prob, setProb] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function onScore() {
    setErr(null);
    setProb(null);

    const id = parseYouTubeId(input);
    setYoutubeId(id);

    if (!id) {
      setErr("Could not parse a valid YouTube ID. Paste an 11-char ID or a YouTube URL.");
      return;
    }

    setLoading(true);
    try {
      const out = await scoreVideo(id);
      setProb(out.yc_like_probability);
    } catch (e: any) {
      setErr(e?.message ?? "Scoring failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ marginTop: 18 }}>
      <div className="row">
        <input
          className="input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="YouTube ID or URL (e.g. vtdm40KJyO4 or https://www.youtube.com/watch?v=vtdm40KJyO4)"
        />
        <button className="button" onClick={onScore} disabled={loading}>
          {loading ? "Scoring..." : "Score"}
        </button>
      </div>

      <div className="small" style={{ marginTop: 10 }}>
        Parsed ID: <b>{youtubeId ?? "—"}</b>
      </div>

      {err && (
        <div className="error" style={{ marginTop: 14 }}>
          {err}
        </div>
      )}

      {prob !== null && <ProbabilityCard p={prob} />}
    </div>
  );
}
