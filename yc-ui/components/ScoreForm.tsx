"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { parseYouTubeId } from "../lib/parseYoutube";
import { scoreVideo, ScoreResponse } from "../lib/api";

import ProgressTimeline, { StageKey } from "./ProgressTimeline";
import HistoryPanel, { HistoryItem } from "./HistoryPanel";
import ConfidenceBadge from "./ConfidenceBadge";
import ContributionCard from "./ContributionCard";

const HISTORY_KEY = "yc_predictor_history_v1";
const HISTORY_MAX = 12;

function loadHistory(): HistoryItem[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as HistoryItem[];
  } catch {
    return [];
  }
}

function saveHistory(items: HistoryItem[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_MAX)));
  } catch {}
}

export default function ScoreForm() {
  const [input, setInput] = useState("");
  const [youtubeId, setYoutubeId] = useState<string | null>(null);

  const [stage, setStage] = useState<StageKey>("idle");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // Load history once
  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  const stageTimer = useRef<number | null>(null);
  function scheduleStages() {
    // Reset
    if (stageTimer.current) window.clearTimeout(stageTimer.current);

    setStage("download");
    stageTimer.current = window.setTimeout(() => setStage("frames"), 2500);
    stageTimer.current = window.setTimeout(() => setStage("audio"), 4500);
    stageTimer.current = window.setTimeout(() => setStage("asr"), 6500);
    stageTimer.current = window.setTimeout(() => setStage("embed"), 12000);
  }

  function finishStages(success: boolean) {
    if (stageTimer.current) window.clearTimeout(stageTimer.current);
    setStage(success ? "done" : "idle");
  }

  async function onScore() {
    setErr(null);
    setResult(null);

    const id = parseYouTubeId(input);
    setYoutubeId(id);

    if (!id) {
      setErr("Could not parse a valid YouTube ID. Paste an 11-char ID or a YouTube URL.");
      return;
    }

    setLoading(true);
    scheduleStages();

    try {
      const out = await scoreVideo(id);
      setResult(out);
      finishStages(true);

      const next: HistoryItem[] = [
        {
          youtube_id: out.youtube_id,
          probability: out.yc_like_probability,
          confidence_label: out.confidence_label,
          at: Date.now(),
        },
        ...history.filter((h) => h.youtube_id !== out.youtube_id),
      ].slice(0, HISTORY_MAX);

      setHistory(next);
      saveHistory(next);
    } catch (e: any) {
      setErr(e?.message ?? "Scoring failed");
      finishStages(false);
    } finally {
      setLoading(false);
    }
  }

  function onPickFromHistory(id: string) {
    setInput(id);
    setYoutubeId(id);
  }

  function onClearHistory() {
    setHistory([]);
    saveHistory([]);
  }

  const probPct = useMemo(() => {
    if (!result) return null;
    return Math.round(result.yc_like_probability * 1000) / 10; 
  }, [result]);

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

      <ProgressTimeline stage={loading ? stage : "idle"} error={err} />

      {result ? (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div>
              <div className="small">YC-like probability</div>
              <div className="metric">{probPct}%</div>
              <div className="small" style={{ marginTop: 6, opacity: 0.8 }}>
                Raw: {result.yc_like_probability.toFixed(4)}
              </div>
            </div>
            <div style={{ alignSelf: "flex-start" }}>
              <ConfidenceBadge label={result.confidence_label} />
            </div>
          </div>
        </div>
      ) : null}

      {result?.contrib ? <ContributionCard contrib={result.contrib} /> : null}

      <HistoryPanel items={history} onPick={onPickFromHistory} onClear={onClearHistory} />
    </div>
  );
}
