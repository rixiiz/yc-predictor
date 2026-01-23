export type Contrib = {
  intercept: number;
  text_logit: number;
  frames_logit: number;
  total_logit: number;
  p_base: number;
  p_text_only: number;
  p_frames_only: number;
  p_full: number;
};

export type ScoreResponse = {
  youtube_id: string;
  yc_like_probability: number;
  confidence_label: string;
  label: string;
  transcript: string;
  frame_features: Record<string, number>;
  contrib: Contrib;
};

export async function scoreVideo(youtubeId: string): Promise<ScoreResponse> {
  const base = process.env.NEXT_PUBLIC_API_BASE;
  if (!base) throw new Error("NEXT_PUBLIC_API_BASE is not set in .env.local");

  const res = await fetch(`${base}/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ youtube_id: youtubeId }),
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) detail = String(data.detail);
    } catch {}
    throw new Error(detail);
  }

  return (await res.json()) as ScoreResponse;
}
