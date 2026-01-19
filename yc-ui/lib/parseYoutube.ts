export function parseYouTubeId(input: string): string | null {
  const s = (input || "").trim();
  if (!s) return null;

  if (/^[A-Za-z0-9_-]{11}$/.test(s)) return s;

  const m =
    s.match(/[?&]v=([A-Za-z0-9_-]{11})/) ||
    s.match(/youtu\.be\/([A-Za-z0-9_-]{11})/) ||
    s.match(/shorts\/([A-Za-z0-9_-]{11})/);

  return m ? m[1] : null;
}
