"use client";

export type HistoryItem = {
  youtube_id: string;
  probability: number;
  confidence_label: string;
  at: number; // epoch ms
};

function fmtTime(ms: number) {
  const d = new Date(ms);
  return d.toLocaleString();
}

export default function HistoryPanel({
  items,
  onPick,
  onClear,
}: {
  items: HistoryItem[];
  onPick: (youtubeId: string) => void;
  onClear: () => void;
}) {
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <div style={{ fontWeight: 700 }}>History</div>
          <div className="small">Saved locally in your browser</div>
        </div>
        <button className="button" onClick={onClear} disabled={items.length === 0}>
          Clear
        </button>
      </div>

      {items.length === 0 ? (
        <div className="small" style={{ marginTop: 12 }}>
          No runs yet.
        </div>
      ) : (
        <div style={{ marginTop: 12, display: "grid", gap: 10 }}>
          {items.map((it) => (
            <button
              key={it.at}
              className="card"
              onClick={() => onPick(it.youtube_id)}
              style={{
                textAlign: "left",
                cursor: "pointer",
                background: "rgba(207, 207, 207, 0.83)",
              }}
            >
              <div className="row" style={{ justifyContent: "space-between" }}>
                <div style={{ fontWeight: 700 }}>{it.youtube_id}</div>
                <div style={{ fontWeight: 800 }}>
                  {(it.probability * 100).toFixed(1)}%
                </div>
              </div>
              <div className="small" style={{ marginTop: 6 }}>
                {it.confidence_label} • {fmtTime(it.at)}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
