"use client";

type Stage = {
  key: string;
  label: string;
};

const STAGES: Stage[] = [
  { key: "idle", label: "Waiting" },
  { key: "download", label: "Downloading video" },
  { key: "frames", label: "Extracting frames" },
  { key: "audio", label: "Extracting audio" },
  { key: "asr", label: "Transcribing speech" },
  { key: "embed", label: "Embedding + scoring" },
  { key: "done", label: "Done" },
];

export type StageKey = (typeof STAGES)[number]["key"];

export default function ProgressTimeline({
  stage,
  error,
}: {
  stage: StageKey;
  error?: string | null;
}) {
  const idx = STAGES.findIndex((s) => s.key === stage);

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="small" style={{ marginBottom: 10 }}>
        Progress
      </div>

      <div style={{ display: "grid", gap: 10 }}>
        {STAGES.map((s, i) => {
          const done = i < idx;
          const active = i === idx;
          const dim = i > idx;

          return (
            <div
              key={s.key}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                opacity: dim ? 0.55 : 1,
              }}
            >
              <div
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: 999,
                  border: "1px solid rgba(255,255,255,0.25)",
                  background: done
                    ? "rgba(34,197,94,0.75)"
                    : active
                    ? "rgba(59,130,246,0.85)"
                    : "rgba(255,255,255,0.08)",
                }}
              />
              <div style={{ fontWeight: active ? 700 : 500 }}>{s.label}</div>
            </div>
          );
        })}
      </div>

      {error ? (
        <div className="error" style={{ marginTop: 14 }}>
          {error}
        </div>
      ) : null}
    </div>
  );
}
