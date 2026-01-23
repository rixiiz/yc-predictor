"use client";

export default function ConfidenceBadge({ label }: { label: string }) {
  const tone =
    label === "Likely to get accepted"
      ? "rgba(34,197,94,0.18)"
      : label === "Almost there"
      ? "rgba(59,130,246,0.18)"
      : label === "Borderline"
      ? "rgba(234,179,8,0.18)"
      : "rgba(239,68,68,0.14)";

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "8px 10px",
        borderRadius: 999,
        border: "1px solid rgba(255,255,255,0.18)",
        background: tone,
        fontSize: 13,
        fontWeight: 700,
      }}
    >
      {label}
    </span>
  );
}
