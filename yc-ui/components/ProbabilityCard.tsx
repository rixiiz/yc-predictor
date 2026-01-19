export default function ProbabilityCard({ p }: { p: number }) {
  const pct = Math.round(p * 1000) / 10; // 1 decimal %
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="small">YC-like probability</div>
      <div className="metric">{pct}%</div>
      <div className="small" style={{ marginTop: 6, opacity: 0.8 }}>
        (Raw: {p.toFixed(4)})
      </div>
    </div>
  );
}
