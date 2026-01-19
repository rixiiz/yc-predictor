"use client";

export type Contrib = {
  intercept: number;
  text_logit: number;
  frames_logit: number;
  total_logit: number;
};

function fmt(n: number) {
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(3)}`;
}

export default function ContributionCard({ contrib }: { contrib: Contrib }) {
  const { intercept, text_logit, frames_logit, total_logit } = contrib;

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div style={{ fontWeight: 700 }}>Feature contribution summary</div>
      <div className="small" style={{ marginTop: 6 }}>
        These are contributions in <b>logit</b> space (linear score before the sigmoid).
        Positive pushes toward “accepted”, negative pushes away.
      </div>

      <div style={{ marginTop: 14, display: "grid", gap: 10 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div>Intercept</div>
          <div style={{ fontWeight: 800 }}>{fmt(intercept)}</div>
        </div>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div>Text embedding</div>
          <div style={{ fontWeight: 800 }}>{fmt(text_logit)}</div>
        </div>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div>Frame features</div>
          <div style={{ fontWeight: 800 }}>{fmt(frames_logit)}</div>
        </div>
        <div
          className="row"
          style={{
            justifyContent: "space-between",
            paddingTop: 10,
            borderTop: "1px solid rgba(255,255,255,0.12)",
          }}
        >
          <div style={{ fontWeight: 800 }}>Total logit</div>
          <div style={{ fontWeight: 900 }}>{fmt(total_logit)}</div>
        </div>
      </div>
    </div>
  );
}
