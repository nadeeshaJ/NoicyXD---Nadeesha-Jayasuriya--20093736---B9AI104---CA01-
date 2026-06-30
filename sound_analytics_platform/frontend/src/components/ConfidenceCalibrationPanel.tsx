import type { AssessmentInfo, PredictResult } from "../lib/api";

const UNKNOWN_THRESH = 0.4;
const HIGH_THRESH = 0.7;

type Props = {
  result: PredictResult;
  assessment: AssessmentInfo;
};

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export function ConfidenceCalibrationPanel({ result, assessment }: Props) {
  const confidence = assessment.confidence;
  const pct = confidence * 100;
  const top2Gap =
    result.predictions.length >= 2
      ? (result.predictions[0].confidence - result.predictions[1].confidence) * 100
      : null;

  return (
    <section className="glass-panel p-5">
      <h3 className="mb-4 text-xs font-bold uppercase tracking-wider text-white/60">Confidence calibration</h3>

      <div className="relative mb-6 h-3 rounded-full bg-white/[0.06] overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${
            assessment.is_unknown
              ? "bg-status-warning"
              : assessment.reliability_level === "High"
                ? "bg-status-success"
                : assessment.reliability_level === "Medium"
                  ? "bg-status-warning"
                  : "bg-status-error"
          }`}
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
        <div className="absolute inset-y-0 w-px bg-white/25" style={{ left: `${UNKNOWN_THRESH * 100}%` }} title="Unknown threshold" />
        <div className="absolute inset-y-0 w-px bg-white/25" style={{ left: `${HIGH_THRESH * 100}%` }} title="High reliability threshold" />
      </div>

      <div className="flex justify-between text-[10px] text-white/35 mb-4 -mt-1">
        <span>0%</span>
        <span>Unknown &lt; {UNKNOWN_THRESH * 100}%</span>
        <span>High ≥ {HIGH_THRESH * 100}%</span>
        <span>100%</span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 text-xs">
        <div className="rounded-lg border border-white/[0.05] bg-white/[0.02] p-3">
          <div className="text-white/45 mb-1">Top-1 confidence</div>
          <div className="text-white font-semibold">{pct.toFixed(1)}%</div>
        </div>
        <div className="rounded-lg border border-white/[0.05] bg-white/[0.02] p-3">
          <div className="text-white/45 mb-1">Top-1 vs top-2 gap</div>
          <div className="text-white font-semibold">{top2Gap != null ? `${top2Gap.toFixed(1)}%` : "—"}</div>
        </div>
        <div className="rounded-lg border border-white/[0.05] bg-white/[0.02] p-3">
          <div className="text-white/45 mb-1">Normalized entropy</div>
          <div className="text-white font-semibold">{assessment.entropy_normalized.toFixed(3)}</div>
          <div className="text-white/35 mt-0.5">Uncertainty: {assessment.uncertainty_level}</div>
        </div>
        <div className="rounded-lg border border-white/[0.05] bg-white/[0.02] p-3">
          <div className="text-white/45 mb-1">Reliability</div>
          <div className="text-white font-semibold">{assessment.reliability_level}</div>
          <div className="text-white/35 mt-0.5">{assessment.is_unknown ? "Below unknown threshold" : formatLabel(assessment.display_label)}</div>
        </div>
      </div>
    </section>
  );
}
