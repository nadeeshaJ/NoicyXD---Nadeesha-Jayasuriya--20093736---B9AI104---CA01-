import { MetricCard } from "./MetricCard";
import { type PredictResult } from "../lib/api";

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

type Props = {
  router: NonNullable<PredictResult["router"]>;
};

export function RouterExplanationPanel({ router }: Props) {
  return (
    <section className="glass-panel p-5">
      <h3 className="text-lg font-semibold text-accent-glow">Router Explanation</h3>
      <p className="mt-2 text-sm text-white/70">{router.primary_reason ?? router.reason}</p>
      {router.hint_note ? <p className="mt-1 text-xs text-white/45">{router.hint_note}</p> : null}

      <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Urban expert confidence"
          value={`${(router.urban_metrics.top_confidence * 100).toFixed(1)}%`}
          hint={formatLabel(router.urban_metrics.top_label)}
        />
        <MetricCard
          label="Animal expert confidence"
          value={`${(router.animal_metrics.top_confidence * 100).toFixed(1)}%`}
          hint={formatLabel(router.animal_metrics.top_label)}
        />
        <MetricCard label="Confidence gap" value={router.confidence_gap.toFixed(3)} hint="Calibrated strength difference" />
        <MetricCard label="Final route" value={router.domain.title()} hint={`Uncertainty: ${router.selected_uncertainty}`} accent />
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-white/10 bg-ink-900/70 p-4 text-sm text-white/75">
          <div className="font-medium text-white">Urban expert</div>
          <div className="mt-2 space-y-1">
            <div>Strength score: {router.urban_metrics.strength_score.toFixed(3)}</div>
            <div>Entropy: {router.urban_metrics.entropy_normalized.toFixed(3)}</div>
            <div>Uncertainty: {router.urban_metrics.uncertainty_level}</div>
          </div>
        </div>
        <div className="rounded-xl border border-white/10 bg-ink-900/70 p-4 text-sm text-white/75">
          <div className="font-medium text-white">Animal expert</div>
          <div className="mt-2 space-y-1">
            <div>Strength score: {router.animal_metrics.strength_score.toFixed(3)}</div>
            <div>Entropy: {router.animal_metrics.entropy_normalized.toFixed(3)}</div>
            <div>Uncertainty: {router.animal_metrics.uncertainty_level}</div>
          </div>
        </div>
      </div>
    </section>
  );
}
