import { MetricCard } from "./MetricCard";
import { type PredictResult } from "../lib/api";
import { Route } from "lucide-react";

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

type Props = {
  router: NonNullable<PredictResult["router"]>;
};

export function RouterExplanationPanel({ router }: Props) {
  const finalRoute = router.domain || "urban";
  return (
    <section className="glass-panel p-6 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-64 h-64 bg-cyanGradient pointer-events-none z-0" />
      
      <div className="relative z-10 flex items-center gap-3 mb-4">
        <div className="rounded-xl bg-cyan-500/10 p-2 text-cyan-glow border border-cyan-500/20 shadow-cyanGlow">
          <Route size={18} />
        </div>
        <div>
          <h3 className="text-lg font-bold text-white tracking-tight">Smart Domain Router Explanation</h3>
          <p className="text-xs text-white/50">Auto-routes audio streams to domain experts based on probability calibration.</p>
        </div>
      </div>
      
      <p className="relative z-10 text-sm text-white/70 leading-relaxed font-medium bg-white/[0.02] border border-white/[0.05] rounded-xl p-4">
        {router.primary_reason ?? router.reason}
        {router.hint_note ? <span className="block mt-1 text-xs text-white/40 font-normal">{router.hint_note}</span> : null}
      </p>

      <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4 relative z-10">
        <MetricCard
          label="Urban Probe Score"
          value={`${(router.urban_probe.top_confidence * 100).toFixed(1)}%`}
          hint={formatLabel(router.urban_probe.top_label)}
        />
        <MetricCard
          label="Animal Probe Score"
          value={`${(router.animal_probe.top_confidence * 100).toFixed(1)}%`}
          hint={formatLabel(router.animal_probe.top_label)}
        />
        <MetricCard label="Confidence Gap" value={router.confidence_gap.toFixed(3)} hint="Calibrated strength gap" />
        <MetricCard label="Final Assigned Route" value={finalRoute.toUpperCase()} hint={`Entropy: ${router.selected_uncertainty}`} accent />
      </div>

      <div className="mt-5 space-y-4 rounded-2xl border border-white/[0.04] bg-white/[0.01] p-5 relative z-10">
        <h4 className="text-xs font-bold uppercase tracking-wider text-white/60">Expert Probe Confidence Comparison</h4>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-xs mb-1.5 font-medium">
              <span className="text-white/70">Urban Expert Probe ({formatLabel(router.urban_probe.top_label)})</span>
              <span className="font-bold text-white">{(router.urban_probe.top_confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${router.urban_probe.top_confidence * 100}%` }}></div>
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1.5 font-medium">
              <span className="text-white/70">Animal Expert Probe ({formatLabel(router.animal_probe.top_label)})</span>
              <span className="font-bold text-white">{(router.animal_probe.top_confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${router.animal_probe.top_confidence * 100}%` }}></div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-4 md:grid-cols-2 relative z-10">
        <div className="rounded-2xl border border-white/[0.05] bg-white/[0.01] p-5 text-sm hover:border-white/[0.08] transition">
          <div className="font-bold text-white mb-3">Urban Domain Metrics</div>
          <div className="space-y-2 text-xs text-white/60">
            <div className="flex justify-between border-b border-white/[0.03] pb-1.5"><span>Calibrated Strength:</span><span className="font-semibold text-white">{router.urban_metrics.strength_score.toFixed(3)}</span></div>
            <div className="flex justify-between border-b border-white/[0.03] pb-1.5"><span>Entropy Rate:</span><span className="font-semibold text-white">{router.urban_metrics.entropy_normalized.toFixed(3)}</span></div>
            <div className="flex justify-between"><span>Uncertainty Assessment:</span><span className="font-semibold text-white">{router.urban_metrics.uncertainty_level}</span></div>
          </div>
        </div>
        <div className="rounded-2xl border border-white/[0.05] bg-white/[0.01] p-5 text-sm hover:border-white/[0.08] transition">
          <div className="font-bold text-white mb-3">Animal Domain Metrics</div>
          <div className="space-y-2 text-xs text-white/60">
            <div className="flex justify-between border-b border-white/[0.03] pb-1.5"><span>Calibrated Strength:</span><span className="font-semibold text-white">{router.animal_metrics.strength_score.toFixed(3)}</span></div>
            <div className="flex justify-between border-b border-white/[0.03] pb-1.5"><span>Entropy Rate:</span><span className="font-semibold text-white">{router.animal_metrics.entropy_normalized.toFixed(3)}</span></div>
            <div className="flex justify-between"><span>Uncertainty Assessment:</span><span className="font-semibold text-white">{router.animal_metrics.uncertainty_level}</span></div>
          </div>
        </div>
      </div>
    </section>
  );
}
