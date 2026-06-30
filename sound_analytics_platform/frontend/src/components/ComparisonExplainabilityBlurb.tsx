import { Eye } from "lucide-react";
import type { ModelCompareResult } from "../lib/api";
import { buildComparisonNarrative } from "../lib/comparisonSummary";
import { PlaySoundButton, type ReportAudioSource } from "./PlaySoundButton";

type Props = {
  comparison: ModelCompareResult;
  audioSource: ReportAudioSource;
};

export function ComparisonExplainabilityBlurb({ comparison, audioSource }: Props) {
  return (
    <div className="mb-5 rounded-2xl border border-accent/15 bg-accent/[0.03] p-5 relative overflow-hidden">
      <div className="absolute top-0 right-0 w-48 h-48 bg-cyanGradient pointer-events-none z-0" />
      <div className="relative z-10 flex flex-wrap items-start justify-between gap-4 mb-3">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-accent/10 p-2 text-accent-soft border border-accent/20">
            <Eye size={16} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-tight">Explainable AI · Model comparison</h3>
            <p className="text-[11px] text-white/45 mt-0.5">
              Plain-language readout of agreement, confidence, and deployment trade-offs
            </p>
          </div>
        </div>
        <PlaySoundButton source={audioSource} size="md" />
      </div>
      <p className="relative z-10 text-sm text-white/75 leading-relaxed font-medium">
        {buildComparisonNarrative(comparison)}
      </p>
    </div>
  );
}
