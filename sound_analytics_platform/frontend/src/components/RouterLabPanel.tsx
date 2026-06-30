import { useState } from "react";
import { FlaskConical, Route } from "lucide-react";
import {
  predictAudio,
  predictFromSample,
  type PendingAudio,
  type PredictResult,
  type ProcessingMode,
} from "../lib/api";
import { RouterExplanationPanel } from "./RouterExplanationPanel";
import { WaveLoader } from "./WaveLoader";

export type RouterLabContext = {
  result: PredictResult;
  pendingAudio?: PendingAudio | null;
  datasetDomain?: "urban" | "animal" | null;
  sampleId?: string | null;
};

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

type WhatIfKey = "urban" | "animal";

type Props = {
  context: RouterLabContext | null;
  modelName: string;
  gradcam: boolean;
  onOpenResult?: (result: PredictResult) => void;
};

export function RouterLabPanel({ context, modelName, gradcam, onOpenResult }: Props) {
  const [whatIf, setWhatIf] = useState<Partial<Record<WhatIfKey, PredictResult>>>({});
  const [loadingMode, setLoadingMode] = useState<WhatIfKey | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!context?.result.router) {
    return (
      <div className="glass-panel p-12 text-center text-white/40 text-sm flex flex-col items-center gap-3">
        <Route size={36} className="text-white/20" />
        <p>
          Run a prediction with <span className="text-white/70 font-medium">Smart Auto-Router</span> on Analyze Live,
          Datasets, or Showcase. The last auto-routed clip appears here for transparency and what-if comparisons.
        </p>
      </div>
    );
  }

  const labContext = context;
  const autoResult = labContext.result;
  const router = autoResult.router!;

  async function runWhatIf(forcedMode: WhatIfKey) {
    setLoadingMode(forcedMode);
    setError(null);
    try {
      let payload: PredictResult;
      if (labContext.pendingAudio) {
        payload = await predictAudio({
          file: labContext.pendingAudio.blob,
          filename: labContext.pendingAudio.filename,
          mode: forcedMode,
          modelName: forcedMode === "animal" ? "mobilenetv2" : modelName,
          inputSource: labContext.pendingAudio.source,
          gradcam,
        });
      } else if (labContext.datasetDomain && labContext.sampleId) {
        payload = await predictFromSample({
          domain: labContext.datasetDomain,
          sampleId: labContext.sampleId,
          mode: forcedMode as ProcessingMode,
          modelName: forcedMode === "animal" ? "mobilenetv2" : modelName,
          gradcam,
        });
      } else {
        throw new Error("No replayable audio context for this auto-routed prediction.");
      }
      setWhatIf((prev) => ({ ...prev, [forcedMode]: payload }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "What-if rerun failed.");
    } finally {
      setLoadingMode(null);
    }
  }

  function renderComparisonRow(mode: WhatIfKey, label: string) {
    const forced = whatIf[mode];
    const autoLabel = formatLabel(autoResult.top_label);
    const autoConf = (autoResult.top_confidence * 100).toFixed(1);
    const forcedLabel = forced ? formatLabel(forced.top_label) : "—";
    const forcedConf = forced ? (forced.top_confidence * 100).toFixed(1) : "—";
    const agrees = forced ? forced.top_label === autoResult.top_label : null;

    return (
      <div className="rounded-xl border border-white/[0.05] bg-white/[0.02] p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <span className="text-xs font-bold uppercase tracking-wider text-white/60">Force {label}</span>
          <button
            type="button"
            className="btn-secondary py-1.5 px-3 text-[11px]"
            onClick={() => runWhatIf(mode)}
            disabled={loadingMode !== null}
          >
            {loadingMode === mode ? "Running…" : `Rerun as ${label}`}
          </button>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 text-xs">
          <div>
            <div className="text-white/40 mb-1">Auto route ({router.domain})</div>
            <div className="font-semibold text-white">
              {autoLabel} <span className="text-accent-soft">{autoConf}%</span>
            </div>
          </div>
          <div>
            <div className="text-white/40 mb-1">Forced {label.toLowerCase()}</div>
            <div className="font-semibold text-white">
              {forcedLabel}{" "}
              {forced ? <span className="text-cyan-glow">{forcedConf}%</span> : <span className="text-white/30">not run</span>}
            </div>
            {agrees !== null ? (
              <div className={`mt-1 text-[10px] font-semibold uppercase ${agrees ? "text-status-success" : "text-status-warning"}`}>
                {agrees ? "Same top label as auto" : "Different from auto route"}
              </div>
            ) : null}
          </div>
        </div>
        {forced && onOpenResult ? (
          <button
            type="button"
            className="mt-3 text-[11px] text-accent-soft hover:text-white transition"
            onClick={() => onOpenResult(forced)}
          >
            Open full forced report →
          </button>
        ) : null}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {loadingMode ? (
        <WaveLoader
          message={`Running forced ${loadingMode} expert...`}
          submessage="Replaying the same clip without auto-routing"
        />
      ) : null}

      {error ? (
        <div className="glass-panel border-status-error/30 bg-status-error/5 p-5 text-status-error text-sm font-medium flex items-center gap-3">
          <span className="h-2 w-2 rounded-full bg-status-error shadow-[0_0_10px_#f43f5e] shrink-0"></span>
          {error}
        </div>
      ) : null}

      <RouterExplanationPanel router={router} />

      <section className="glass-panel p-6 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-48 h-48 bg-cyanGradient pointer-events-none z-0" />
        <div className="relative z-10 flex items-center gap-3 mb-4">
          <div className="rounded-xl bg-status-warning/10 p-2 text-status-warning border border-status-warning/20">
            <FlaskConical size={18} />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white tracking-tight">Router What-If Lab</h3>
            <p className="text-xs text-white/50">
              Replay the same audio through urban-only and animal-only experts to compare against the auto route.
            </p>
          </div>
        </div>

        <div className="relative z-10 space-y-4">
          {renderComparisonRow("urban", "Urban")}
          {renderComparisonRow("animal", "Animal")}
        </div>

        <p className="relative z-10 mt-4 text-[11px] text-white/35">
          Context:{" "}
          {labContext.pendingAudio
            ? `${labContext.pendingAudio.source} clip`
            : labContext.sampleId
              ? `dataset sample ${labContext.sampleId}`
              : "unknown source"}
        </p>
      </section>
    </div>
  );
}
