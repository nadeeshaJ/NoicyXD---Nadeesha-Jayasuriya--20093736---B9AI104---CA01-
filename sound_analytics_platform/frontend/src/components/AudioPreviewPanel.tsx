import { CheckCircle2, XCircle } from "lucide-react";
import { pngDataUrl, type AudioPreview } from "../lib/api";

type Props = {
  preview: AudioPreview;
  onRunAnalysis: () => void;
  onCompareModels: () => void;
  disabled?: boolean;
};

export function AudioPreviewPanel({ preview, onRunAnalysis, onCompareModels, disabled }: Props) {
  return (
    <section className="glass-panel p-5">
      <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">Pre-Inference Validation</h2>
          <p className="mt-1 text-sm text-white/60">
            Inspect the raw signal and automated preprocessing checks before running CNN inference.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button className="btn-primary" disabled={disabled || !preview.valid} onClick={onRunAnalysis}>
            Run Analysis
          </button>
          <button className="btn-secondary" disabled={disabled || !preview.valid} onClick={onCompareModels}>
            Compare All Models
          </button>
        </div>
      </div>

      <div className="mb-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {preview.validation_checks.map((check) => (
          <div
            key={check.name}
            className={`rounded-xl border p-3 ${check.passed ? "border-accent/30 bg-accent/5" : "border-amber-400/30 bg-amber-400/5"}`}
          >
            <div className="flex items-center gap-2 text-sm font-medium text-white">
              {check.passed ? <CheckCircle2 size={16} className="text-accent" /> : <XCircle size={16} className="text-amber-300" />}
              {check.name}
            </div>
            <div className="mt-2 text-xs text-white/55">Target: {check.target}</div>
            <div className="text-xs text-white/70">Actual: {check.actual}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-medium text-white/80">Raw Waveform Preview</h3>
          <img src={pngDataUrl(preview.waveform_png)} alt="Waveform preview" className="w-full rounded-xl border border-white/10" />
        </div>
        <div>
          <h3 className="mb-2 text-sm font-medium text-white/80">Mel-Spectrogram Preview</h3>
          <img src={pngDataUrl(preview.mel_png)} alt="Mel preview" className="w-full rounded-xl border border-white/10" />
        </div>
      </div>
    </section>
  );
}
