import { Download } from "lucide-react";
import { exportPredictionReport, pngDataUrl, type PredictResult } from "../lib/api";
import { MetricCard } from "./MetricCard";
import { RouterExplanationPanel } from "./RouterExplanationPanel";

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

type Props = {
  result: PredictResult;
  benchmarks: any[];
  modelName: string;
};

function reliabilityClass(level: string, isUnknown: boolean) {
  if (isUnknown) return "border border-amber-400/30 bg-amber-400/10 text-amber-100";
  if (level === "High") return "border border-accent/30 bg-accent/10 text-accent-glow";
  if (level === "Medium") return "border border-yellow-400/20 bg-yellow-400/10 text-yellow-100";
  return "border border-red-400/20 bg-red-400/10 text-red-100";
}

export function AnalysisResults({ result, benchmarks, modelName }: Props) {
  const activeBenchmark = benchmarks.find((row) => row.model_key === (result.model_key ?? modelName));
  const assessment = result.assessment;
  const isCorrect =
    result.ground_truth_label &&
    (result.top_label === result.ground_truth_label ||
      (result.ground_truth_label === "dog_bark" && result.top_label === "dog") ||
      (result.ground_truth_label === "dog" && result.top_label === "dog_bark"));

  async function handleExport() {
    const blob = await exportPredictionReport(result);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "sound_analysis_report.zip";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <>
      {result.ground_truth_label ? (
        <div className={`glass-panel p-5 text-sm ${isCorrect ? "border border-accent/30" : "border border-amber-400/20"}`}>
          <div className="font-medium text-white">Dataset Ground Truth Comparison</div>
          <p className="mt-2 text-white/70">
            Known label: <strong>{formatLabel(result.ground_truth_label)}</strong> · Predicted:{" "}
            <strong>{formatLabel(result.top_label)}</strong> · Sample:{" "}
            <code className="text-accent-glow">{result.sample_id}</code>
          </p>
        </div>
      ) : null}

      {result.router ? <RouterExplanationPanel router={result.router} /> : null}

      <section className={`glass-panel p-5 ${reliabilityClass(assessment.reliability_level, assessment.is_unknown)}`}>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-lg font-semibold">{assessment.display_name}</div>
            <p className="mt-2 text-sm">
              Confidence: {(assessment.confidence * 100).toFixed(1)}% · Reliability: {assessment.reliability_level}
              {assessment.is_unknown ? " · Outside trained class list threshold" : ""}
            </p>
            <p className="mt-2 text-sm opacity-90">{assessment.reliability_message}</p>
            <p className="mt-2 text-xs opacity-70">{assessment.calibration_note}</p>
          </div>
          <button className="btn-secondary inline-flex items-center gap-2" onClick={handleExport}>
            <Download size={16} />
            Export Report
          </button>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Best Guess" value={formatLabel(result.top_label)} hint={`${(result.top_confidence * 100).toFixed(1)}% softmax`} accent />
        <MetricCard
          label="Live Latency"
          value={result.inference_ms ? `${result.inference_ms.toFixed(2)} ms` : "—"}
          hint={activeBenchmark ? `Benchmark ${activeBenchmark.inference_ms_mean} ms` : undefined}
        />
        <MetricCard label="Uncertainty" value={assessment.uncertainty_level} hint={`Entropy ${assessment.entropy_normalized.toFixed(3)}`} />
        <MetricCard label="Saved To Supabase" value={result.saved_prediction_id ? "Yes" : "No"} hint={result.input_source ?? "live input"} />
      </section>

      {benchmarks.length >= 3 ? (
        <section className="glass-panel p-5">
          <h3 className="mb-4 text-lg font-medium">Efficiency vs Accuracy Trade-off</h3>
          <div className="grid gap-4 lg:grid-cols-3">
            {benchmarks.slice(0, 3).map((row) => (
              <div key={row.model_key} className={`rounded-xl border p-4 ${row.is_deployed ? "border-accent/30 bg-accent/5" : "border-white/10 bg-ink-900/70"}`}>
                <div className="font-semibold">{row.display_name}</div>
                <div className="mt-3 space-y-1 text-sm text-white/70">
                  <div>Accuracy: {row.test_accuracy ? `${(row.test_accuracy * 100).toFixed(1)}%` : "—"}</div>
                  <div>Latency: {row.inference_ms_mean} ms</div>
                  <div>Size: {row.model_file_size_mb} MB</div>
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <section className="grid gap-4 xl:grid-cols-3">
        {[
          { title: "Raw Waveform", src: result.waveform_png },
          { title: "Mel-Spectrogram", src: result.mel_png },
          { title: "Grad-CAM Overlay", src: result.gradcam_png ?? result.rgb_png },
        ].map((panel) => (
          <div key={panel.title} className="glass-panel p-4">
            <h3 className="mb-3 text-sm font-medium text-white/80">{panel.title}</h3>
            <img src={pngDataUrl(panel.src)} alt={panel.title} className="w-full rounded-xl border border-white/10" />
          </div>
        ))}
      </section>

      <section className="glass-panel p-5">
        <h3 className="mb-4 text-lg font-medium">Top Predictions</h3>
        <div className="space-y-4">
          {result.predictions.map((prediction) => (
            <div key={prediction.label}>
              <div className="mb-1 flex justify-between text-sm">
                <span>{formatLabel(prediction.label)}</span>
                <span>{(prediction.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${prediction.confidence * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
