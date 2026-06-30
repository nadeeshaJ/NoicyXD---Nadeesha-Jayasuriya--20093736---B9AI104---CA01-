import { Download, AlertCircle, CheckCircle2, Zap, Volume2, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { API_BASE, exportPredictionReport, pngDataUrl, type PendingAudio, type PredictResult } from "../lib/api";
import { MetricCard } from "./MetricCard";
import { ConfidenceCalibrationPanel } from "./ConfidenceCalibrationPanel";
import { RouterExplanationPanel } from "./RouterExplanationPanel";

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

type Props = {
  result: PredictResult;
  benchmarks: any[];
  modelName: string;
  pendingAudio?: PendingAudio | null;
  datasetDomain?: "urban" | "animal" | null;
};

function reliabilityClass(level: string, isUnknown: boolean) {
  if (isUnknown) return "border-status-warning/30 bg-status-warning/[0.02] text-status-warning";
  if (level === "High") return "border-status-success/30 bg-status-success/[0.02] text-status-success";
  if (level === "Medium") return "border-status-warning/20 bg-status-warning/[0.02] text-status-warning";
  return "border-status-error/30 bg-status-error/[0.02] text-status-error";
}

export function AnalysisResults({ result, benchmarks, modelName, pendingAudio, datasetDomain }: Props) {
  const activeBenchmark = benchmarks.find((row) => row.model_key === (result.model_key ?? modelName));
  const assessment = result.assessment;
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackError, setPlaybackError] = useState<string | null>(null);

  const resolvedDatasetDomain =
    datasetDomain ??
    (result.input_source === "dataset" && result.effective_mode
      ? (result.effective_mode as "urban" | "animal")
      : null);
  const isDatasetSample = result.input_source === "dataset" && Boolean(result.sample_id);
  const canPlayDataset = isDatasetSample && Boolean(resolvedDatasetDomain);
  const canPlayBlob = !isDatasetSample && Boolean(pendingAudio?.blob);
  const canPlay = canPlayDataset || canPlayBlob;

  useEffect(() => {
    return () => {
      audioRef.current?.pause();
      audioRef.current = null;
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, []);

  function releaseObjectUrl() {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }

  function stopPlayback() {
    audioRef.current?.pause();
    audioRef.current = null;
    setIsPlaying(false);
    releaseObjectUrl();
  }

  async function togglePlay() {
    setPlaybackError(null);

    if (isPlaying) {
      stopPlayback();
      return;
    }

    let src: string;
    if (canPlayDataset && resolvedDatasetDomain && result.sample_id) {
      src = `${API_BASE}/api/datasets/${resolvedDatasetDomain}/samples/${encodeURIComponent(result.sample_id)}/audio`;
    } else if (canPlayBlob && pendingAudio?.blob) {
      releaseObjectUrl();
      src = URL.createObjectURL(pendingAudio.blob);
      objectUrlRef.current = src;
    } else {
      return;
    }

    const audio = new Audio(src);
    audioRef.current = audio;
    audio.onended = () => stopPlayback();
    audio.onerror = () => {
      setPlaybackError("Could not play this audio clip.");
      stopPlayback();
    };

    try {
      await audio.play();
      setIsPlaying(true);
    } catch {
      setPlaybackError("Playback blocked or unsupported format.");
      stopPlayback();
    }
  }

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
    <div className="space-y-6">
      {result.ground_truth_label ? (
        <div className={`glass-panel p-5 text-sm flex items-center justify-between gap-4 ${isCorrect ? "border-status-success/30 bg-status-success/[0.01]" : "border-status-warning/20 bg-status-warning/[0.01]"}`}>
          <div className="flex items-center gap-3">
            {isCorrect ? (
              <CheckCircle2 className="text-status-success shrink-0" size={20} />
            ) : (
              <AlertCircle className="text-status-warning shrink-0" size={20} />
            )}
            <div>
              <div className="font-bold text-white tracking-wide">Dataset Auditing Comparison</div>
              <p className="mt-0.5 text-xs text-white/50">
                Ground Truth: <strong className="text-white">{formatLabel(result.ground_truth_label)}</strong> · Predicted:{" "}
                <strong className={isCorrect ? "text-status-success" : "text-status-warning"}>{formatLabel(result.top_label)}</strong> · Sample:{" "}
                <code className="text-accent-glow font-mono bg-white/[0.03] px-1.5 py-0.5 rounded border border-white/[0.05]">{result.sample_id}</code>
              </p>
            </div>
          </div>
          <div className={`px-2.5 py-1 rounded text-[10px] font-bold uppercase tracking-wider ${
            isCorrect ? "bg-status-success/15 text-status-success border border-status-success/25" : "bg-status-warning/15 text-status-warning border border-status-warning/25"
          }`}>
            {isCorrect ? "Match" : "Mismatch"}
          </div>
        </div>
      ) : null}

      {result.router ? <RouterExplanationPanel router={result.router} /> : null}

      <section className={`glass-panel p-6 relative overflow-hidden border ${reliabilityClass(assessment.reliability_level, assessment.is_unknown)}`}>
        <div className="absolute top-0 right-0 w-48 h-48 bg-gradient-to-bl from-white/[0.01] to-transparent pointer-events-none" />
        <div className="flex flex-wrap items-center justify-between gap-6 relative z-10">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-3">
              <div className="text-xs uppercase font-bold tracking-widest opacity-60">Classification Assessment</div>
              {canPlay ? (
                <button
                  type="button"
                  className={`px-3 py-1 rounded-xl border text-[11px] font-bold transition flex items-center gap-1.5 ${
                    isPlaying
                      ? "bg-status-success/15 border-status-success/30 text-status-success shadow-glow animate-pulse"
                      : "bg-white/[0.03] border-white/[0.05] text-white/70 hover:bg-white/[0.08]"
                  }`}
                  onClick={() => void togglePlay()}
                >
                  {isPlaying ? <Square size={10} className="fill-current" /> : <Volume2 size={10} />}
                  {isPlaying ? "Stop Sound" : "Play Sound"}
                </button>
              ) : null}
            </div>
            {playbackError ? (
              <p className="text-[11px] text-status-warning font-medium">{playbackError}</p>
            ) : null}
            <div className="text-2xl font-extrabold text-white tracking-tight">{assessment.display_name}</div>
            <div className="text-xs opacity-80 font-medium">
              Confidence Score: {(assessment.confidence * 100).toFixed(1)}% · Reliability Rating: {assessment.reliability_level}
              {assessment.is_unknown ? " · Flagged as Unknown Source (Low Confidence)" : ""}
            </div>
            <p className="mt-2 text-sm text-white/70 leading-relaxed font-medium">{assessment.reliability_message}</p>
            <p className="text-xs text-white/40 italic">{assessment.calibration_note}</p>
          </div>
          <button className="btn-primary" onClick={handleExport}>
            <Download size={16} />
            Export Archive ZIP
          </button>
        </div>
      </section>

      <ConfidenceCalibrationPanel result={result} assessment={assessment} />

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Primary Guess" value={formatLabel(result.top_label)} hint={`${(result.top_confidence * 100).toFixed(1)}% confidence`} accent />
        <MetricCard
          label="Inference Latency"
          value={result.inference_ms ? `${result.inference_ms.toFixed(1)} ms` : "—"}
          hint={activeBenchmark ? `Benchmark: ${activeBenchmark.inference_ms_mean} ms` : undefined}
        />
        <MetricCard label="Entropy Rate" value={assessment.uncertainty_level} hint={`Entropy: ${assessment.entropy_normalized.toFixed(3)}`} />
        <MetricCard label="Supabase Sync" value={result.saved_prediction_id ? "Success" : "Offline"} hint={result.input_source ?? "live input"} />
      </section>

      {benchmarks.length >= 3 ? (
        <section className="glass-panel p-6">
          <div className="flex items-center gap-2 mb-4">
            <Zap size={16} className="text-accent-soft" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">Efficiency vs Accuracy Trade-off</h3>
          </div>
          <div className="grid gap-4 lg:grid-cols-3">
            {benchmarks.slice(0, 3).map((row) => (
              <div 
                key={row.model_key} 
                className={`rounded-2xl border p-4 transition duration-300 ${
                  row.is_deployed 
                    ? "border-accent/30 bg-accent/[0.02] hover:bg-accent/[0.04]" 
                    : "border-white/[0.05] bg-white/[0.01] hover:bg-white/[0.02]"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="font-bold text-white text-sm">{row.display_name}</div>
                  {row.is_deployed && (
                    <span className="rounded bg-accent/25 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-accent-glow">Active</span>
                  )}
                </div>
                <div className="space-y-1.5 text-xs text-white/55">
                  <div className="flex justify-between"><span>Accuracy:</span><span className="text-white font-semibold">{row.test_accuracy ? `${(row.test_accuracy * 100).toFixed(1)}%` : "—"}</span></div>
                  <div className="flex justify-between"><span>Latency:</span><span className="text-white font-mono">{row.inference_ms_mean} ms</span></div>
                  <div className="flex justify-between"><span>Size:</span><span className="text-white font-mono">{row.model_file_size_mb} MB</span></div>
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <section className="grid gap-6 md:grid-cols-3">
        {[
          { title: "Waveform Representation", src: result.waveform_png },
          { title: "Input Mel-Spectrogram", src: result.mel_png },
          { title: "Grad-CAM Explainability", src: result.gradcam_png ?? result.rgb_png },
        ].map((panel) => (
          <div key={panel.title} className="glass-panel p-4 hover:border-white/[0.08] transition">
            <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-white/60">{panel.title}</h3>
            <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-brand-dark/30">
              <img src={pngDataUrl(panel.src)} alt={panel.title} className="w-full hover:scale-105 transition duration-500" />
            </div>
          </div>
        ))}
      </section>

      <section className="glass-panel p-6">
        <h3 className="mb-5 text-sm font-bold uppercase tracking-wider text-white">Softmax Distribution (Top 3 Candidates)</h3>
        <div className="space-y-5">
          {result.predictions.map((prediction) => (
            <div key={prediction.label}>
              <div className="mb-1.5 flex justify-between text-xs font-medium">
                <span className="text-white/80">{formatLabel(prediction.label)}</span>
                <span className="text-accent-glow font-bold">{(prediction.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill animate-pulse-slow" style={{ width: `${prediction.confidence * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
