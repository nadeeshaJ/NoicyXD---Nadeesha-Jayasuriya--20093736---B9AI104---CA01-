import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  FolderOpen,
  History,
  Radar,
  Sparkles,
  Waves,
} from "lucide-react";
import { useEffect, useState } from "react";
import { AnalysisResults } from "./components/AnalysisResults";
import { AnalyticsDashboardPanel } from "./components/AnalyticsDashboardPanel";
import { AudioInputPanel } from "./components/AudioInputPanel";
import { AudioPreviewPanel } from "./components/AudioPreviewPanel";
import { DatasetsPanel } from "./components/DatasetsPanel";
import { ModelComparisonPanel } from "./components/ModelComparisonPanel";
import { WaveLoader } from "./components/WaveLoader";
import {
  checkHealth,
  compareModels,
  fetchBenchmarksFromApi,
  fetchHistoryFromApi,
  predictAudio,
  previewAudio,
  type AudioPreview,
  type ModelCompareResult,
  type PendingAudio,
  type PredictResult,
  type ProcessingMode,
} from "./lib/api";
import { getSessionId } from "./lib/session";
import { fetchBenchmarksFromSupabase, fetchHistoryFromSupabase, supabaseConfigured } from "./lib/supabase";

type Tab = "analyze" | "datasets" | "analytics" | "history" | "models";

const modeOptions: Array<{ id: ProcessingMode; label: string }> = [
  { id: "urban", label: "Urban Sound" },
  { id: "animal", label: "Animal Vocalization" },
  { id: "auto", label: "Smart Auto-Router" },
];

const modelOptions = [
  { id: "mobilenetv2", label: "MobileNetV2 (Deployed)" },
  { id: "resnet50", label: "ResNet50" },
  { id: "custom_cnn", label: "Custom CNN" },
];

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

const getModeColorClass = (m: string) => {
  if (m === "urban") return "border-accent/30 text-accent-soft bg-accent/5";
  if (m === "animal") return "border-cyan-glow/30 text-cyan-glow bg-cyan-glow/5";
  return "border-status-warning/30 text-status-warning bg-status-warning/5";
};

const getModelColorClass = (m: string) => {
  if (m === "custom_cnn") return "border-status-success/30 text-status-success bg-status-success/5";
  if (m === "resnet50") return "border-indigo-400/30 text-indigo-400 bg-indigo-500/5";
  return "border-cyan-glow/30 text-cyan-glow bg-cyan-glow/5";
};

export default function App() {
  const [tab, setTab] = useState<Tab>("analyze");
  const [mode, setMode] = useState<ProcessingMode>("urban");
  const [modelName, setModelName] = useState("mobilenetv2");
  const [gradcam, setGradcam] = useState(true);
  const [loading, setLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingAudio, setPendingAudio] = useState<PendingAudio | null>(null);
  const [preview, setPreview] = useState<AudioPreview | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [comparison, setComparison] = useState<ModelCompareResult | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [benchmarks, setBenchmarks] = useState<any[]>([]);
  const [apiStatus, setApiStatus] = useState<string>("Checking API...");
  const [checkpointsReady, setCheckpointsReady] = useState(true);
  const [checkpointSummary, setCheckpointSummary] = useState("");

  useEffect(() => {
    checkHealth()
      .then((payload) => {
        setApiStatus(`Online · ${payload.device}`);
        setCheckpointsReady(Boolean(payload.checkpoints_ready));
        setCheckpointSummary(payload.checkpoint_summary || "");
      })
      .catch(() => setApiStatus("Offline"));
    loadBenchmarks();
  }, []);

  async function loadBenchmarks() {
    try {
      setBenchmarks(supabaseConfigured ? await fetchBenchmarksFromSupabase() : await fetchBenchmarksFromApi());
    } catch {
      try {
        setBenchmarks(await fetchBenchmarksFromApi());
      } catch {
        setBenchmarks([]);
      }
    }
  }

  async function loadHistory() {
    try {
      setHistory(supabaseConfigured ? await fetchHistoryFromSupabase(getSessionId()) : await fetchHistoryFromApi());
    } catch {
      try {
        setHistory(await fetchHistoryFromApi());
      } catch {
        setHistory([]);
      }
    }
  }

  useEffect(() => {
    if (tab === "history") loadHistory();
  }, [tab, result]);

  async function handleAudioSelected(blob: Blob, source: "upload" | "microphone", filename?: string) {
    setPreviewLoading(true);
    setError(null);
    setResult(null);
    setComparison(null);
    const pending = { blob, source, filename };
    setPendingAudio(pending);
    try {
      setPreview(
        await previewAudio({
          file: blob,
          filename,
          inputSource: source,
        }),
      );
      setTab("analyze");
    } catch (err) {
      setPreview(null);
      setError(err instanceof Error ? err.message : "Audio preview failed.");
    } finally {
      setPreviewLoading(false);
    }
  }

  async function handleRunAnalysis() {
    if (!pendingAudio) return;
    setLoading(true);
    setError(null);
    try {
      setResult(
        await predictAudio({
          file: pendingAudio.blob,
          filename: pendingAudio.filename,
          mode,
          modelName,
          inputSource: pendingAudio.source,
          gradcam,
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleCompareModels() {
    if (!pendingAudio) return;
    setLoading(true);
    setError(null);
    try {
      setComparison(
        await compareModels({
          file: pendingAudio.blob,
          filename: pendingAudio.filename,
          mode,
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Model comparison failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-brand-dark bg-[length:32px_32px] bg-grid relative overflow-hidden">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-7xl h-[400px] bg-glowGradient pointer-events-none z-0" />
      
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-8 lg:flex-row lg:px-8 relative z-10">
        <aside className="glass-panel w-full shrink-0 p-6 lg:w-80 flex flex-col justify-between">
          <div>
            <div className="mb-8 flex items-center gap-3">
              <div className="rounded-2xl bg-accent/10 p-3 text-accent-soft border border-accent/20 shadow-glow">
                <Waves size={24} className="animate-pulse-slow" />
              </div>
              <div>
                <div className="text-lg font-black tracking-wider text-white bg-gradient-to-r from-white via-white to-white/70 bg-clip-text">noicyXD</div>
                <div className="text-[10px] font-bold uppercase tracking-wider text-white/40">Audio Diagnostics</div>
              </div>
            </div>

            <nav className="space-y-1">
              {[
                { id: "analyze", label: "Analyze Live", icon: Radar },
                { id: "datasets", label: "Project Datasets", icon: FolderOpen },
                { id: "analytics", label: "Analytics Dashboard", icon: BarChart3 },
                { id: "history", label: "Prediction History", icon: History },
                { id: "models", label: "CNN Models", icon: BrainCircuit },
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  className={tab === id ? "btn-tab-active" : "btn-tab-inactive"}
                  onClick={() => setTab(id as Tab)}
                >
                  <Icon size={18} />
                  {label}
                </button>
              ))}
            </nav>

            {/* Removed sidebar controls to place them globally on the horizontal bar */}
          </div>

          <div className="mt-8 rounded-2xl border border-white/[0.05] bg-white/[0.01] p-5 text-xs text-white/45 space-y-3">
            <div className="flex items-center gap-2 text-white/80 font-bold">
              <Database size={15} className="text-accent-soft" />
              System Status
            </div>
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${apiStatus.startsWith("Online") ? "bg-status-success shadow-[0_0_10px_#10b981]" : "bg-status-error shadow-[0_0_10px_#f43f5e] animate-pulse"}`}></span>
              <span>API: {apiStatus}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${supabaseConfigured ? "bg-status-success shadow-[0_0_10px_#10b981]" : "bg-status-warning shadow-[0_0_10px_#f59e0b]"}`}></span>
              <span>DB: {supabaseConfigured ? "Supabase Connected" : "Local Database Mode"}</span>
            </div>
          </div>
        </aside>

        <main className="flex-1 space-y-6 min-w-0">
          <header className="glass-panel p-6 relative overflow-hidden max-w-5xl">
            <div className="absolute top-0 right-0 w-64 h-full bg-cyanGradient pointer-events-none z-0" />
            <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
              <div className="flex-1">
                <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl leading-none">
                  Intelligent Sound Classification & Neural Auditing
                </h1>
                <p className="mt-2 max-w-xl text-xs leading-relaxed text-white/50">
                  Analyze raw acoustic streams, compare deep learning models in real-time, and audit decision boundaries with explainable AI heatmaps and expert routing telemetry.
                </p>
              </div>

              {/* Dynamic Control Card inside Header */}
              <div className="flex flex-col sm:flex-row lg:flex-col gap-3 bg-brand-dark/50 border border-white/[0.06] rounded-2xl p-3 shrink-0 lg:w-64 relative z-10 w-full lg:w-auto">
                {/* Processing Mode */}
                <div className="flex-1">
                  <label className="mb-1.5 block text-[10px] font-bold uppercase tracking-wider text-white/40">Processing Mode</label>
                  <select 
                    className={`w-full border rounded-xl px-3 py-1.5 text-xs font-semibold focus:outline-none focus:ring-1 focus:ring-accent/20 transition cursor-pointer ${getModeColorClass(mode)}`}
                    value={mode} 
                    onChange={(e) => setMode(e.target.value as ProcessingMode)}
                  >
                    {modeOptions.map((option) => (
                      <option key={option.id} value={option.id} className="bg-brand-dark text-white">
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Backend Model */}
                <div className="flex-1">
                  <label className="mb-1.5 block text-[10px] font-bold uppercase tracking-wider text-white/40">Backend Model</label>
                  <select 
                    className={`w-full border rounded-xl px-3 py-1.5 text-xs font-semibold focus:outline-none focus:ring-1 focus:ring-accent/20 transition cursor-pointer ${getModelColorClass(modelName)}`}
                    value={modelName} 
                    onChange={(e) => setModelName(e.target.value)}
                  >
                    {modelOptions.map((option) => (
                      <option key={option.id} value={option.id} className="bg-brand-dark text-white">
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Grad-CAM explainability */}
                <div className="flex items-center sm:items-end lg:items-center pt-2 sm:pt-0 lg:pt-2">
                  <label className={`w-full border rounded-xl px-3 py-2 text-xs font-semibold transition cursor-pointer select-none flex items-center justify-between gap-3 ${
                    gradcam 
                      ? "border-status-success/30 text-status-success bg-status-success/5 shadow-glow" 
                      : "border-white/10 text-white/40 bg-white/[0.01]"
                  }`}>
                    <span className="uppercase tracking-wider text-[10px]">Grad-CAM</span>
                    <input 
                      type="checkbox" 
                      checked={gradcam} 
                      onChange={(e) => setGradcam(e.target.checked)} 
                      className="rounded border-white/10 bg-brand-dark text-accent focus:ring-accent/20 focus:ring-offset-brand-dark"
                    />
                  </label>
                </div>
              </div>
            </div>
          </header>

          {loading || previewLoading ? (
            <WaveLoader 
              message={previewLoading ? "Running validation preview..." : "Processing audio signals..."}
              submessage="Applying Short-Time Fourier Transform & CNN Inference"
            />
          ) : null}
          
          {error ? (
            <div className="glass-panel border-status-error/30 bg-status-error/5 p-5 text-status-error text-sm font-medium flex items-center gap-3">
              <span className="h-2 w-2 rounded-full bg-status-error shadow-[0_0_10px_#f43f5e] shrink-0"></span>
              {error}
            </div>
          ) : null}

          {!checkpointsReady && apiStatus.startsWith("Online") ? (
            <div className="glass-panel border-status-warning/30 bg-status-warning/5 p-5 text-status-warning text-sm">
              <div className="font-semibold">Trained model weights missing or invalid</div>
              <p className="mt-2 opacity-90">
                {checkpointSummary || "Predictions will be wrong until real best_model.pt files are installed."}
              </p>
              <p className="mt-2 text-xs opacity-80">
                From repo root:{" "}
                <code className="rounded bg-black/30 px-1 py-0.5">
                  python scripts/setup_checkpoints.py --source PATH_TO_experiments
                </code>
              </p>
            </div>
          ) : null}

          {tab === "analyze" ? (
            <>
              <AudioInputPanel onAudioSelected={handleAudioSelected} disabled={loading || previewLoading} />
              {preview ? (
                <AudioPreviewPanel
                  preview={preview}
                  pendingAudio={pendingAudio}
                  onRunAnalysis={handleRunAnalysis}
                  onCompareModels={handleCompareModels}
                  disabled={loading}
                />
              ) : (
                <div className="glass-panel p-12 text-center text-white/40 text-sm flex flex-col items-center justify-center gap-3">
                  <Waves size={36} className="text-white/20" />
                  <span>Upload a WAV file or record audio using the mic to see validation checks and waveform previews.</span>
                </div>
              )}
            </>
          ) : null}

          {tab === "datasets" ? (
            <DatasetsPanel
              mode={mode}
              modelName={modelName}
              gradcam={gradcam}
              onResult={(payload) => {
                setResult(payload);
                setComparison(null);
                setError(null);
              }}
              onComparison={setComparison}
              onLoading={setLoading}
              onError={setError}
            />
          ) : null}

          {tab === "analytics" ? <AnalyticsDashboardPanel /> : null}

          {tab === "history" ? (
            <section className="glass-panel p-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-bold text-white">Prediction History Logs</h2>
                  <p className="text-xs text-white/50">Stored predictions in Supabase linked to your session.</p>
                </div>
                <button className="btn-secondary" onClick={loadHistory}>
                  Refresh Logs
                </button>
              </div>
              {history.length === 0 ? (
                <p className="text-white/40 text-sm p-6 text-center border border-dashed border-white/5 rounded-2xl">No predictions stored yet in this session.</p>
              ) : (
                <div className="overflow-x-auto rounded-xl border border-white/[0.05]">
                  <table className="min-w-full text-left text-sm">
                    <thead className="bg-white/[0.02] text-white/40 font-semibold">
                      <tr>
                        <th className="px-4 py-3">Timestamp</th>
                        <th className="px-4 py-3">Source</th>
                        <th className="px-4 py-3">Mode</th>
                        <th className="px-4 py-3">Top Guess</th>
                        <th className="px-4 py-3">Confidence</th>
                        <th className="px-4 py-3">Reliability</th>
                        <th className="px-4 py-3 text-right">Inference</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.05]">
                      {history.map((row) => (
                        <tr key={row.id} className="hover:bg-white/[0.02] text-white/70 transition">
                          <td className="px-4 py-4 whitespace-nowrap text-xs text-white/50">{new Date(row.created_at).toLocaleString()}</td>
                          <td className="px-4 py-4 whitespace-nowrap capitalize text-xs">
                            <span className="px-2 py-1 rounded bg-white/[0.04] border border-white/[0.05]">
                              {row.input_source ?? "upload"}
                            </span>
                          </td>
                          <td className="px-4 py-4 whitespace-nowrap capitalize text-xs">{row.processing_mode}</td>
                          <td className="px-4 py-4 whitespace-nowrap font-medium text-white">{formatLabel(row.display_label || row.top_label)}</td>
                          <td className="px-4 py-4 whitespace-nowrap text-xs">{(Number(row.top_confidence) * 100).toFixed(1)}%</td>
                          <td className="px-4 py-4 whitespace-nowrap text-xs">
                            <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase ${
                              row.reliability_level === 'High' ? 'bg-status-success/10 text-status-success border border-status-success/20' :
                              row.reliability_level === 'Medium' ? 'bg-status-warning/10 text-status-warning border border-status-warning/20' :
                              'bg-status-error/10 text-status-error border border-status-error/20'
                            }`}>
                              {row.reliability_level ?? "Low"}
                            </span>
                          </td>
                          <td className="px-4 py-4 whitespace-nowrap text-right text-xs font-mono text-white/50">
                            {row.inference_ms ? `${Number(row.inference_ms).toFixed(1)} ms` : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          ) : null}

          {tab === "models" ? (
            <section className="grid gap-6 lg:grid-cols-3">
              {benchmarks.map((row) => (
                <div 
                  key={row.model_key} 
                  className={`glass-panel p-6 relative overflow-hidden border ${
                    row.is_deployed ? "border-accent/40 bg-accent/[0.03] shadow-glow" : "border-white/[0.05]"
                  }`}
                >
                  <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-accent/5 to-transparent pointer-events-none" />
                  
                  <div className="flex items-start justify-between mb-4">
                    <div className="text-lg font-bold text-white tracking-tight">{row.display_name}</div>
                    {row.is_deployed ? (
                      <span className="rounded-full bg-accent/25 border border-accent/30 px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-accent-glow shadow-glow animate-pulse-slow">
                        Active Expert
                      </span>
                    ) : null}
                  </div>
                  
                  <div className="space-y-3.5 text-sm">
                    <div className="flex justify-between border-b border-white/[0.05] pb-2">
                      <span className="text-white/40">Test Accuracy</span>
                      <span className="font-semibold text-white">{row.test_accuracy ? `${(row.test_accuracy * 100).toFixed(1)}%` : "—"}</span>
                    </div>
                    <div className="flex justify-between border-b border-white/[0.05] pb-2">
                      <span className="text-white/40">Macro F1 Score</span>
                      <span className="font-semibold text-white">{row.test_macro_f1 ? row.test_macro_f1.toFixed(3) : "—"}</span>
                    </div>
                    <div className="flex justify-between border-b border-white/[0.05] pb-2">
                      <span className="text-white/40">Mean Latency</span>
                      <span className="font-semibold text-white font-mono">{row.inference_ms_mean} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/40">Checkpoint Size</span>
                      <span className="font-semibold text-white font-mono">{row.model_file_size_mb} MB</span>
                    </div>
                  </div>
                </div>
              ))}
            </section>
          ) : null}
        </main>
      </div>

      {/* Result/Comparison Modal Overlay */}
      {(result || comparison) && (
        <div className="fixed inset-0 z-50 overflow-y-auto bg-brand-dark/80 backdrop-blur-md p-4 md:p-10 flex justify-center items-start">
          <div className="fixed inset-0" onClick={() => { setResult(null); setComparison(null); }} />
          
          <div className="relative w-full max-w-6xl rounded-3xl border border-white/[0.08] bg-brand-dark/95 shadow-2xl p-6 md:p-8 z-10 my-auto">
            {/* Close button */}
            <button
              onClick={() => { setResult(null); setComparison(null); }}
              className="absolute top-5 right-5 rounded-full p-2.5 text-white/45 hover:text-white hover:bg-white/[0.05] transition z-20 border border-white/[0.06] bg-brand-dark"
              aria-label="Close modal"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            <div className="mt-2">
              {result && (
                <div>
                  <div className="mb-6 relative">
                    <h2 className="text-2xl font-black text-white tracking-tight">Classification Analysis Report</h2>
                    <p className="text-xs text-white/50 mt-1">Deep Learning Inference outputs & heatmaps for sample {result.sample_id || "live input"}</p>
                  </div>
                  <AnalysisResults result={result} benchmarks={benchmarks} modelName={modelName} />
                </div>
              )}

              {comparison && (
                <div>
                  <div className="mb-6 relative">
                    <h2 className="text-2xl font-black text-white tracking-tight">Multi-Model Cross Auditing Report</h2>
                    <p className="text-xs text-white/50 mt-1">Comparative inference metrics across MobileNetV2, ResNet50, and Custom CNN architectures</p>
                  </div>
                  <ModelComparisonPanel comparison={comparison} />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
