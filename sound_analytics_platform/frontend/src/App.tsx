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

  useEffect(() => {
    checkHealth()
      .then((payload) => setApiStatus(`Online · ${payload.device}`))
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

            <div className="mt-8 space-y-5 border-t border-white/[0.05] pt-6">
              <div>
                <label className="mb-2 block text-[10px] font-bold uppercase tracking-[0.2em] text-white/40">Processing Mode</label>
                <select className="input-shell" value={mode} onChange={(e) => setMode(e.target.value as ProcessingMode)}>
                  {modeOptions.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-2 block text-[10px] font-bold uppercase tracking-[0.2em] text-white/40">Backend Model</label>
                <select className="input-shell" value={modelName} onChange={(e) => setModelName(e.target.value)}>
                  {modelOptions.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              <label className="flex items-center gap-3 text-sm text-white/60 hover:text-white transition cursor-pointer select-none">
                <input 
                  type="checkbox" 
                  checked={gradcam} 
                  onChange={(e) => setGradcam(e.target.checked)} 
                  className="rounded border-white/10 bg-brand-dark text-accent focus:ring-accent/20 focus:ring-offset-brand-dark"
                />
                <span>Grad-CAM explainability</span>
              </label>
            </div>
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
          <header className="glass-panel p-8 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-80 h-full bg-cyanGradient pointer-events-none z-0" />
            <div className="relative z-10">
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/10 px-3 py-1.5 text-xs font-semibold text-accent-glow shadow-glow">
                <Sparkles size={14} className="animate-spin-slow" />
                noicyXD · Audio Diagnostics Platform
              </div>
              <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">Validate, classify, and audit sound event streams</h1>
              <p className="mt-3 max-w-3xl text-sm leading-relaxed text-white/60">
                Pre-inference waveform validation, same-clip multi-model predictions auditing, dual-expert routing telemetry, and live database predictions monitoring.
              </p>
            </div>
          </header>

          {loading || previewLoading ? (
            <div className="glass-panel flex flex-col items-center justify-center p-12 text-center">
              <div className="flex items-end justify-center gap-1.5 h-12 mb-5">
                <div className="w-1.5 bg-accent rounded-full visualizer-bar" style={{ animationDelay: '0.1s', height: '12px' }}></div>
                <div className="w-1.5 bg-accent-soft rounded-full visualizer-bar" style={{ animationDelay: '0.3s', height: '12px' }}></div>
                <div className="w-1.5 bg-cyan-glow rounded-full visualizer-bar" style={{ animationDelay: '0.5s', height: '12px' }}></div>
                <div className="w-1.5 bg-accent-soft rounded-full visualizer-bar" style={{ animationDelay: '0.2s', height: '12px' }}></div>
                <div className="w-1.5 bg-accent rounded-full visualizer-bar" style={{ animationDelay: '0.4s', height: '12px' }}></div>
              </div>
              <div className="text-base font-bold tracking-wide text-white/95">
                {previewLoading ? "Running validation preview..." : "Processing audio signals..."}
              </div>
              <div className="text-xs text-white/40 mt-1">Applying Short-Time Fourier Transform & CNN Inference</div>
            </div>
          ) : null}
          
          {error ? (
            <div className="glass-panel border-status-error/30 bg-status-error/5 p-5 text-status-error text-sm font-medium flex items-center gap-3">
              <span className="h-2 w-2 rounded-full bg-status-error shadow-[0_0_10px_#f43f5e] shrink-0"></span>
              {error}
            </div>
          ) : null}

          {tab === "analyze" ? (
            <>
              <AudioInputPanel onAudioSelected={handleAudioSelected} disabled={loading || previewLoading} />
              {preview ? (
                <AudioPreviewPanel
                  preview={preview}
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
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-brand-dark/85 backdrop-blur-md overflow-y-auto">
          <div className="absolute inset-0" onClick={() => { setResult(null); setComparison(null); }} />
          
          <div className="relative w-full max-w-6xl rounded-3xl border border-white/[0.08] bg-brand-dark/95 shadow-2xl overflow-y-auto max-h-[90vh] z-10 p-6 md:p-8 custom-scrollbar">
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
