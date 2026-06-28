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
    <div className="min-h-screen bg-[length:24px_24px] bg-grid">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-6 lg:flex-row lg:px-8">
        <aside className="glass-panel w-full shrink-0 p-5 lg:w-72">
          <div className="mb-8 flex items-center gap-3">
            <div className="rounded-2xl bg-accent/15 p-3 text-accent">
              <Waves size={22} />
            </div>
            <div>
              <div className="text-lg font-semibold">Sound Analytics</div>
              <div className="text-xs text-white/50">Supabase + CNN Platform</div>
            </div>
          </div>

          <nav className="space-y-2">
            {[
              { id: "analyze", label: "Analyze Live", icon: Radar },
              { id: "datasets", label: "Project Datasets", icon: FolderOpen },
              { id: "analytics", label: "Analytics", icon: BarChart3 },
              { id: "history", label: "History", icon: History },
              { id: "models", label: "Models", icon: BrainCircuit },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm transition ${
                  tab === id ? "bg-accent/15 text-accent-glow" : "text-white/70 hover:bg-white/5"
                }`}
                onClick={() => setTab(id as Tab)}
              >
                <Icon size={18} />
                {label}
              </button>
            ))}
          </nav>

          <div className="mt-8 space-y-4 border-t border-white/10 pt-6">
            <div>
              <label className="mb-2 block text-xs uppercase tracking-[0.18em] text-white/45">Processing Mode</label>
              <select className="input-shell" value={mode} onChange={(e) => setMode(e.target.value as ProcessingMode)}>
                {modeOptions.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-2 block text-xs uppercase tracking-[0.18em] text-white/45">Backend Model</label>
              <select className="input-shell" value={modelName} onChange={(e) => setModelName(e.target.value)}>
                {modelOptions.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <label className="flex items-center gap-3 text-sm text-white/70">
              <input type="checkbox" checked={gradcam} onChange={(e) => setGradcam(e.target.checked)} />
              Grad-CAM explainability
            </label>
          </div>

          <div className="mt-8 rounded-xl border border-white/10 bg-ink-900/70 p-4 text-xs text-white/50">
            <div className="mb-2 flex items-center gap-2 text-white/70">
              <Database size={14} />
              System Status
            </div>
            <div>{apiStatus}</div>
            <div className="mt-1">{supabaseConfigured ? "Supabase connected" : "Supabase env missing"}</div>
          </div>
        </aside>

        <main className="flex-1 space-y-6">
          <header className="glass-panel p-6">
            <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/10 px-3 py-1 text-xs text-accent-glow">
              <Sparkles size={14} />
              Production Sound Analytics Engine
            </div>
            <h1 className="text-3xl font-semibold tracking-tight">Validate, analyze, compare, and monitor</h1>
            <p className="mt-2 max-w-3xl text-sm text-white/60">
              Pre-inference validation, multi-model comparison, dataset ground-truth auditing, and live telemetry dashboards.
            </p>
          </header>

          {loading || previewLoading ? (
            <div className="glass-panel flex items-center justify-center p-10 text-white/70">
              <Activity className="mr-3 animate-pulse text-accent" />
              {previewLoading ? "Running validation preview..." : "Running inference pipeline..."}
            </div>
          ) : null}
          {error ? <div className="glass-panel border border-red-400/20 p-4 text-red-200">{error}</div> : null}

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
                <div className="glass-panel p-8 text-center text-white/55">
                  Upload or record audio to see validation checks and waveform preview before inference.
                </div>
              )}
              {comparison ? <ModelComparisonPanel comparison={comparison} /> : null}
              {result ? <AnalysisResults result={result} benchmarks={benchmarks} modelName={modelName} /> : null}
            </>
          ) : null}

          {tab === "datasets" ? (
            <>
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
              {comparison ? <ModelComparisonPanel comparison={comparison} /> : null}
              {result ? <AnalysisResults result={result} benchmarks={benchmarks} modelName={modelName} /> : null}
            </>
          ) : null}

          {tab === "analytics" ? <AnalyticsDashboardPanel /> : null}

          {tab === "history" ? (
            <section className="glass-panel p-5">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-xl font-semibold">Prediction History</h2>
                <button className="btn-secondary" onClick={loadHistory}>
                  Refresh
                </button>
              </div>
              {history.length === 0 ? (
                <p className="text-white/55">No predictions stored yet.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-left text-sm">
                    <thead className="text-white/45">
                      <tr>
                        <th className="px-3 py-2">Time</th>
                        <th className="px-3 py-2">Source</th>
                        <th className="px-3 py-2">Mode</th>
                        <th className="px-3 py-2">Prediction</th>
                        <th className="px-3 py-2">Confidence</th>
                        <th className="px-3 py-2">Reliability</th>
                        <th className="px-3 py-2">Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.map((row) => (
                        <tr key={row.id} className="border-t border-white/10 text-white/75">
                          <td className="px-3 py-3">{new Date(row.created_at).toLocaleString()}</td>
                          <td className="px-3 py-3">{row.input_source ?? "upload"}</td>
                          <td className="px-3 py-3">{row.processing_mode}</td>
                          <td className="px-3 py-3">{formatLabel(row.display_label || row.top_label)}</td>
                          <td className="px-3 py-3">{(Number(row.top_confidence) * 100).toFixed(1)}%</td>
                          <td className="px-3 py-3">{row.reliability_level ?? "—"}</td>
                          <td className="px-3 py-3">{row.inference_ms ? `${Number(row.inference_ms).toFixed(1)} ms` : "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          ) : null}

          {tab === "models" ? (
            <section className="grid gap-4 lg:grid-cols-3">
              {benchmarks.map((row) => (
                <div key={row.model_key} className={`glass-panel p-5 ${row.is_deployed ? "shadow-glow border-accent/30" : ""}`}>
                  <div className="mb-2 text-lg font-semibold">{row.display_name}</div>
                  {row.is_deployed ? (
                    <div className="mb-4 inline-flex rounded-full bg-accent/15 px-3 py-1 text-xs text-accent-glow">Deployed</div>
                  ) : null}
                  <div className="space-y-2 text-sm text-white/70">
                    <div>Accuracy: {row.test_accuracy ? `${(row.test_accuracy * 100).toFixed(1)}%` : "—"}</div>
                    <div>Macro F1: {row.test_macro_f1 ? row.test_macro_f1.toFixed(3) : "—"}</div>
                    <div>Latency: {row.inference_ms_mean} ms</div>
                    <div>Checkpoint: {row.model_file_size_mb} MB</div>
                  </div>
                </div>
              ))}
            </section>
          ) : null}
        </main>
      </div>
    </div>
  );
}
