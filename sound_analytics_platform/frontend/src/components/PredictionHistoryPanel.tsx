import { useEffect, useState } from "react";
import { History, RefreshCw } from "lucide-react";
import { fetchHistoryFromApi, type PredictionHistoryRow } from "../lib/api";
import { WaveLoader } from "./WaveLoader";

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatModelName(modelKey: string) {
  if (modelKey === "mobilenetv2") return "MobileNetV2";
  if (modelKey === "custom_cnn") return "Custom CNN";
  if (modelKey === "resnet50") return "ResNet50";
  return formatLabel(modelKey);
}

function reliabilityFromRow(row: PredictionHistoryRow): string {
  if (row.reliability_level) return row.reliability_level;
  const confidence = Number(row.top_confidence ?? 0);
  if (confidence < 0.4 || row.is_unknown) return "Low";
  if (confidence >= 0.7) return "High";
  return "Medium";
}

function reliabilityClass(level: string) {
  if (level === "High") return "bg-status-success/10 text-status-success border border-status-success/20";
  if (level === "Medium") return "bg-status-warning/10 text-status-warning border border-status-warning/20";
  return "bg-status-error/10 text-status-error border border-status-error/20";
}

type Props = {
  active?: boolean;
  refreshKey?: string | null;
};

export function PredictionHistoryPanel({ active = true, refreshKey }: Props) {
  const [rows, setRows] = useState<PredictionHistoryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setRows(await fetchHistoryFromApi());
    } catch (err) {
      setRows([]);
      setError(err instanceof Error ? err.message : "Failed to load prediction history.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (active) {
      load();
    }
  }, [active, refreshKey]);

  if (loading) {
    return (
      <WaveLoader
        message="Loading prediction history..."
        submessage="Fetching saved inference logs for this browser session"
      />
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <div className="glass-panel border-status-error/30 bg-status-error/5 p-5 text-status-error text-sm font-medium flex items-center gap-3">
          <span className="h-2 w-2 rounded-full bg-status-error shadow-[0_0_10px_#f43f5e] shrink-0"></span>
          {error}
        </div>
        <button className="btn-secondary py-2 text-xs" onClick={load}>
          <RefreshCw size={13} className="mr-1.5 inline" />
          Retry
        </button>
      </div>
    );
  }

  return (
    <section className="glass-panel p-6">
      <div className="mb-6 flex items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white">Prediction History Logs</h2>
          <p className="text-xs text-white/50">Saved inferences for this browser session, newest first.</p>
        </div>
        <button className="btn-secondary py-2 text-xs shrink-0" onClick={load} disabled={loading}>
          <RefreshCw size={13} className="mr-1.5 inline" />
          Refresh Logs
        </button>
      </div>

      {rows.length === 0 ? (
        <div className="text-center p-8 border border-dashed border-white/5 rounded-2xl flex flex-col items-center gap-3">
          <History size={32} className="text-white/20" />
          <p className="text-white/40 text-sm">No predictions stored yet in this session.</p>
          <p className="text-xs text-white/35 max-w-md leading-relaxed">
            Run <span className="text-white/55">Analyze Live</span> or analyze a sample under
            {" "}<span className="text-white/55">Project Datasets</span>. Each completed inference is logged here automatically.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-white/[0.05]">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-white/[0.02] text-white/40 font-semibold">
              <tr>
                <th className="px-4 py-3">Timestamp</th>
                <th className="px-4 py-3">Source</th>
                <th className="px-4 py-3">Mode</th>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Top Guess</th>
                <th className="px-4 py-3">Confidence</th>
                <th className="px-4 py-3">Reliability</th>
                <th className="px-4 py-3 text-right">Inference</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05]">
              {rows.map((row) => {
                const reliability = reliabilityFromRow(row);
                return (
                  <tr key={row.id} className="hover:bg-white/[0.02] text-white/70 transition">
                    <td className="px-4 py-4 whitespace-nowrap text-xs text-white/50">
                      {new Date(row.created_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap capitalize text-xs">
                      <span className="px-2 py-1 rounded bg-white/[0.04] border border-white/[0.05]">
                        {row.input_source ?? "upload"}
                      </span>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap capitalize text-xs">{row.processing_mode}</td>
                    <td className="px-4 py-4 whitespace-nowrap text-xs text-white/60">
                      {formatModelName(row.model_key)}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap font-medium text-white">
                      {formatLabel(row.display_label || row.top_label)}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-xs">
                      {(Number(row.top_confidence) * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-xs">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase ${reliabilityClass(reliability)}`}>
                        {reliability}
                      </span>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-right text-xs font-mono text-white/50">
                      {row.inference_ms ? `${Number(row.inference_ms).toFixed(1)} ms` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
