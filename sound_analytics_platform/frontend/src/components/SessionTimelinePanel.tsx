import { useEffect, useState } from "react";
import { Download, GitBranch, RefreshCw } from "lucide-react";
import {
  exportSessionReport,
  fetchAnalyticsDashboard,
  fetchHistoryFromApi,
  type AnalyticsDashboard,
  type PredictionHistoryRow,
} from "../lib/api";
import { getSessionId } from "../lib/session";
import { MetricCard } from "./MetricCard";
import { WaveLoader } from "./WaveLoader";

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatTime(iso: string) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

type Props = {
  active?: boolean;
  refreshKey?: string | null;
  embedded?: boolean;
  skipMetrics?: boolean;
  onAutoRowClick?: (row: PredictionHistoryRow) => void;
};

export function SessionTimelinePanel({
  active = true,
  refreshKey,
  embedded = false,
  skipMetrics = false,
  onAutoRowClick,
}: Props) {
  const [rows, setRows] = useState<PredictionHistoryRow[]>([]);
  const [metrics, setMetrics] = useState<AnalyticsDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sessionId = getSessionId();
  const shortSession = `${sessionId.slice(0, 8)}…`;

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [history, dashboard] = await Promise.all([
        fetchHistoryFromApi(80),
        fetchAnalyticsDashboard(),
      ]);
      setRows(history);
      setMetrics(dashboard);
    } catch (err) {
      setRows([]);
      setMetrics(null);
      setError(err instanceof Error ? err.message : "Failed to load session timeline.");
    } finally {
      setLoading(false);
    }
  }

  async function handleExport() {
    setExporting(true);
    setError(null);
    try {
      const blob = await exportSessionReport();
      downloadBlob(blob, `session_report_${sessionId.slice(0, 8)}.zip`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Session export failed.");
    } finally {
      setExporting(false);
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
        message="Building session timeline..."
        submessage="Merging prediction history with analytics for this browser session"
      />
    );
  }

  const isAutoRow = (row: PredictionHistoryRow) =>
    row.processing_mode === "auto" || Boolean(row.routed_domain);

  return (
    <div className={embedded ? undefined : "space-y-6"}>
      {!embedded ? (
        <div className="glass-panel p-6 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-48 h-48 bg-cyanGradient pointer-events-none z-0" />
          <div className="relative z-10 flex flex-wrap items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-accent/10 p-2 text-accent-soft border border-accent/20">
                <GitBranch size={18} />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white tracking-tight">Session Story Timeline</h2>
                <p className="text-xs text-white/50 mt-0.5">
                  Session <span className="font-mono text-white/70">{shortSession}</span> · {rows.length} logged predictions
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <button className="btn-secondary py-2 text-xs" onClick={load} disabled={exporting}>
                <RefreshCw size={13} className="mr-1.5 inline" />
                Refresh
              </button>
              <button className="btn-primary py-2 text-xs" onClick={handleExport} disabled={exporting}>
                <Download size={13} className="mr-1.5 inline" />
                {exporting ? "Exporting…" : "Export session ZIP"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {error ? (
        <div className="glass-panel border-status-error/30 bg-status-error/5 p-5 text-status-error text-sm font-medium flex items-center gap-3">
          <span className="h-2 w-2 rounded-full bg-status-error shadow-[0_0_10px_#f43f5e] shrink-0"></span>
          {error}
        </div>
      ) : null}

      {metrics && !skipMetrics ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard label="Total predictions" value={String(metrics.total_predictions)} hint="This session" />
          <MetricCard
            label="Avg latency"
            value={metrics.avg_latency_ms != null ? `${metrics.avg_latency_ms.toFixed(0)} ms` : "—"}
            hint="Mean inference time"
          />
          <MetricCard label="Last hour" value={String(metrics.predictions_last_hour)} hint="Recent activity" />
          <MetricCard
            label="Low confidence"
            value={String(metrics.low_confidence_count)}
            hint={`${metrics.unknown_count} unknown`}
          />
        </div>
      ) : null}

      {rows.length === 0 ? (
        <div className="glass-panel p-12 text-center text-white/40 text-sm">
          No predictions logged yet. Run Analyze Live or Project Datasets to build your session story.
        </div>
      ) : (
        <div className="glass-panel p-6">
          <h3 className="text-xs font-bold uppercase tracking-wider text-white/60 mb-6">Chronological activity</h3>
          <ol className="relative border-l border-white/[0.08] ml-3 space-y-6">
            {rows.map((row) => (
              <li key={row.id} className="relative pl-6">
                <span className="absolute -left-[5px] top-1.5 h-2.5 w-2.5 rounded-full bg-accent shadow-glow border-2 border-brand-dark" />
                <div
                  className={`rounded-xl border border-white/[0.05] bg-white/[0.02] p-4 hover:border-white/[0.08] transition ${
                    isAutoRow(row) && onAutoRowClick ? "cursor-pointer" : ""
                  }`}
                  onClick={isAutoRow(row) && onAutoRowClick ? () => onAutoRowClick(row) : undefined}
                  onKeyDown={
                    isAutoRow(row) && onAutoRowClick
                      ? (event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            onAutoRowClick(row);
                          }
                        }
                      : undefined
                  }
                  role={isAutoRow(row) && onAutoRowClick ? "button" : undefined}
                  tabIndex={isAutoRow(row) && onAutoRowClick ? 0 : undefined}
                >
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <div className="text-sm font-semibold text-white">
                        {formatLabel(row.top_label)}{" "}
                        <span className="text-accent-soft">{(row.top_confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="text-xs text-white/45 mt-1">{formatTime(row.created_at)}</div>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      <span className="rounded-lg border border-white/10 bg-white/[0.03] px-2 py-0.5 text-[10px] font-semibold uppercase text-white/60">
                        {row.processing_mode}
                      </span>
                      {row.routed_domain ? (
                        <span className="rounded-lg border border-cyan-glow/20 bg-cyan-glow/5 px-2 py-0.5 text-[10px] font-semibold uppercase text-cyan-glow">
                          → {row.routed_domain}
                        </span>
                      ) : null}
                    </div>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-white/40">
                    <span>Model: {row.model_key}</span>
                    <span>Source: {row.input_source}</span>
                    {row.inference_ms != null ? <span>{row.inference_ms.toFixed(0)} ms</span> : null}
                    {row.reliability_level ? <span>Reliability: {row.reliability_level}</span> : null}
                    {isAutoRow(row) && onAutoRowClick ? (
                      <span className="text-cyan-glow/80">Click for router details →</span>
                    ) : null}
                  </div>
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
