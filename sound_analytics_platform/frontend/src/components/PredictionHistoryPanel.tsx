import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, History, RefreshCw } from "lucide-react";
import { fetchHistoryFromApi, type PredictionHistoryRow } from "../lib/api";
import { hasGroundTruth } from "../lib/labelMatching";
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

type AuditFilter = "all" | "auditable" | "correct" | "mismatch";

type Props = {
  active?: boolean;
  refreshKey?: string | null;
};

function filterRows(rows: PredictionHistoryRow[], filter: AuditFilter): PredictionHistoryRow[] {
  if (filter === "all") return rows;
  if (filter === "auditable") return rows.filter((row) => hasGroundTruth(row));
  if (filter === "correct") return rows.filter((row) => hasGroundTruth(row) && row.audit_match === true);
  return rows.filter((row) => hasGroundTruth(row) && row.audit_match === false);
}

export function PredictionHistoryPanel({ active = true, refreshKey }: Props) {
  const [rows, setRows] = useState<PredictionHistoryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [auditFilter, setAuditFilter] = useState<AuditFilter>("all");

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

  const auditableRows = useMemo(() => rows.filter((row) => hasGroundTruth(row)), [rows]);
  const correctCount = useMemo(
    () => auditableRows.filter((row) => row.audit_match === true).length,
    [auditableRows],
  );
  const mismatchCount = useMemo(
    () => auditableRows.filter((row) => row.audit_match === false).length,
    [auditableRows],
  );
  const filteredRows = useMemo(() => filterRows(rows, auditFilter), [rows, auditFilter]);

  const filterOptions: Array<{ id: AuditFilter; label: string; count?: number }> = [
    { id: "all", label: "All", count: rows.length },
    { id: "auditable", label: "Dataset audits", count: auditableRows.length },
    { id: "correct", label: "Correct", count: correctCount },
    { id: "mismatch", label: "Mismatches", count: mismatchCount },
  ];

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
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white">Prediction History Logs</h2>
          <p className="text-xs text-white/50 mt-1">
            Saved inferences for this browser session, newest first.
            {auditableRows.length > 0 ? (
              <>
                {" "}
                Dataset audits: {correctCount} correct · {mismatchCount}{" "}
                {mismatchCount === 1 ? "mismatch" : "mismatches"}.
              </>
            ) : null}
          </p>
        </div>
        <button className="btn-secondary py-2 text-xs shrink-0" onClick={load} disabled={loading}>
          <RefreshCw size={13} className="mr-1.5 inline" />
          Refresh Logs
        </button>
      </div>

      {rows.length > 0 ? (
        <div className="mb-5 flex flex-wrap gap-2">
          {filterOptions.map(({ id, label, count }) => (
            <button
              key={id}
              type="button"
              className={`rounded-xl border px-3 py-1.5 text-[11px] font-semibold transition ${
                auditFilter === id
                  ? id === "mismatch"
                    ? "border-status-warning/40 bg-status-warning/10 text-status-warning"
                    : id === "correct"
                      ? "border-status-success/40 bg-status-success/10 text-status-success"
                      : "border-accent/40 bg-accent/10 text-accent-soft"
                  : "border-white/[0.08] bg-white/[0.03] text-white/55 hover:text-white hover:bg-white/[0.06]"
              }`}
              onClick={() => setAuditFilter(id)}
            >
              {label}
              {count != null ? ` (${count})` : ""}
            </button>
          ))}
        </div>
      ) : null}

      {rows.length === 0 ? (
        <div className="text-center p-8 border border-dashed border-white/5 rounded-2xl flex flex-col items-center gap-3">
          <History size={32} className="text-white/20" />
          <p className="text-white/40 text-sm">No predictions stored yet in this session.</p>
          <p className="text-xs text-white/35 max-w-md leading-relaxed">
            Run <span className="text-white/55">Analyze Live</span> or analyze a sample under
            {" "}<span className="text-white/55">Project Datasets</span> / <span className="text-white/55">Showcase</span>.
            Dataset runs include ground-truth auditing here.
          </p>
        </div>
      ) : filteredRows.length === 0 ? (
        <div className="text-center p-8 border border-dashed border-white/5 rounded-2xl text-white/40 text-sm">
          {auditFilter === "mismatch"
            ? "No mismatches in this session — all audited dataset predictions matched ground truth."
            : auditFilter === "correct"
              ? "No correct audited predictions to show yet."
              : "No dataset samples with ground truth in this filter."}
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-white/[0.05]">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-white/[0.02] text-white/40 font-semibold">
              <tr>
                <th className="px-4 py-3">Timestamp</th>
                <th className="px-4 py-3">Source</th>
                <th className="px-4 py-3">Ground truth</th>
                <th className="px-4 py-3">Audit</th>
                <th className="px-4 py-3">Mode</th>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Top guess</th>
                <th className="px-4 py-3">Confidence</th>
                <th className="px-4 py-3">Reliability</th>
                <th className="px-4 py-3 text-right">Inference</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05]">
              {filteredRows.map((row) => {
                const reliability = reliabilityFromRow(row);
                const auditable = hasGroundTruth(row);
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
                    <td className="px-4 py-4 whitespace-nowrap text-xs text-white/60">
                      {auditable ? formatLabel(row.ground_truth_label!) : "—"}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-xs">
                      {!auditable ? (
                        <span className="text-white/30">N/A</span>
                      ) : row.audit_match ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-status-success/10 text-status-success border border-status-success/20">
                          <CheckCircle2 size={11} />
                          Match
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-status-warning/10 text-status-warning border border-status-warning/20">
                          <AlertTriangle size={11} />
                          Mismatch
                        </span>
                      )}
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
