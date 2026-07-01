import { useEffect, useState } from "react";
import { BarChart3, ClipboardList, Download, GitBranch, History, RefreshCw, Route } from "lucide-react";
import {
  exportSessionReport,
  fetchAnalyticsDashboard,
  type AnalyticsDashboard,
  type PredictResult,
  type PredictionHistoryRow,
} from "../lib/api";
import { getSessionId } from "../lib/session";
import { AnalyticsDashboardPanel } from "./AnalyticsDashboardPanel";
import { MetricCard } from "./MetricCard";
import { PredictionHistoryPanel } from "./PredictionHistoryPanel";
import { RouterLabPanel, type RouterLabContext } from "./RouterLabPanel";
import { SessionTimelinePanel } from "./SessionTimelinePanel";

type SubTab = "timeline" | "history" | "charts" | "router";

type Props = {
  active?: boolean;
  refreshKey?: string | null;
  routerContext: RouterLabContext | null;
  modelName: string;
  gradcam: boolean;
  onOpenRouterResult?: (result: PredictResult) => void;
};

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

const SUB_TABS: Array<{ id: SubTab; label: string; icon: typeof GitBranch }> = [
  { id: "timeline", label: "Timeline", icon: GitBranch },
  { id: "history", label: "History", icon: History },
  { id: "charts", label: "Charts", icon: BarChart3 },
  { id: "router", label: "Router", icon: Route },
];

export function SessionAuditPanel({
  active = true,
  refreshKey,
  routerContext,
  modelName,
  gradcam,
  onOpenRouterResult,
}: Props) {
  const [subTab, setSubTab] = useState<SubTab>("timeline");
  const [metrics, setMetrics] = useState<AnalyticsDashboard | null>(null);
  const [headerLoading, setHeaderLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [localRefreshKey, setLocalRefreshKey] = useState(0);
  const [selectedRouterRowId, setSelectedRouterRowId] = useState<string | null>(null);

  const sessionId = getSessionId();
  const shortSession = `${sessionId.slice(0, 8)}…`;
  const combinedRefreshKey = `${refreshKey ?? ""}-${localRefreshKey}`;

  async function loadHeaderMetrics() {
    setHeaderLoading(true);
    setError(null);
    try {
      setMetrics(await fetchAnalyticsDashboard());
    } catch (err) {
      setMetrics(null);
      setError(err instanceof Error ? err.message : "Failed to load session summary.");
    } finally {
      setHeaderLoading(false);
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

  function handleRefreshAll() {
    setLocalRefreshKey((key) => key + 1);
    void loadHeaderMetrics();
  }

  function handleAutoRowClick(row: PredictionHistoryRow) {
    setSelectedRouterRowId(row.id);
    setSubTab("router");
  }

  useEffect(() => {
    if (active) {
      void loadHeaderMetrics();
    }
  }, [active, refreshKey, localRefreshKey]);

  const routerRowMatches =
    selectedRouterRowId != null &&
    routerContext?.result.saved_prediction_id === selectedRouterRowId;

  const showRouterLab = routerContext?.result.router && (selectedRouterRowId == null || routerRowMatches);

  return (
    <div className="space-y-6">
      <div className="glass-panel p-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-48 h-48 bg-cyanGradient pointer-events-none z-0" />
        <div className="relative z-10 flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-accent/10 p-2 text-accent-soft border border-accent/20">
              <ClipboardList size={18} />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">Session & Audit</h2>
              <p className="text-xs text-white/50 mt-0.5">
                Session <span className="font-mono text-white/70">{shortSession}</span>
                {metrics ? ` · ${metrics.total_predictions} logged predictions` : null}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <button className="btn-secondary py-2 text-xs" onClick={handleRefreshAll} disabled={exporting}>
              <RefreshCw size={13} className="mr-1.5 inline" />
              Refresh
            </button>
            <button className="btn-primary py-2 text-xs" onClick={() => void handleExport()} disabled={exporting}>
              <Download size={13} className="mr-1.5 inline" />
              {exporting ? "Exporting…" : "Export session ZIP"}
            </button>
          </div>
        </div>

        <div className="relative z-10 mt-5 flex flex-wrap gap-2">
          {SUB_TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              className={`rounded-xl border px-3.5 py-2 text-xs font-semibold transition flex items-center gap-1.5 ${
                subTab === id
                  ? "border-accent/40 bg-accent/10 text-accent-soft"
                  : "border-white/[0.08] bg-white/[0.03] text-white/55 hover:text-white hover:bg-white/[0.06]"
              }`}
              onClick={() => setSubTab(id)}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </div>
      </div>

      {error ? (
        <div className="glass-panel border-status-error/30 bg-status-error/5 p-5 text-status-error text-sm font-medium flex items-center gap-3">
          <span className="h-2 w-2 rounded-full bg-status-error shadow-[0_0_10px_#f43f5e] shrink-0"></span>
          {error}
        </div>
      ) : null}

      {metrics && subTab !== "charts" && !headerLoading ? (
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

      {subTab === "timeline" ? (
        <SessionTimelinePanel
          active={active}
          refreshKey={combinedRefreshKey}
          embedded
          skipMetrics
          onAutoRowClick={handleAutoRowClick}
        />
      ) : null}

      {subTab === "history" ? (
        <PredictionHistoryPanel
          active={active}
          refreshKey={combinedRefreshKey}
          embedded
          onAutoRowClick={handleAutoRowClick}
        />
      ) : null}

      {subTab === "charts" ? (
        <AnalyticsDashboardPanel active={active} refreshKey={combinedRefreshKey} embedded />
      ) : null}

      {subTab === "router" ? (
        showRouterLab ? (
          <RouterLabPanel
            context={routerContext}
            modelName={modelName}
            gradcam={gradcam}
            embedded
            onOpenResult={onOpenRouterResult}
          />
        ) : (
          <div className="glass-panel p-12 text-center text-white/40 text-sm flex flex-col items-center gap-3">
            <Route size={36} className="text-white/20" />
            {selectedRouterRowId && !routerRowMatches ? (
              <p className="max-w-md leading-relaxed">
                Router details are available for your most recent auto-routed prediction. Re-run this clip with{" "}
                <span className="text-white/70 font-medium">Smart Auto-Router</span> from Analyze Live or Project
                Datasets, or click an auto-routed entry from your latest session run.
              </p>
            ) : (
              <p className="max-w-md leading-relaxed">
                Run a prediction with <span className="text-white/70 font-medium">Smart Auto-Router</span> on Analyze
                Live or Project Datasets. Auto-routed timeline and history rows open router transparency here.
              </p>
            )}
          </div>
        )
      ) : null}
    </div>
  );
}
