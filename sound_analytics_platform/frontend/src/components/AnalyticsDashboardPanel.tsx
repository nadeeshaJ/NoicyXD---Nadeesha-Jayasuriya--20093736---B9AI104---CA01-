import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchAnalyticsDashboard, type AnalyticsDashboard } from "../lib/api";
import { MetricCard } from "./MetricCard";

function DistributionChart({
  title,
  data,
}: {
  title: string;
  data: Array<{ name: string; count: number }>;
}) {
  return (
    <div className="glass-panel p-5">
      <h3 className="mb-4 text-sm font-medium text-white/80">{title}</h3>
      {data.length === 0 ? (
        <p className="text-sm text-white/45">No data yet.</p>
      ) : (
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
              <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} allowDecimals={false} />
              <Tooltip contentStyle={{ background: "#111827", border: "1px solid rgba(255,255,255,0.1)" }} />
              <Bar dataKey="count" fill="#10b981" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export function AnalyticsDashboardPanel() {
  const [data, setData] = useState<AnalyticsDashboard | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      setData(await fetchAnalyticsDashboard());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load analytics.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  if (error) {
    return <div className="glass-panel border border-red-400/20 p-4 text-red-200">{error}</div>;
  }

  if (!data) {
    return <div className="glass-panel p-8 text-center text-white/55">Loading operational telemetry...</div>;
  }

  const latencyTrend = data.latency_trend.map((point, index) => ({
    index: index + 1,
    latency_ms: point.latency_ms,
    label: point.label,
  }));

  return (
    <div className="space-y-6">
      <section className="glass-panel p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white">Operational Dashboard</h2>
            <p className="mt-1 text-sm text-white/60">Live MLOps telemetry from Supabase inference logs for your session.</p>
          </div>
          <button className="btn-secondary" onClick={load}>
            Refresh
          </button>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <MetricCard label="Total Predictions" value={data.total_predictions} hint="Logged in this browser session" accent />
          <MetricCard
            label="Avg Latency"
            value={data.avg_latency_ms ? `${data.avg_latency_ms.toFixed(2)} ms` : "—"}
            hint="Mean inference time"
          />
          <MetricCard label="Last Hour Activity" value={data.predictions_last_hour} hint="Recent system utilization proxy" />
        </div>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <MetricCard label="Low Confidence Events" value={data.low_confidence_count} hint="Reliability Low or < 40%" />
          <MetricCard label="Unknown / Uncertain" value={data.unknown_count} hint="Below unknown threshold" />
        </div>
      </section>

      <section className="glass-panel p-5">
        <h3 className="mb-4 text-sm font-medium text-white/80">Latency Trend</h3>
        {latencyTrend.length === 0 ? (
          <p className="text-sm text-white/45">Run a few analyses to populate latency trends.</p>
        ) : (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={latencyTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="index" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#111827", border: "1px solid rgba(255,255,255,0.1)" }} />
                <Line type="monotone" dataKey="latency_ms" stroke="#34d399" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        <DistributionChart title="Urban Monitoring Events" data={data.urban_event_summary} />
        <DistributionChart title="Animal Monitoring Events" data={data.animal_event_summary} />
        <DistributionChart title="Predictions by Class" data={data.class_distribution} />
        <DistributionChart title="Predictions by Model" data={data.model_distribution} />
        <DistributionChart title="Predictions by Mode" data={data.mode_distribution} />
        <DistributionChart title="Predictions by Input Source" data={data.source_distribution} />
      </section>
    </div>
  );
}
