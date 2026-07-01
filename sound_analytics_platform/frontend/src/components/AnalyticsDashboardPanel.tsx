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
import { supabaseConfigured } from "../lib/supabase";
import { MetricCard } from "./MetricCard";
import { WaveLoader } from "./WaveLoader";
import { BarChart3, RefreshCw } from "lucide-react";

function DistributionChart({
  title,
  data,
  gradientId,
  color1,
  color2,
}: {
  title: string;
  data: Array<{ name: string; count: number }>;
  gradientId: string;
  color1: string;
  color2: string;
}) {
  return (
    <div className="glass-panel p-5 relative overflow-hidden">
      <h3 className="mb-4 text-xs font-bold uppercase tracking-wider text-white/60 relative z-10">{title}</h3>
      {data.length === 0 ? (
        <p className="text-xs text-white/40 p-6 text-center border border-dashed border-white/5 rounded-2xl relative z-10">No data available yet.</p>
      ) : (
        <div className="h-56 relative z-10 min-w-0">
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <BarChart data={data}>
              <defs>
                <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color1} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={color2} stopOpacity={0.15}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.03)" />
              <XAxis
                dataKey="name"
                tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
                axisLine={{ stroke: "rgba(255,255,255,0.05)" }}
                interval={0}
                angle={-20}
                textAnchor="end"
                height={50}
              />
              <YAxis tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }} axisLine={{ stroke: "rgba(255,255,255,0.05)" }} allowDecimals={false} />
              <Tooltip 
                contentStyle={{ 
                  background: "#0b0f19", 
                  border: "1px solid rgba(255,255,255,0.08)", 
                  borderRadius: "12px", 
                  boxShadow: "0 10px 30px rgba(0,0,0,0.5)" 
                }} 
              />
              <Bar dataKey="count" fill={`url(#${gradientId})`} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export function AnalyticsDashboardPanel({
  active = true,
  refreshKey,
  embedded = false,
}: {
  active?: boolean;
  refreshKey?: string | null;
  embedded?: boolean;
}) {
  const [data, setData] = useState<AnalyticsDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setData(await fetchAnalyticsDashboard());
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "Failed to load analytics.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (active) {
      void load();
    }
  }, [active, refreshKey]);

  if (loading) {
    return (
      <WaveLoader
        message="Loading analytics dashboard..."
        submessage="Aggregating session telemetry from prediction logs"
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

  if (!data) {
    return null;
  }

  const latencyTrend = data.latency_trend.map((point, index) => ({
    index: index + 1,
    latency_ms: point.latency_ms,
    label: point.label,
  }));

  const isEmpty = data.total_predictions === 0;

  return (
    <div className="space-y-6">
      {isEmpty ? (
        <div className="glass-panel p-8 text-center relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-full bg-glowGradient pointer-events-none z-0" />
          <div className="relative z-10 flex flex-col items-center gap-3 max-w-lg mx-auto">
            <BarChart3 size={36} className="text-white/20" />
            <h3 className="text-lg font-bold text-white">No session telemetry yet</h3>
            <p className="text-xs text-white/50 leading-relaxed">
              Run predictions from <span className="text-white/70">Analyze Live</span> or
              {" "}<span className="text-white/70">Project Datasets</span> to populate this dashboard.
              Each saved inference adds latency trends, class distributions, and monitoring summaries for your browser session.
            </p>
            {!supabaseConfigured ? (
              <p className="text-[11px] text-status-warning/90 leading-relaxed">
                Supabase is not configured in the frontend env — the backend still logs predictions when API keys are set server-side.
              </p>
            ) : null}
          </div>
        </div>
      ) : null}

      <section className="glass-panel p-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-80 h-full bg-glowGradient pointer-events-none z-0" />
        
        {!embedded ? (
          <div className="mb-6 flex items-center justify-between relative z-10">
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">Telemetry Operational Dashboard</h2>
              <p className="text-xs text-white/50 leading-relaxed">Live MLOps telemetry aggregated from prediction logs for this browser session.</p>
            </div>
            <button className="btn-secondary py-2 text-xs" onClick={load} disabled={loading}>
              <RefreshCw size={13} className="mr-1.5" />
              Sync Logs
            </button>
          </div>
        ) : (
          <div className="mb-6 relative z-10">
            <h3 className="text-xs font-bold uppercase tracking-wider text-white/60">Session charts</h3>
          </div>
        )}

        <div className="grid gap-4 md:grid-cols-3 relative z-10">
          <MetricCard label="Session Predictions" value={data.total_predictions} hint="Total logged events" accent />
          <MetricCard
            label="Average Latency"
            value={data.avg_latency_ms ? `${data.avg_latency_ms.toFixed(1)} ms` : "—"}
            hint="Mean inference duration"
          />
          <MetricCard label="Last Hour Activity" value={data.predictions_last_hour} hint="Inference requests / hour" />
        </div>
        <div className="mt-4 grid gap-4 md:grid-cols-2 relative z-10">
          <MetricCard label="Low Confidence Detections" value={data.low_confidence_count} hint="Probability triggers < 40%" />
          <MetricCard label="Out-of-Distribution Events" value={data.unknown_count} hint="Flagged unknown by calibrated confidence" />
        </div>
      </section>

      <section className="glass-panel p-6 relative overflow-hidden">
        <h3 className="mb-4 text-xs font-bold uppercase tracking-wider text-white/60">Inference Latency Trend (Recent runs)</h3>
        {latencyTrend.length === 0 ? (
          <p className="text-xs text-white/40 p-6 text-center border border-dashed border-white/5 rounded-2xl">Perform predictions to populate latency charts.</p>
        ) : (
          <div className="h-64 relative z-10 min-w-0">
            <ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <LineChart data={latencyTrend}>
                <defs>
                  <linearGradient id="latencyGlow" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="index" tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }} axisLine={{ stroke: "rgba(255,255,255,0.05)" }} />
                <YAxis tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }} axisLine={{ stroke: "rgba(255,255,255,0.05)" }} />
                <Tooltip 
                  contentStyle={{ 
                    background: "#0b0f19", 
                    border: "1px solid rgba(255,255,255,0.08)", 
                    borderRadius: "12px", 
                    boxShadow: "0 10px 30px rgba(0,0,0,0.5)" 
                  }} 
                />
                <Line type="monotone" dataKey="latency_ms" stroke="#22d3ee" strokeWidth={3} dot={{ r: 4, fill: "#22d3ee", strokeWidth: 0 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      <section className="grid gap-6 xl:grid-cols-2">
        <DistributionChart title="Urban Monitoring Events" data={data.urban_event_summary} gradientId="urbanGrad" color1="#8b5cf6" color2="#6366f1" />
        <DistributionChart title="Animal Monitoring Events" data={data.animal_event_summary} gradientId="animalGrad" color1="#22d3ee" color2="#06b6d4" />
        <DistributionChart title="Predictions by Class" data={data.class_distribution} gradientId="classGrad" color1="#a78bfa" color2="#8b5cf6" />
        <DistributionChart title="Predictions by Model" data={data.model_distribution} gradientId="modelGrad" color1="#10b981" color2="#059669" />
        <DistributionChart title="Predictions by Mode" data={data.mode_distribution} gradientId="modeGrad" color1="#f59e0b" color2="#d97706" />
        <DistributionChart title="Predictions by Input Source" data={data.source_distribution} gradientId="srcGrad" color1="#f43f5e" color2="#e11d48" />
      </section>
    </div>
  );
}
