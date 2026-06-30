import { Gauge, Target, Timer, Users } from "lucide-react";
import type { ModelCompareResult } from "../lib/api";

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function buildSummary(comparison: ModelCompareResult) {
  const rows = comparison.comparisons;
  if (rows.length === 0) return null;

  const withLatency = rows.filter((r) => r.inference_ms != null);
  const fastest = withLatency.length
    ? withLatency.reduce((a, b) => ((a.inference_ms ?? Infinity) < (b.inference_ms ?? Infinity) ? a : b))
    : null;

  const mostConfident = rows.reduce((a, b) => (a.top_confidence >= b.top_confidence ? a : b));

  const labelCounts = new Map<string, number>();
  for (const row of rows) {
    labelCounts.set(row.top_label, (labelCounts.get(row.top_label) ?? 0) + 1);
  }
  const majorityCount = Math.max(...labelCounts.values());
  const agreementPct = Math.round((majorityCount / rows.length) * 100);

  const deployed = rows.find((r) => r.model_key === "mobilenetv2") ?? mostConfident;
  const recommended = comparison.effective_mode === "animal" ? deployed : (fastest && fastest.model_key === "mobilenetv2" ? fastest : deployed);

  return { fastest, mostConfident, agreementPct, recommended };
}

function WinnerTile({
  icon: Icon,
  label,
  value,
  hint,
}: {
  icon: typeof Gauge;
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
      <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-white/45 mb-2">
        <Icon size={12} />
        {label}
      </div>
      <div className="text-sm font-semibold text-white">{value}</div>
      {hint ? <div className="mt-1 text-[11px] text-white/40">{hint}</div> : null}
    </div>
  );
}

export function ComparisonWinnerCard({ comparison }: { comparison: ModelCompareResult }) {
  const summary = buildSummary(comparison);
  if (!summary) return null;

  const { fastest, mostConfident, agreementPct, recommended } = summary;

  return (
    <div className="mb-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <WinnerTile
        icon={Timer}
        label="Fastest"
        value={fastest?.display_name ?? "—"}
        hint={fastest?.inference_ms != null ? `${fastest.inference_ms.toFixed(1)} ms` : undefined}
      />
      <WinnerTile
        icon={Target}
        label="Highest confidence"
        value={mostConfident.display_name}
        hint={`${formatLabel(mostConfident.top_label)} · ${(mostConfident.top_confidence * 100).toFixed(1)}%`}
      />
      <WinnerTile
        icon={Users}
        label="Label agreement"
        value={`${agreementPct}%`}
        hint={agreementPct === 100 ? "All models agree" : "Models differ on top class"}
      />
      <WinnerTile
        icon={Gauge}
        label="Suggested pick"
        value={recommended.display_name}
        hint={comparison.effective_mode === "animal" ? "Animal expert checkpoint" : "Deployed urban model"}
      />
    </div>
  );
}
