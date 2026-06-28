import clsx from "clsx";
import type { ReactNode } from "react";

export function MetricCard({
  label,
  value,
  hint,
  accent = false,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <div className={clsx("metric-card", accent && "shadow-glow border-accent/30")}>
      <div className="text-xs uppercase tracking-[0.2em] text-white/50">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
      {hint ? <div className="mt-1 text-xs text-white/45">{hint}</div> : null}
    </div>
  );
}
