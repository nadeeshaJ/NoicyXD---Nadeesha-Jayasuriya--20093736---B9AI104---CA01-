import { useState } from "react";
import { X } from "lucide-react";

const STORAGE_KEY = "sap-help-dismissed";

export function AppHelpBanner() {
  const [visible, setVisible] = useState(() => localStorage.getItem(STORAGE_KEY) !== "1");

  if (!visible) return null;

  function dismiss() {
    localStorage.setItem(STORAGE_KEY, "1");
    setVisible(false);
  }

  return (
    <div className="glass-panel border-accent/20 bg-accent/[0.04] p-4 flex flex-wrap items-start justify-between gap-3 text-sm sap-help-banner">
      <div className="text-white/70 text-xs leading-relaxed max-w-3xl">
        <span className="text-white font-semibold">Quick map:</span>{" "}
        Analyze Live for uploads · Project Datasets for labelled clips and curated explainable demos ·
        Session & Audit for timeline, history, charts, and router transparency ·
        CNN Models lists benchmark stats.
      </div>
      <button
        type="button"
        className="shrink-0 rounded-lg p-1.5 text-white/40 hover:text-white hover:bg-white/[0.06] transition"
        onClick={dismiss}
        aria-label="Dismiss"
      >
        <X size={16} />
      </button>
    </div>
  );
}
