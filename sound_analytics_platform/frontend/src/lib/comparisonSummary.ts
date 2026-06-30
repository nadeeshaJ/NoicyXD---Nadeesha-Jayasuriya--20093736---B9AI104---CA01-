import type { ModelCompareResult } from "./api";

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export type ComparisonSummary = {
  fastest: ModelCompareResult["comparisons"][number] | null;
  mostConfident: ModelCompareResult["comparisons"][number];
  agreementPct: number;
  recommended: ModelCompareResult["comparisons"][number];
  majorityLabel: string;
  uniqueLabels: string[];
};

export function buildComparisonSummary(comparison: ModelCompareResult): ComparisonSummary | null {
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
  const majorityLabel =
    [...labelCounts.entries()].find(([, count]) => count === majorityCount)?.[0] ?? mostConfident.top_label;
  const uniqueLabels = [...labelCounts.keys()];

  const deployed = rows.find((r) => r.model_key === "mobilenetv2") ?? mostConfident;
  const recommended =
    comparison.effective_mode === "animal"
      ? deployed
      : fastest && fastest.model_key === "mobilenetv2"
        ? fastest
        : deployed;

  return { fastest, mostConfident, agreementPct, recommended, majorityLabel, uniqueLabels };
}

export function buildComparisonNarrative(comparison: ModelCompareResult): string {
  const summary = buildComparisonSummary(comparison);
  const rows = comparison.comparisons;
  if (!summary || rows.length === 0) {
    return "No model comparison results are available for this clip.";
  }

  const mode = comparison.effective_mode;
  const perModel = rows
    .map(
      (row) =>
        `${row.display_name} → ${formatLabel(row.top_label)} (${(row.top_confidence * 100).toFixed(1)}%)`,
    )
    .join("; ");

  const parts: string[] = [
    `${rows.length} CNN architectures classified the same audio in ${mode} mode: ${perModel}.`,
  ];

  if (summary.agreementPct === 100) {
    parts.push(`All models agree on ${formatLabel(summary.majorityLabel)}.`);
  } else {
    parts.push(
      `Top-label agreement is ${summary.agreementPct}% — predictions split across ${summary.uniqueLabels.map(formatLabel).join(", ")}.`,
    );
  }

  if (summary.fastest?.inference_ms != null) {
    parts.push(
      `${summary.fastest.display_name} was fastest (${summary.fastest.inference_ms.toFixed(1)} ms live inference).`,
    );
  }

  parts.push(
    `${summary.mostConfident.display_name} returned the highest softmax confidence. Suggested pick for this clip: ${summary.recommended.display_name}.`,
  );

  if (comparison.input_source === "dataset" && comparison.sample_id) {
    parts.push(`Use Play Sound to verify the dataset clip (${comparison.sample_id}) against these labels.`);
  } else {
    parts.push("Use Play Sound to check whether the winning label matches what you hear.");
  }

  return parts.join(" ");
}
