import type { ModelCompareResult } from "../lib/api";

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

type Props = {
  comparison: ModelCompareResult;
};

export function ModelComparisonPanel({ comparison }: Props) {
  return (
    <section className="glass-panel p-5">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-white">Same-Clip Multi-Model Comparison</h2>
        <p className="mt-1 text-sm text-white/60">
          All available models classified the same audio clip in <strong>{comparison.effective_mode}</strong> mode.
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="text-white/45">
            <tr>
              <th className="px-3 py-2">Model</th>
              <th className="px-3 py-2">Prediction</th>
              <th className="px-3 py-2">Confidence</th>
              <th className="px-3 py-2">Live Latency</th>
              <th className="px-3 py-2">Checkpoint</th>
              <th className="px-3 py-2">Benchmark Latency</th>
            </tr>
          </thead>
          <tbody>
            {comparison.comparisons.map((row, index) => (
              <tr key={row.model_key} className={`border-t border-white/10 ${index === 0 ? "bg-accent/5" : ""}`}>
                <td className="px-3 py-3 font-medium text-white">{row.display_name}</td>
                <td className="px-3 py-3">{formatLabel(row.top_label)}</td>
                <td className="px-3 py-3">{(row.top_confidence * 100).toFixed(1)}%</td>
                <td className="px-3 py-3">{row.inference_ms ? `${row.inference_ms.toFixed(2)} ms` : "—"}</td>
                <td className="px-3 py-3">{row.checkpoint_size_mb ? `${row.checkpoint_size_mb} MB` : "—"}</td>
                <td className="px-3 py-3">{row.benchmark_latency_ms ? `${row.benchmark_latency_ms} ms` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
