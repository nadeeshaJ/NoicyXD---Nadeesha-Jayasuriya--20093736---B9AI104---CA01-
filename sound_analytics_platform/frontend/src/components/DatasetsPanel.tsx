import { Database, Play, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import {
  compareSampleModels,
  fetchCuratedSamples,
  fetchDatasetOverview,
  fetchDatasetSamples,
  predictFromSample,
  type DatasetOverview,
  type DatasetSample,
  type ModelCompareResult,
  type PredictResult,
  type ProcessingMode,
} from "../lib/api";

type Props = {
  mode: ProcessingMode;
  modelName: string;
  gradcam: boolean;
  onResult: (result: PredictResult) => void;
  onComparison: (comparison: ModelCompareResult) => void;
  onLoading: (loading: boolean) => void;
  onError: (message: string | null) => void;
};

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export function DatasetsPanel({ mode, modelName, gradcam, onResult, onComparison, onLoading, onError }: Props) {
  const [overview, setOverview] = useState<DatasetOverview[]>([]);
  const [domain, setDomain] = useState<"urban" | "animal">("urban");
  const [labelFilter, setLabelFilter] = useState("");
  const [curated, setCurated] = useState<DatasetSample[]>([]);
  const [samples, setSamples] = useState<DatasetSample[]>([]);

  useEffect(() => {
    fetchDatasetOverview().then(setOverview).catch(() => setOverview([]));
  }, []);

  useEffect(() => {
    fetchCuratedSamples(domain).then(setCurated).catch(() => setCurated([]));
    fetchDatasetSamples(domain, labelFilter || undefined).then(setSamples).catch(() => setSamples([]));
  }, [domain, labelFilter]);

  const activeOverview = overview.find((item) => item.domain === domain);

  async function analyzeSample(sample: DatasetSample) {
    onLoading(true);
    onError(null);
    try {
      const result = await predictFromSample({
        domain: sample.domain as "urban" | "animal",
        sampleId: sample.sample_id,
        mode: mode === "auto" ? "auto" : (sample.domain as "urban" | "animal"),
        modelName,
        gradcam,
      });
      onResult(result);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Failed to analyze dataset sample.");
    } finally {
      onLoading(false);
    }
  }

  async function compareSample(sample: DatasetSample) {
    onLoading(true);
    onError(null);
    try {
      onComparison(
        await compareSampleModels({
          domain: sample.domain as "urban" | "animal",
          sampleId: sample.sample_id,
          mode,
        }),
      );
    } catch (err) {
      onError(err instanceof Error ? err.message : "Failed to compare models on sample.");
    } finally {
      onLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <section className="glass-panel p-6">
        <div className="mb-4 flex items-center gap-2 text-accent-glow">
          <Database size={18} />
          <h2 className="text-xl font-semibold text-white">Project Data Sources</h2>
        </div>
        <p className="text-sm text-white/60">
          Analyze clips directly from your existing UrbanSound8K and ESC-50 test splits — the same data used in training
          and evaluation. Compare model predictions against known ground-truth labels.
        </p>

        <div className="mt-5 grid gap-4 md:grid-cols-2">
          {overview.map((item) => (
            <button
              key={item.domain}
              className={`rounded-2xl border p-4 text-left transition ${
                domain === item.domain ? "border-accent/40 bg-accent/10" : "border-white/10 bg-ink-900/60 hover:bg-white/5"
              }`}
              onClick={() => {
                setDomain(item.domain as "urban" | "animal");
                setLabelFilter("");
              }}
            >
              <div className="text-lg font-semibold">{item.title}</div>
              <div className="mt-2 text-sm text-white/60">{item.source}</div>
              <div className="mt-3 grid grid-cols-2 gap-2 text-sm text-white/75">
                <div>Total clips: {item.total_clips?.toLocaleString() ?? "—"}</div>
                <div>Test clips: {item.test_clips}</div>
                <div>Classes: {item.num_classes}</div>
                <div>Domain: {item.domain}</div>
              </div>
            </button>
          ))}
        </div>
      </section>

      {activeOverview ? (
        <section className="glass-panel p-5">
          <div className="mb-3 text-sm font-medium text-white/80">Supported classes in {activeOverview.title}</div>
          <div className="flex flex-wrap gap-2">
            {activeOverview.classes.map((label) => (
              <button
                key={label}
                className={`rounded-full px-3 py-1 text-xs ${
                  labelFilter === label ? "bg-accent text-ink-950" : "bg-white/10 text-white/70 hover:bg-white/15"
                }`}
                onClick={() => setLabelFilter(labelFilter === label ? "" : label)}
              >
                {formatLabel(label)}
              </button>
            ))}
          </div>
        </section>
      ) : null}

      <section className="glass-panel p-5">
        <div className="mb-4 flex items-center gap-2">
          <Sparkles size={16} className="text-accent" />
          <h3 className="text-lg font-medium">Recommended Demo Samples</h3>
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {curated.map((sample) => (
            <div key={sample.sample_id} className="rounded-xl border border-white/10 bg-ink-900/70 p-4">
              <div className="font-medium">{formatLabel(sample.label)}</div>
              <div className="mt-1 text-xs text-white/45">{sample.filename}</div>
              <div className="mt-2 text-sm text-white/60">{sample.note}</div>
              <div className="mt-4 flex gap-2">
                <button className="btn-primary flex-1" onClick={() => analyzeSample(sample)}>
                  <Play size={14} />
                  Analyze
                </button>
                <button className="btn-secondary flex-1" onClick={() => compareSample(sample)}>
                  Compare
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="glass-panel p-5">
        <h3 className="mb-4 text-lg font-medium">Browse Test Split Samples</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="text-white/45">
              <tr>
                <th className="px-3 py-2">Filename</th>
                <th className="px-3 py-2">Ground Truth</th>
                <th className="px-3 py-2">Action</th>
              </tr>
            </thead>
            <tbody>
              {samples.map((sample) => (
                <tr key={sample.sample_id} className="border-t border-white/10 text-white/75">
                  <td className="px-3 py-3 font-mono text-xs">{sample.filename}</td>
                  <td className="px-3 py-3">{formatLabel(sample.label)}</td>
                  <td className="px-3 py-3">
                    <div className="flex gap-2">
                      <button className="btn-secondary" onClick={() => analyzeSample(sample)}>
                        Analyze
                      </button>
                      <button className="btn-secondary" onClick={() => compareSample(sample)}>
                        Compare
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
