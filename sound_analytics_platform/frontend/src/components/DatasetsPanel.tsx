import { Database, BarChart, HardDrive, Volume2, Square } from "lucide-react";
import { useEffect, useState } from "react";
import {
  compareSampleModels,
  fetchDatasetOverview,
  fetchDatasetSamples,
  predictFromSample,
  type DatasetOverview,
  type DatasetSample,
  type ModelCompareResult,
  type PredictResult,
  API_BASE,
} from "../lib/api";

import { CuratedDemoPanel } from "./CuratedDemoPanel";

type DatasetLoadingAction = "analyze" | "compare" | "showcase";

type Props = {
  modelName: string;
  gradcam: boolean;
  disabled?: boolean;
  onResult: (result: PredictResult, sample?: DatasetSample) => void;
  onComparison: (comparison: ModelCompareResult, sample?: DatasetSample) => void;
  onLoading: (loading: boolean, action?: DatasetLoadingAction) => void;
  onDomainChange: (domain: "urban" | "animal") => void;
  onError: (message: string | null) => void;
};

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export function DatasetsPanel({
  modelName,
  gradcam,
  disabled = false,
  onResult,
  onComparison,
  onLoading,
  onDomainChange,
  onError,
}: Props) {
  const [overview, setOverview] = useState<DatasetOverview[]>([]);
  const [domain, setDomain] = useState<"urban" | "animal">("urban");

  useEffect(() => {
    onDomainChange(domain);
  }, [domain, onDomainChange]);
  const [labelFilter, setLabelFilter] = useState("");
  const [samples, setSamples] = useState<DatasetSample[]>([]);

  useEffect(() => {
    fetchDatasetOverview().then(setOverview).catch(() => setOverview([]));
  }, []);

  useEffect(() => {
    fetchDatasetSamples(domain, labelFilter || undefined).then(setSamples).catch(() => setSamples([]));
  }, [domain, labelFilter]);

  const activeOverview = overview.find((item) => item.domain === domain);

  async function analyzeSample(sample: DatasetSample) {
    onLoading(true, "analyze");
    onError(null);
    try {
      const sampleDomain = sample.domain as "urban" | "animal";
      const result = await predictFromSample({
        domain: sampleDomain,
        sampleId: sample.sample_id,
        mode: sampleDomain,
        modelName,
        gradcam,
      });
      onResult(result, sample);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Failed to analyze dataset sample.");
    } finally {
      onLoading(false);
    }
  }

  async function compareSample(sample: DatasetSample) {
    onLoading(true, "compare");
    onError(null);
    try {
      const sampleDomain = sample.domain as "urban" | "animal";
      onComparison(
        await compareSampleModels({
          domain: sampleDomain,
          sampleId: sample.sample_id,
          mode: sampleDomain,
        }),
        sample,
      );
    } catch (err) {
      onError(err instanceof Error ? err.message : "Failed to compare models on sample.");
    } finally {
      onLoading(false);
    }
  }

  const [activeAudio, setActiveAudio] = useState<HTMLAudioElement | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      activeAudio?.pause();
    };
  }, [activeAudio]);

  function togglePlaySample(sample: DatasetSample) {
    if (playingId === sample.sample_id) {
      activeAudio?.pause();
      setActiveAudio(null);
      setPlayingId(null);
      return;
    }

    activeAudio?.pause();

    const url = `${API_BASE}/api/datasets/${sample.domain}/samples/${sample.sample_id}/audio`;
    const audio = new Audio(url);
    audio.play().catch((err) => console.error("Audio playback failed", err));
    setActiveAudio(audio);
    setPlayingId(sample.sample_id);
    
    audio.onended = () => {
      setActiveAudio(null);
      setPlayingId(null);
    };
  }

  return (
    <div className="space-y-6">
      <section className="glass-panel p-6 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-64 h-64 bg-glowGradient pointer-events-none z-0" />
        
        <div className="relative z-10 mb-4 flex items-center gap-3">
          <div className="rounded-xl bg-accent/10 p-2.5 text-accent-soft border border-accent/20 shadow-glow">
            <Database size={18} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">Project Audio Datasets</h2>
            <p className="text-xs text-white/50">Run evaluations on test splits from UrbanSound8K and ESC-50 animals.</p>
          </div>
        </div>

        <div className="mt-6 grid gap-5 md:grid-cols-2 relative z-10">
          {overview.map((item) => {
            const isActive = domain === item.domain;
            return (
              <button
                key={item.domain}
                className={`rounded-2xl border p-5 text-left transition duration-300 ${
                  isActive 
                    ? "border-accent/40 bg-accent/[0.03] shadow-glow" 
                    : "border-white/[0.05] bg-white/[0.01] hover:bg-white/[0.03] hover:border-white/[0.08]"
                }`}
                onClick={() => {
                  if (disabled) return;
                  setDomain(item.domain as "urban" | "animal");
                  setLabelFilter("");
                }}
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className={`h-1.5 w-1.5 rounded-full ${isActive ? "bg-accent-soft shadow-[0_0_8px_#8b5cf6]" : "bg-white/20"}`}></span>
                  <div className="text-lg font-bold text-white tracking-tight">{item.title}</div>
                </div>
                <div className="text-xs text-white/40 mb-4 font-medium">{item.source}</div>
                
                <div className="grid grid-cols-2 gap-3 text-xs text-white/70">
                  <div className="flex items-center gap-2">
                    <HardDrive size={13} className="text-white/30" />
                    <span>Total Clips: {item.total_clips?.toLocaleString() ?? "—"}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <BarChart size={13} className="text-white/30" />
                    <span>Test Split: {item.test_clips}</span>
                  </div>
                  <div>Classes: {item.num_classes}</div>
                  <div className="capitalize">Domain: {item.domain}</div>
                </div>
              </button>
            );
          })}
        </div>
      </section>

      <CuratedDemoPanel
        modelName={modelName}
        gradcam={gradcam}
        disabled={disabled}
        onResult={onResult}
        onComparison={onComparison}
        onLoading={onLoading}
        onError={onError}
      />

      {activeOverview ? (
        <section className="glass-panel p-6">
          <h3 className="mb-3.5 text-xs font-bold uppercase tracking-wider text-white/60">Class Distribution Filter</h3>
          <div className="flex flex-wrap gap-2">
            {activeOverview.classes.map((label) => {
              const isActive = labelFilter === label;
              return (
                <button
                  key={label}
                  className={`rounded-xl px-3.5 py-1.5 text-xs font-semibold transition duration-300 border ${
                    isActive 
                      ? "bg-accent text-white border-accent shadow-glow" 
                      : "bg-white/[0.02] border-white/[0.05] text-white/60 hover:text-white hover:border-white/10 hover:bg-white/[0.04]"
                  }`}
                  onClick={() => !disabled && setLabelFilter(isActive ? "" : label)}
                  disabled={disabled}
                >
                  {formatLabel(label)}
                </button>
              );
            })}
          </div>
        </section>
      ) : null}

      <section className="glass-panel p-6">
        <h3 className="mb-5 text-sm font-bold uppercase tracking-wider text-white">Browse Test Split Table</h3>
        <div className="overflow-x-auto rounded-xl border border-white/[0.05]">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-white/[0.02] text-white/40 font-semibold">
              <tr>
                <th className="px-4 py-3">Filename</th>
                <th className="px-4 py-3">Ground Truth</th>
                <th className="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05]">
              {samples.map((sample) => (
                <tr key={sample.sample_id} className="hover:bg-white/[0.02] text-white/70 transition">
                  <td className="px-4 py-3 font-mono text-xs text-white/50">
                    <div className="flex items-center gap-2.5">
                      <button 
                        className={`p-1.5 rounded-lg border transition ${
                          playingId === sample.sample_id 
                            ? "bg-status-success/20 border-status-success/40 text-status-success shadow-glow animate-pulse" 
                            : "bg-white/[0.03] border-white/[0.05] text-white/40 hover:text-white hover:border-white/10"
                        }`}
                        onClick={() => togglePlaySample(sample)}
                        disabled={disabled}
                        title={playingId === sample.sample_id ? "Stop audio" : "Play audio"}
                      >
                        {playingId === sample.sample_id ? <Square size={11} className="fill-current" /> : <Volume2 size={11} />}
                      </button>
                      <span>{sample.filename}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 font-semibold text-white">{formatLabel(sample.label)}</td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2.5 justify-end">
                      <button className="btn-secondary py-1.5 px-3 text-xs" onClick={() => analyzeSample(sample)} disabled={disabled}>
                        Analyze
                      </button>
                      <button className="btn-secondary py-1.5 px-3 text-xs" onClick={() => compareSample(sample)} disabled={disabled}>
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
