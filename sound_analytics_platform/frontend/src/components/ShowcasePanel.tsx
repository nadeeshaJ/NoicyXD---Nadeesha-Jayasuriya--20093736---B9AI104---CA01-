import { useEffect, useState } from "react";
import { Play, Sparkles } from "lucide-react";
import {
  fetchCuratedSamples,
  predictFromSample,
  type DatasetSample,
  type PredictResult,
  type ProcessingMode,
} from "../lib/api";

type DemoScenario = {
  id: string;
  title: string;
  description: string;
  domain: "urban" | "animal";
  mode: ProcessingMode;
  sampleLabel: string;
};

const SCENARIO_DEFS: DemoScenario[] = [
  {
    id: "urban-siren",
    title: "Urban siren",
    description: "UrbanSound8K siren clip · urban mode",
    domain: "urban",
    mode: "urban",
    sampleLabel: "siren",
  },
  {
    id: "urban-jackhammer",
    title: "Construction noise",
    description: "Jackhammer sample · urban mode",
    domain: "urban",
    mode: "urban",
    sampleLabel: "jackhammer",
  },
  {
    id: "animal-dog",
    title: "Animal dog bark",
    description: "ESC-50 dog vocalization · animal mode",
    domain: "animal",
    mode: "animal",
    sampleLabel: "dog",
  },
  {
    id: "auto-urban-dog-bark",
    title: "Auto-router · urban dog bark",
    description: "UrbanSound8K dog_bark · auto mode picks expert",
    domain: "urban",
    mode: "auto",
    sampleLabel: "dog_bark",
  },
  {
    id: "auto-animal-dog",
    title: "Auto-router · animal dog",
    description: "ESC-50 dog · auto mode picks expert",
    domain: "animal",
    mode: "auto",
    sampleLabel: "dog",
  },
];

type ResolvedScenario = DemoScenario & { sample?: DatasetSample };

type Props = {
  gradcam: boolean;
  disabled?: boolean;
  onResult: (result: PredictResult, sample?: DatasetSample) => void;
  onLoading: (loading: boolean) => void;
  onError: (message: string | null) => void;
};

export function ShowcasePanel({ gradcam, disabled = false, onResult, onLoading, onError }: Props) {
  const [scenarios, setScenarios] = useState<ResolvedScenario[]>([]);
  const [loadingMap, setLoadingMap] = useState<Record<string, boolean>>({});

  useEffect(() => {
    async function load() {
      try {
        const [urban, animal] = await Promise.all([
          fetchCuratedSamples("urban"),
          fetchCuratedSamples("animal"),
        ]);
        const byDomainLabel = (domain: "urban" | "animal", label: string) =>
          (domain === "urban" ? urban : animal).find((s) => s.label === label);

        setScenarios(
          SCENARIO_DEFS.map((def) => ({
            ...def,
            sample: byDomainLabel(def.domain, def.sampleLabel),
          })).filter((s) => s.sample),
        );
      } catch {
        setScenarios([]);
      }
    }
    void load();
  }, []);

  async function runScenario(scenario: ResolvedScenario) {
    if (!scenario.sample || disabled) return;
    setLoadingMap((m) => ({ ...m, [scenario.id]: true }));
    onLoading(true);
    onError(null);
    try {
      const result = await predictFromSample({
        domain: scenario.domain,
        sampleId: scenario.sample.sample_id,
        mode: scenario.mode,
        modelName: "mobilenetv2",
        gradcam,
      });
      onResult(result, scenario.sample);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Showcase run failed.");
    } finally {
      setLoadingMap((m) => ({ ...m, [scenario.id]: false }));
      onLoading(false);
    }
  }

  return (
    <section className="glass-panel p-6">
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-2">
          <Sparkles size={18} className="text-accent-soft" />
          <h2 className="text-xl font-bold text-white">Showcase</h2>
        </div>
        <p className="text-xs text-white/50 max-w-2xl leading-relaxed">
          One-click runs on curated test clips. Each scenario uses MobileNetV2 and opens the full analysis report.
        </p>
      </div>

      {scenarios.length === 0 ? (
        <p className="text-sm text-white/40 p-8 text-center border border-dashed border-white/5 rounded-2xl">
          Curated samples unavailable. Check dataset paths and API connection.
        </p>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {scenarios.map((scenario) => {
            const busy = loadingMap[scenario.id];
            return (
              <div
                key={scenario.id}
                className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5 flex flex-col justify-between gap-4"
              >
                <div>
                  <div className="text-sm font-bold text-white">{scenario.title}</div>
                  <p className="mt-1 text-xs text-white/45 leading-relaxed">{scenario.description}</p>
                  {scenario.sample ? (
                    <code className="mt-2 inline-block text-[10px] text-white/35 font-mono bg-black/20 px-2 py-0.5 rounded">
                      {scenario.sample.sample_id}
                    </code>
                  ) : null}
                </div>
                <button
                  className="btn-primary w-full py-2.5 text-xs font-bold flex items-center justify-center gap-2"
                  disabled={disabled || busy || !scenario.sample}
                  onClick={() => void runScenario(scenario)}
                >
                  <Play size={14} />
                  {busy ? "Running…" : "Run scenario"}
                </button>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
