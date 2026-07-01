import { useEffect, useState } from "react";
import { BarChart, Play, Sparkles, Square, Volume2 } from "lucide-react";
import {
  compareSampleModels,
  fetchCuratedSamples,
  predictFromSample,
  API_BASE,
  type DatasetSample,
  type ModelCompareResult,
  type PredictResult,
  type ProcessingMode,
} from "../lib/api";

type ExplainableDemo = {
  id: string;
  title: string;
  description: string;
  domain: "urban" | "animal";
  mode: ProcessingMode;
  sampleLabel: string;
};

const EXPLAINABLE_DEMOS: ExplainableDemo[] = [
  {
    id: "urban-siren",
    title: "Urban siren",
    description:
      "Classic emergency siren from UrbanSound8K. Ground truth: siren. Run in urban mode to see the city-noise expert classify emergency signals.",
    domain: "urban",
    mode: "urban",
    sampleLabel: "siren",
  },
  {
    id: "urban-jackhammer",
    title: "Construction noise",
    description:
      "Jackhammer clip from UrbanSound8K. Ground truth: jackhammer. Heavy machinery and construction-site sounds in the urban class set.",
    domain: "urban",
    mode: "urban",
    sampleLabel: "jackhammer",
  },
  {
    id: "urban-car-horn",
    title: "Traffic horn",
    description:
      "Car horn from the urban test split. Ground truth: car_horn. Distinct from sirens — tests fine-grained traffic-noise discrimination.",
    domain: "urban",
    mode: "urban",
    sampleLabel: "car_horn",
  },
  {
    id: "urban-street-music",
    title: "Street music",
    description:
      "Street performance clip from UrbanSound8K. Ground truth: street_music. Human-made ambient sound — a different urban texture from machines or traffic.",
    domain: "urban",
    mode: "urban",
    sampleLabel: "street_music",
  },
  {
    id: "auto-urban-dog-bark",
    title: "Auto-router · urban dog bark",
    description:
      "UrbanSound8K dog_bark clip in Smart Auto-Router mode. The same sound class exists in both datasets — watch the router pick urban vs animal experts in Session & Audit.",
    domain: "urban",
    mode: "auto",
    sampleLabel: "dog_bark",
  },
  {
    id: "auto-animal-dog",
    title: "Auto-router · animal dog",
    description:
      "ESC-50 dog vocalization in Smart Auto-Router mode. Tests routing on an animal-dataset clip where the urban expert might still react to bark-like sounds.",
    domain: "animal",
    mode: "auto",
    sampleLabel: "dog",
  },
  {
    id: "animal-cow",
    title: "Farm cow moo",
    description:
      "ESC-50 cow vocalization. Ground truth: cow. Clear livestock sound — run in animal mode for the ESC-50 MobileNetV2 expert path.",
    domain: "animal",
    mode: "animal",
    sampleLabel: "cow",
  },
  {
    id: "animal-frog",
    title: "Frog croak",
    description:
      "ESC-50 frog call from the animal test split. Ground truth: frog. Small-animal vocalization — distinct from mammals and birds for class diversity.",
    domain: "animal",
    mode: "animal",
    sampleLabel: "frog",
  },
];

type ResolvedDemo = ExplainableDemo & { sample?: DatasetSample };

type Props = {
  modelName: string;
  gradcam: boolean;
  disabled?: boolean;
  onResult: (result: PredictResult, sample?: DatasetSample) => void;
  onComparison: (comparison: ModelCompareResult, sample?: DatasetSample) => void;
  onLoading: (loading: boolean, action?: "analyze" | "compare" | "showcase") => void;
  onError: (message: string | null) => void;
};

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function modeBadgeClass(mode: ProcessingMode) {
  if (mode === "urban") return "border-accent/30 text-accent-soft bg-accent/5";
  if (mode === "animal") return "border-cyan-glow/30 text-cyan-glow bg-cyan-glow/5";
  return "border-status-warning/30 text-status-warning bg-status-warning/5";
}

function modeLabel(mode: ProcessingMode) {
  if (mode === "auto") return "Smart Auto-Router";
  if (mode === "animal") return "Animal mode";
  return "Urban mode";
}

export function CuratedDemoPanel({
  modelName,
  gradcam,
  disabled = false,
  onResult,
  onComparison,
  onLoading,
  onError,
}: Props) {
  const [demos, setDemos] = useState<ResolvedDemo[]>([]);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [activeAudio, setActiveAudio] = useState<HTMLAudioElement | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [urban, animal] = await Promise.all([
          fetchCuratedSamples("urban"),
          fetchCuratedSamples("animal"),
        ]);
        const byDomainLabel = (domain: "urban" | "animal", label: string) =>
          (domain === "urban" ? urban : animal).find((sample) => sample.label === label);

        setDemos(
          EXPLAINABLE_DEMOS.map((demo) => ({
            ...demo,
            sample: byDomainLabel(demo.domain, demo.sampleLabel),
          })).filter((demo) => demo.sample),
        );
      } catch {
        setDemos([]);
      }
    }
    void load();
  }, []);

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

  async function analyzeDemo(demo: ResolvedDemo) {
    if (!demo.sample || disabled) return;
    setBusyId(demo.id);
    onLoading(true, "showcase");
    onError(null);
    try {
      const effectiveModel =
        demo.mode === "animal" || demo.mode === "auto" ? "mobilenetv2" : modelName;
      const result = await predictFromSample({
        domain: demo.domain,
        sampleId: demo.sample.sample_id,
        mode: demo.mode,
        modelName: effectiveModel,
        gradcam,
      });
      onResult(result, demo.sample);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Demo analysis failed.");
    } finally {
      setBusyId(null);
      onLoading(false);
    }
  }

  async function compareDemo(demo: ResolvedDemo) {
    if (!demo.sample || disabled || demo.mode === "auto") return;
    setBusyId(`${demo.id}-compare`);
    onLoading(true, "compare");
    onError(null);
    try {
      onComparison(
        await compareSampleModels({
          domain: demo.domain,
          sampleId: demo.sample.sample_id,
          mode: demo.mode,
        }),
        demo.sample,
      );
    } catch (err) {
      onError(err instanceof Error ? err.message : "Demo comparison failed.");
    } finally {
      setBusyId(null);
      onLoading(false);
    }
  }

  return (
    <section className="glass-panel p-6">
      <div className="mb-6">
        <div className="flex items-center gap-2.5 mb-2">
          <Sparkles size={18} className="text-accent-soft" />
          <h3 className="text-sm font-bold uppercase tracking-wider text-white">Curated Explainable Samples</h3>
        </div>
        <p className="text-xs text-white/50 max-w-3xl leading-relaxed">
          Hand-picked clips from UrbanSound8K and ESC-50 with known ground truth. Each card explains what the run
          tests — urban expert, animal expert, or Smart Auto-Router — and opens a full report with Grad-CAM.
        </p>
      </div>

      {demos.length === 0 ? (
        <p className="text-sm text-white/40 p-8 text-center border border-dashed border-white/5 rounded-2xl">
          Curated samples unavailable. Check dataset paths and API connection.
        </p>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {demos.map((demo) => {
            const busy = busyId === demo.id;
            const compareBusy = busyId === `${demo.id}-compare`;
            const sample = demo.sample!;
            return (
              <div
                key={demo.id}
                className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5 flex flex-col justify-between gap-4 hover:border-white/[0.1] transition"
              >
                <div>
                  <div className="flex flex-wrap items-center gap-2 mb-2">
                    <span className="text-sm font-bold text-white">{demo.title}</span>
                    <span
                      className={`rounded-lg border px-2 py-0.5 text-[10px] font-semibold uppercase ${modeBadgeClass(demo.mode)}`}
                    >
                      {modeLabel(demo.mode)}
                    </span>
                  </div>
                  <p className="text-xs text-white/50 leading-relaxed">{demo.description}</p>
                  <div className="mt-3 flex flex-wrap gap-2 text-[10px]">
                    <span className="rounded-md border border-white/10 bg-white/[0.03] px-2 py-0.5 text-white/55 capitalize">
                      {demo.domain} dataset
                    </span>
                    <span className="rounded-md border border-white/10 bg-white/[0.03] px-2 py-0.5 text-white/55">
                      Ground truth: {formatLabel(sample.label)}
                    </span>
                  </div>
                </div>

                <div className="flex gap-2.5 items-center">
                  <button
                    type="button"
                    className={`p-2.5 rounded-xl border transition flex items-center justify-center shrink-0 ${
                      playingId === sample.sample_id
                        ? "bg-status-success/20 border-status-success/40 text-status-success shadow-glow animate-pulse"
                        : "bg-white/[0.03] border-white/[0.05] text-white/60 hover:text-white hover:bg-white/[0.06] hover:border-white/10"
                    }`}
                    onClick={() => togglePlaySample(sample)}
                    disabled={disabled || busy || compareBusy}
                    title={playingId === sample.sample_id ? "Stop audio" : "Listen to clip"}
                  >
                    {playingId === sample.sample_id ? (
                      <Square size={14} className="fill-current" />
                    ) : (
                      <Volume2 size={14} />
                    )}
                  </button>

                  <button
                    type="button"
                    className="btn-primary flex-1 py-2.5 text-xs font-bold flex items-center justify-center gap-1.5"
                    disabled={disabled || busy || compareBusy}
                    onClick={() => void analyzeDemo(demo)}
                  >
                    <Play size={11} className="fill-current" />
                    {busy ? "Running…" : "Analyze"}
                  </button>

                  {demo.mode !== "auto" ? (
                    <button
                      type="button"
                      className="p-2.5 rounded-xl border border-white/[0.05] bg-white/[0.03] text-white/60 hover:text-white hover:bg-white/[0.06] hover:border-white/10 transition flex items-center justify-center shrink-0"
                      onClick={() => void compareDemo(demo)}
                      disabled={disabled || busy || compareBusy}
                      title="Compare across all models"
                    >
                      <BarChart size={14} />
                    </button>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
