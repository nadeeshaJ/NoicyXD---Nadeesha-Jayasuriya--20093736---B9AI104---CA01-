import { Cpu, HardDrive, Smartphone, TreePine } from "lucide-react";
import type { ModelBenchmarkRow } from "../lib/supabase";

const ANIMAL_EXPERT = {
  model_key: "mobilenetv2_animal",
  display_name: "MobileNetV2 (Animal Expert)",
  test_accuracy: 0.6,
  test_macro_recall: 0.592,
  test_macro_f1: 0.607,
  inference_ms_mean: 4.2,
  model_file_size_mb: 8.76,
  total_parameters: 1538890,
  is_deployed: true,
  domain: "animal" as const,
  notes: "ESC-50 animals · mobilenetv2_imagenet_only checkpoint",
};

const DEPLOYMENT_PROFILES = [
  {
    icon: Smartphone,
    title: "Mobile / edge node",
    model: "MobileNetV2",
    latency: "~4 ms",
    size: "~9 MB",
    use: "Patrol apps, wildlife nodes, low-power devices",
  },
  {
    icon: Cpu,
    title: "GPU server",
    model: "ResNet50",
    latency: "~4–5 ms",
    size: "~90 MB",
    use: "Higher accuracy when compute budget allows",
  },
  {
    icon: HardDrive,
    title: "Baseline reference",
    model: "Custom CNN",
    latency: "~1 ms",
    size: "~25 MB",
    use: "From-scratch baseline; lower accuracy",
  },
];

function BenchmarkCard({ row }: { row: ModelBenchmarkRow | typeof ANIMAL_EXPERT }) {
  const isAnimal = "domain" in row && row.domain === "animal";
  const border = row.is_deployed
    ? isAnimal
      ? "border-cyan-glow/40 bg-cyan-glow/[0.03] shadow-glow"
      : "border-accent/40 bg-accent/[0.03] shadow-glow"
    : "border-white/[0.05]";

  return (
    <div className={`glass-panel p-6 relative overflow-hidden border ${border}`}>
      <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-white/[0.03] to-transparent pointer-events-none" />
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="text-lg font-bold text-white tracking-tight">{row.display_name}</div>
          {isAnimal ? (
            <div className="text-[10px] text-cyan-glow/70 uppercase tracking-wider mt-1">ESC-50 Animals</div>
          ) : (
            <div className="text-[10px] text-white/40 uppercase tracking-wider mt-1">UrbanSound8K fold-10</div>
          )}
        </div>
        {row.is_deployed ? (
          <span
            className={`rounded-full px-3 py-1 text-[10px] font-bold uppercase tracking-wider ${
              isAnimal ? "bg-cyan-glow/20 border border-cyan-glow/30 text-cyan-glow" : "bg-accent/25 border border-accent/30 text-accent-glow"
            }`}
          >
            Deployed
          </span>
        ) : null}
      </div>
      <div className="space-y-3.5 text-sm">
        <div className="flex justify-between border-b border-white/[0.05] pb-2">
          <span className="text-white/40">Test accuracy</span>
          <span className="font-semibold text-white">{row.test_accuracy ? `${(row.test_accuracy * 100).toFixed(1)}%` : "—"}</span>
        </div>
        <div className="flex justify-between border-b border-white/[0.05] pb-2">
          <span className="text-white/40">Macro recall</span>
          <span className="font-semibold text-white">{row.test_macro_recall ? row.test_macro_recall.toFixed(3) : "—"}</span>
        </div>
        <div className="flex justify-between border-b border-white/[0.05] pb-2">
          <span className="text-white/40">Macro F1</span>
          <span className="font-semibold text-white">{row.test_macro_f1 ? row.test_macro_f1.toFixed(3) : "—"}</span>
        </div>
        <div className="flex justify-between border-b border-white/[0.05] pb-2">
          <span className="text-white/40">Mean latency</span>
          <span className="font-semibold text-white font-mono">{row.inference_ms_mean} ms</span>
        </div>
        <div className="flex justify-between">
          <span className="text-white/40">Checkpoint size</span>
          <span className="font-semibold text-white font-mono">{row.model_file_size_mb} MB</span>
        </div>
      </div>
      {"notes" in row && row.notes ? (
        <p className="mt-4 text-[11px] text-white/40 leading-relaxed">{row.notes}</p>
      ) : null}
    </div>
  );
}

type Props = {
  benchmarks: ModelBenchmarkRow[];
};

export function ModelsPanel({ benchmarks }: Props) {
  const urban = benchmarks.filter((b) => b.model_key !== "mobilenetv2_animal");

  return (
    <div className="space-y-8">
      <section>
        <div className="mb-4 flex items-center gap-2">
          <TreePine size={16} className="text-accent-soft" />
          <h2 className="text-lg font-bold text-white">Urban models (UrbanSound8K)</h2>
        </div>
        <div className="grid gap-6 lg:grid-cols-3">
          {urban.map((row) => (
            <BenchmarkCard key={row.model_key} row={row} />
          ))}
        </div>
      </section>

      <section>
        <div className="mb-4 flex items-center gap-2">
          <TreePine size={16} className="text-cyan-glow" />
          <h2 className="text-lg font-bold text-white">Animal expert (ESC-50)</h2>
        </div>
        <div className="grid gap-6 lg:grid-cols-3">
          <BenchmarkCard row={ANIMAL_EXPERT} />
        </div>
      </section>

      <section className="glass-panel p-6">
        <h2 className="text-lg font-bold text-white mb-1">Deployment profiles</h2>
        <p className="text-xs text-white/45 mb-5">How each architecture maps to a runtime scenario (from benchmark runs).</p>
        <div className="grid gap-4 md:grid-cols-3">
          {DEPLOYMENT_PROFILES.map((profile) => (
            <div key={profile.title} className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
              <profile.icon size={18} className="text-white/50 mb-2" />
              <div className="font-semibold text-white text-sm">{profile.title}</div>
              <div className="mt-2 space-y-1 text-xs text-white/55">
                <div>Model: <span className="text-white/80">{profile.model}</span></div>
                <div>Latency: <span className="text-white/80 font-mono">{profile.latency}</span></div>
                <div>Size: <span className="text-white/80 font-mono">{profile.size}</span></div>
              </div>
              <p className="mt-3 text-[11px] text-white/40 leading-relaxed">{profile.use}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
