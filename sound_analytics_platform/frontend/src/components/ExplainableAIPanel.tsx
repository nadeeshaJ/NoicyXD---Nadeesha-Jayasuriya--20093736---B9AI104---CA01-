import { Eye } from "lucide-react";
import { pngDataUrl, type AssessmentInfo, type PredictResult } from "../lib/api";
import { ConfidenceCalibrationPanel } from "./ConfidenceCalibrationPanel";
import { PlaySoundButton, type ReportAudioSource } from "./PlaySoundButton";
import { RouterExplanationPanel } from "./RouterExplanationPanel";

function formatLabel(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatModelName(modelKey: string) {
  if (modelKey === "mobilenetv2") return "MobileNetV2";
  if (modelKey === "custom_cnn") return "Custom CNN";
  if (modelKey === "resnet50") return "ResNet50";
  return formatLabel(modelKey);
}

function buildNarrativeSummary(result: PredictResult, assessment: AssessmentInfo): string {
  const label = formatLabel(result.top_label);
  const conf = (assessment.confidence * 100).toFixed(1);
  const model = formatModelName(result.model_key);
  const parts: string[] = [
    `${model} predicted ${label} at ${conf}% confidence (${assessment.reliability_level} reliability).`,
  ];

  if (result.gradcam_png) {
    parts.push(
      "Grad-CAM highlights time–frequency regions on the Mel-spectrogram that drove the decision — warmer colours mean stronger CNN attention.",
    );
  } else {
    parts.push(
      "Grad-CAM was off for this run; review the Mel-spectrogram and model input RGB to inspect what the network saw.",
    );
  }

  if (result.router) {
    const route = result.router.domain.toUpperCase();
    parts.push(
      `Smart Auto-Router sent this clip to the ${route} expert because ${(result.router.primary_reason ?? result.router.reason).toLowerCase()}`,
    );
  }

  if (assessment.is_unknown) {
    parts.push("Confidence is below the unknown threshold — verify by listening before trusting this label.");
  }

  return parts.join(" ");
}

type Props = {
  result: PredictResult;
  assessment: AssessmentInfo;
  audioSource: ReportAudioSource;
};

export function ExplainableAIPanel({ result, assessment, audioSource }: Props) {
  const gradcamMethod =
    typeof result.gradcam_summary?.method === "string"
      ? result.gradcam_summary.method
      : "Grad-CAM on final convolutional layer";

  return (
    <section className="glass-panel p-6 relative overflow-hidden border border-accent/15">
      <div className="absolute top-0 right-0 w-72 h-72 bg-cyanGradient pointer-events-none z-0" />

      <div className="relative z-10 flex flex-wrap items-start justify-between gap-4 mb-5">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-accent/10 p-2.5 text-accent-soft border border-accent/20 shadow-glow">
            <Eye size={20} />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white tracking-tight">Explainable AI</h3>
            <p className="text-xs text-white/50 mt-0.5">
              Why the model chose this label — attention maps, confidence, and routing transparency
            </p>
          </div>
        </div>
        <PlaySoundButton source={audioSource} size="md" />
      </div>

      <p className="relative z-10 text-sm text-white/75 leading-relaxed font-medium bg-white/[0.02] border border-white/[0.05] rounded-xl p-4 mb-6">
        {buildNarrativeSummary(result, assessment)}
      </p>

      <div className="relative z-10 space-y-6">
        <ConfidenceCalibrationPanel result={result} assessment={assessment} embedded />

        <div>
          <h4 className="mb-4 text-xs font-bold uppercase tracking-wider text-white/60">
            Signal &amp; attention visualizations
          </h4>
          <div className="grid gap-4 md:grid-cols-3">
            {[
              { title: "Waveform", hint: "Raw amplitude over time", src: result.waveform_png },
              { title: "Mel-spectrogram", hint: "Frequency content fed to the CNN", src: result.mel_png },
              {
                title: result.gradcam_png ? "Grad-CAM overlay" : "Model input (RGB)",
                hint: result.gradcam_png ? gradcamMethod : "224×224 CNN input without attention overlay",
                src: result.gradcam_png ?? result.rgb_png,
              },
            ].map((panel) => (
              <div key={panel.title} className="rounded-2xl border border-white/[0.05] bg-white/[0.02] p-4">
                <h5 className="text-xs font-bold uppercase tracking-wider text-white/70">{panel.title}</h5>
                <p className="text-[10px] text-white/40 mt-0.5 mb-3">{panel.hint}</p>
                <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-brand-dark/30">
                  <img src={pngDataUrl(panel.src)} alt={panel.title} className="w-full" />
                </div>
              </div>
            ))}
          </div>
        </div>

        {result.router ? <RouterExplanationPanel router={result.router} embedded /> : null}

        <div className="rounded-2xl border border-white/[0.05] bg-white/[0.02] p-5">
          <h4 className="mb-4 text-xs font-bold uppercase tracking-wider text-white/60">
            Decision distribution (top candidates)
          </h4>
          <div className="space-y-4">
            {result.predictions.map((prediction) => (
              <div key={prediction.label}>
                <div className="mb-1.5 flex justify-between text-xs font-medium">
                  <span className="text-white/80">{formatLabel(prediction.label)}</span>
                  <span className="text-accent-glow font-bold">{(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${prediction.confidence * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
