import { CheckCircle2, XCircle, ArrowRight, Volume2, Square } from "lucide-react";
import { pngDataUrl, type AudioPreview } from "../lib/api";
import { useState, useEffect } from "react";

type Props = {
  preview: AudioPreview;
  pendingAudio: { blob: Blob; source: string; filename?: string } | null;
  onRunAnalysis: () => void;
  onCompareModels: () => void;
  disabled?: boolean;
};

export function AudioPreviewPanel({ preview, pendingAudio, onRunAnalysis, onCompareModels, disabled }: Props) {
  const [activeAudio, setActiveAudio] = useState<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    // Cleanup playback on unmount or file change
    return () => {
      activeAudio?.pause();
    };
  }, [activeAudio, pendingAudio]);

  function togglePlay() {
    if (activeAudio) {
      activeAudio.pause();
      setActiveAudio(null);
      setIsPlaying(false);
      return;
    }

    if (pendingAudio && pendingAudio.blob) {
      const url = URL.createObjectURL(pendingAudio.blob);
      const audio = new Audio(url);
      audio.play().catch((err) => console.error("Playback failed", err));
      setActiveAudio(audio);
      setIsPlaying(true);
      
      audio.onended = () => {
        setActiveAudio(null);
        setIsPlaying(false);
      };
    }
  }

  return (
    <section className="glass-panel p-6 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-64 h-64 bg-glowGradient pointer-events-none z-0" />
      
      <div className="relative z-10 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between mb-6">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-bold text-white tracking-tight">Pre-Inference Validation Checks</h2>
            {pendingAudio && (
              <button
                className={`px-3 py-1 rounded-xl border text-[11px] font-bold transition flex items-center gap-1.5 ${
                  isPlaying 
                    ? "bg-status-success/15 border-status-success/30 text-status-success shadow-glow animate-pulse" 
                    : "bg-white/[0.03] border-white/[0.05] text-white/70 hover:bg-white/[0.08]"
                }`}
                onClick={togglePlay}
              >
                {isPlaying ? <Square size={10} className="fill-current" /> : <Volume2 size={10} />}
                {isPlaying ? "Pause Sound" : "Play Sound"}
              </button>
            )}
          </div>
          <p className="mt-1 text-xs text-white/50 leading-relaxed">
            Inspect the signal characteristics and checks before dispatching the tensor to CNN pipelines.
          </p>
        </div>
        <div className="flex flex-row flex-nowrap items-center gap-3 shrink-0">
          <button
            className="btn-primary whitespace-nowrap"
            disabled={disabled || !preview.valid}
            onClick={onRunAnalysis}
          >
            Run Classifier
            <ArrowRight size={15} />
          </button>
          <button
            className="btn-secondary whitespace-nowrap"
            disabled={disabled || !preview.valid}
            onClick={onCompareModels}
          >
            Compare All Models
          </button>
        </div>
      </div>

      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4 relative z-10">
        {preview.validation_checks.map((check) => (
          <div
            key={check.name}
            className={`rounded-2xl border p-4 transition duration-300 ${
              check.passed 
                ? "border-status-success/20 bg-status-success/[0.02] hover:bg-status-success/[0.04]" 
                : "border-status-warning/20 bg-status-warning/[0.02] hover:bg-status-warning/[0.04]"
            }`}
          >
            <div className="flex items-center gap-2.5 text-sm font-bold text-white mb-2">
              {check.passed ? (
                <CheckCircle2 size={16} className="text-status-success shrink-0" />
              ) : (
                <XCircle size={16} className="text-status-warning shrink-0" />
              )}
              {check.name}
            </div>
            <div className="text-[11px] text-white/40 mb-1">Target: {check.target}</div>
            <div className="text-xs font-semibold text-white/70">Actual: {check.actual}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-6 md:grid-cols-2 relative z-10">
        <div className="rounded-2xl border border-white/[0.05] bg-white/[0.01] p-4 hover:border-white/[0.08] transition">
          <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-white/60">Waveform Amplitude (Time Domain)</h3>
          <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-brand-dark/30">
            <img src={pngDataUrl(preview.waveform_png)} alt="Waveform preview" className="w-full hover:scale-105 transition duration-500" />
          </div>
        </div>
        
        <div className="rounded-2xl border border-white/[0.05] bg-white/[0.01] p-4 hover:border-white/[0.08] transition">
          <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-white/60">Mel-Spectrogram (Frequency Domain)</h3>
          <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-brand-dark/30">
            <img src={pngDataUrl(preview.mel_png)} alt="Mel preview" className="w-full hover:scale-105 transition duration-500" />
          </div>
        </div>
      </div>
    </section>
  );
}
