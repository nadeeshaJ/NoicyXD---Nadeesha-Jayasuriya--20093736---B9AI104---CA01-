import { Square, Volume2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { API_BASE, type PendingAudio } from "../lib/api";

export type ReportAudioSource = {
  pendingAudio?: PendingAudio | null;
  datasetDomain?: "urban" | "animal" | null;
  sampleId?: string | null;
};

export function canPlayReportAudio(source: ReportAudioSource | null | undefined): boolean {
  if (!source) return false;
  if (source.pendingAudio?.blob) return true;
  return Boolean(source.datasetDomain && source.sampleId);
}

export function PlaySoundButton({
  source,
  className = "",
  size = "sm",
}: {
  source: ReportAudioSource | null | undefined;
  className?: string;
  size?: "sm" | "md";
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackError, setPlaybackError] = useState<string | null>(null);

  const canPlay = canPlayReportAudio(source);

  useEffect(() => {
    return () => {
      audioRef.current?.pause();
      audioRef.current = null;
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, []);

  function releaseObjectUrl() {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }

  function stopPlayback() {
    audioRef.current?.pause();
    audioRef.current = null;
    setIsPlaying(false);
    releaseObjectUrl();
  }

  async function togglePlay() {
    if (!source || !canPlay) return;
    setPlaybackError(null);

    if (isPlaying) {
      stopPlayback();
      return;
    }

    let src: string;
    if (source.pendingAudio?.blob) {
      releaseObjectUrl();
      src = URL.createObjectURL(source.pendingAudio.blob);
      objectUrlRef.current = src;
    } else if (source.datasetDomain && source.sampleId) {
      src = `${API_BASE}/api/datasets/${source.datasetDomain}/samples/${encodeURIComponent(source.sampleId)}/audio`;
    } else {
      return;
    }

    const audio = new Audio(src);
    audioRef.current = audio;
    audio.onended = () => stopPlayback();
    audio.onerror = () => {
      setPlaybackError("Could not play this audio clip.");
      stopPlayback();
    };

    try {
      await audio.play();
      setIsPlaying(true);
    } catch {
      setPlaybackError("Playback blocked or unsupported format.");
      stopPlayback();
    }
  }

  if (!canPlay) return null;

  const sizeClass =
    size === "md"
      ? "px-4 py-2 text-xs"
      : "px-3 py-1 text-[11px]";

  return (
    <div className={className}>
      <button
        type="button"
        className={`rounded-xl border font-bold transition flex items-center gap-1.5 ${sizeClass} ${
          isPlaying
            ? "bg-status-success/15 border-status-success/30 text-status-success shadow-glow animate-pulse"
            : "bg-white/[0.03] border-white/[0.05] text-white/70 hover:bg-white/[0.08]"
        }`}
        onClick={() => void togglePlay()}
      >
        {isPlaying ? <Square size={size === "md" ? 12 : 10} className="fill-current" /> : <Volume2 size={size === "md" ? 14 : 10} />}
        {isPlaying ? "Stop Sound" : "Play Sound"}
      </button>
      {playbackError ? (
        <p className="mt-1 text-[11px] text-status-warning font-medium">{playbackError}</p>
      ) : null}
    </div>
  );
}
