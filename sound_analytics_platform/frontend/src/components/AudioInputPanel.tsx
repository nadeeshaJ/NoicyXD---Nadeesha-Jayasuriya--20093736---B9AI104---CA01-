import { Mic, Square, Upload, FileAudio } from "lucide-react";
import { useRef, useState, useEffect } from "react";

type Props = {
  onAudioSelected: (blob: Blob, source: "upload" | "microphone", filename?: string) => void;
  disabled?: boolean;
};

export function AudioInputPanel({ onAudioSelected, disabled }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const [recording, setRecording] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  async function startRecording() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        onAudioSelected(blob, "microphone", "recording.wav");
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      setCountdown(4);

      let timeLeft = 4;
      timerRef.current = window.setInterval(() => {
        timeLeft -= 1;
        setCountdown(timeLeft);
        if (timeLeft <= 0) {
          if (timerRef.current) clearInterval(timerRef.current);
          stopRecording();
        }
      }, 1000);
    } catch {
      setError("Microphone access denied or unavailable.");
    }
  }

  function stopRecording() {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    mediaRecorderRef.current?.stop();
    setRecording(false);
    setCountdown(0);
  }

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div 
        className="glass-panel p-6 flex flex-col justify-between hover:border-accent/20 cursor-pointer group"
        onClick={() => !disabled && fileRef.current?.click()}
      >
        <div>
          <div className="mb-4 flex items-center gap-3">
            <div className="rounded-xl bg-accent/10 p-2.5 text-accent-soft border border-accent/10 group-hover:scale-110 transition-transform">
              <Upload size={18} />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white tracking-wide">Upload WAV</h3>
              <p className="text-xs text-white/40">Select a local mono WAV file</p>
            </div>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".wav,audio/wav"
            disabled={disabled}
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onAudioSelected(file, "upload", file.name);
            }}
          />
          <div className="border border-dashed border-white/10 rounded-xl p-6 text-center text-xs text-white/50 bg-white/[0.01] group-hover:bg-white/[0.02] group-hover:border-white/20 transition-all">
            <FileAudio size={24} className="mx-auto mb-2 text-white/25 group-hover:text-accent-soft transition-colors" />
            <span>Drag and drop or click to browse files</span>
          </div>
        </div>
        <p className="mt-4 text-[10px] text-white/30">Auto-validates sample rate, duration, and channel formatting.</p>
      </div>

      <div className="glass-panel p-6 flex flex-col justify-between">
        <div>
          <div className="mb-4 flex items-center gap-3">
            <div className={`rounded-xl p-2.5 border transition-all ${recording ? "bg-status-error/15 text-status-error border-status-error/30 animate-pulse" : "bg-accent/10 text-accent-soft border-accent/10"}`}>
              <Mic size={18} />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white tracking-wide">Live Microphone</h3>
              <p className="text-xs text-white/40">Capture 4 seconds of environmental noise</p>
            </div>
          </div>

          <div className="flex flex-col items-center justify-center p-6 border border-white/[0.03] rounded-xl bg-white/[0.01] min-h-[96px] relative overflow-hidden">
            {recording && (
              <div className="absolute inset-0 bg-status-error/[0.02] flex items-center justify-center">
                <div className="h-20 w-20 rounded-full bg-status-error/10 border border-status-error/20 absolute recording-pulse"></div>
              </div>
            )}
            
            {!recording ? (
              <button className="btn-primary w-full shadow-glow" disabled={disabled} onClick={startRecording}>
                <Mic size={16} />
                Record Live Clip
              </button>
            ) : (
              <div className="w-full flex flex-col items-center gap-3 relative z-10">
                <button className="btn-secondary w-full border-status-error/20 hover:border-status-error/40 text-status-error hover:bg-status-error/5" onClick={stopRecording}>
                  <Square size={14} className="fill-current" />
                  Stop Recording ({countdown}s)
                </button>
                <div className="w-full bg-white/[0.08] h-1 rounded-full overflow-hidden">
                  <div className="bg-status-error h-full transition-all duration-1000 ease-linear" style={{ width: `${(countdown / 4) * 100}%` }}></div>
                </div>
              </div>
            )}
          </div>
        </div>
        {error ? <p className="mt-4 text-xs text-status-error font-medium">{error}</p> : <p className="mt-4 text-[10px] text-white/30">Ensures correct gain-scaling and pads/trims to 4.0s.</p>}
      </div>
    </div>
  );
}
