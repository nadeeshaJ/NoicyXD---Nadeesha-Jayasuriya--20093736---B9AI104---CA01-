import { Mic, Square, Upload } from "lucide-react";
import { useRef, useState } from "react";

type Props = {
  onAudioSelected: (blob: Blob, source: "upload" | "microphone", filename?: string) => void;
  disabled?: boolean;
};

export function AudioInputPanel({ onAudioSelected, disabled }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      window.setTimeout(() => {
        if (recorder.state === "recording") stopRecording();
      }, 4000);
    } catch {
      setError("Microphone access denied or unavailable.");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="glass-panel p-5">
        <div className="mb-3 flex items-center gap-2 text-sm font-medium text-white/80">
          <Upload size={16} className="text-accent" />
          Upload WAV
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".wav,audio/wav"
          disabled={disabled}
          className="input-shell"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onAudioSelected(file, "upload", file.name);
          }}
        />
        <p className="mt-3 text-xs text-white/45">Upload triggers validation preview before inference.</p>
      </div>

      <div className="glass-panel p-5">
        <div className="mb-3 flex items-center gap-2 text-sm font-medium text-white/80">
          <Mic size={16} className="text-accent" />
          Live Microphone
        </div>
        {!recording ? (
          <button className="btn-primary w-full" disabled={disabled} onClick={startRecording}>
            <Mic size={16} />
            Record 4 seconds
          </button>
        ) : (
          <button className="btn-secondary w-full border-red-400/30 text-red-200" onClick={stopRecording}>
            <Square size={16} />
            Stop recording
          </button>
        )}
        {error ? <p className="mt-3 text-xs text-red-300">{error}</p> : null}
      </div>
    </div>
  );
}
