const TARGET_SAMPLE_RATE = 22050;

function writeString(view: DataView, offset: number, text: string) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function encodeWav(audioBuffer: AudioBuffer): Blob {
  const channel = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const dataSize = channel.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < channel.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, channel[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/** Decode browser MediaRecorder output (WebM/Opus) and export mono WAV at 22.05 kHz. */
export async function recordingToWavBlob(input: Blob): Promise<Blob> {
  const arrayBuffer = await input.arrayBuffer();
  const decodeContext = new AudioContext();
  try {
    const decoded = await decodeContext.decodeAudioData(arrayBuffer.slice(0));
    const offline = new OfflineAudioContext(
      1,
      Math.max(1, Math.ceil(decoded.duration * TARGET_SAMPLE_RATE)),
      TARGET_SAMPLE_RATE,
    );
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start(0);
    const rendered = await offline.startRendering();
    return encodeWav(rendered);
  } finally {
    await decodeContext.close();
  }
}

export function pickRecorderMimeType(): string | undefined {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}
