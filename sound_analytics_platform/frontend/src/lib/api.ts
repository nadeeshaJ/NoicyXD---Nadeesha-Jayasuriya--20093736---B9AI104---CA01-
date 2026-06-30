import { getSessionId } from "./session";
import type { ModelBenchmarkRow } from "./supabase";

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type ProcessingMode = "urban" | "animal" | "auto";

export type AssessmentInfo = {
  confidence: number;
  entropy_normalized: number;
  uncertainty_level: string;
  reliability_level: string;
  reliability_message: string;
  is_unknown: boolean;
  display_label: string;
  display_name: string;
  best_guess_label: string;
  calibration_note: string;
};

export type RouterExpertMetrics = {
  top_label: string;
  top_confidence: number;
  entropy_normalized: number;
  uncertainty_level: string;
  strength_score: number;
};

export type PredictResult = {
  processing_mode: string;
  effective_mode: string;
  model_key: string;
  top_label: string;
  top_confidence: number;
  predictions: Array<{ label: string; confidence: number }>;
  probabilities: Record<string, number>;
  inference_ms?: number;
  device_used?: string;
  assessment: AssessmentInfo;
  waveform_png: string;
  mel_png: string;
  rgb_png: string;
  gradcam_png?: string;
  gradcam_summary?: Record<string, unknown>;
  router?: {
    domain: string;
    reason: string;
    primary_reason?: string;
    hint_note?: string | null;
    urban_score: number;
    animal_score: number;
    confidence_gap: number;
    selected_uncertainty: string;
    urban_metrics: RouterExpertMetrics;
    animal_metrics: RouterExpertMetrics;
    urban_probe: { top_label: string; top_confidence: number };
    animal_probe: { top_label: string; top_confidence: number };
  };
  benchmark?: Record<string, unknown>;
  saved_prediction_id?: string | null;
  ground_truth_label?: string | null;
  sample_id?: string | null;
  input_source?: string | null;
  dataset_domain?: "urban" | "animal" | null;
};

export type DatasetOverview = {
  domain: string;
  dataset_key: string;
  title: string;
  total_clips: number | null;
  num_classes: number;
  test_clips: number;
  source?: string;
  classes: string[];
};

export type DatasetSample = {
  sample_id: string;
  filename: string;
  label: string;
  audio_path?: string;
  image_path?: string;
  domain: string;
  note?: string;
  curated?: boolean;
};

export type ValidationCheck = {
  name: string;
  target: string;
  actual: string;
  passed: boolean;
};

export type AudioPreview = {
  valid: boolean;
  filename?: string | null;
  input_source: string;
  original_duration_sec: number;
  processed_duration_sec: number;
  sample_rate: number;
  channels: string;
  waveform_png: string;
  mel_png: string;
  validation_checks: ValidationCheck[];
};

export type ModelCompareItem = {
  model_key: string;
  display_name: string;
  top_label: string;
  top_confidence: number;
  inference_ms?: number | null;
  checkpoint_size_mb?: number | null;
  benchmark_latency_ms?: number | null;
};

export type ModelCompareResult = {
  effective_mode: string;
  comparisons: ModelCompareItem[];
  sample_id?: string | null;
  dataset_domain?: "urban" | "animal" | null;
  input_source?: string | null;
};

export type PredictionHistoryRow = {
  id: string;
  session_id: string;
  processing_mode: string;
  routed_domain: string | null;
  model_key: string;
  input_source: string;
  original_filename: string | null;
  top_label: string;
  top_confidence: number;
  inference_ms: number | null;
  reliability_level: string | null;
  is_unknown: boolean | null;
  display_label: string | null;
  gradcam_enabled: boolean;
  created_at: string;
  ground_truth_label?: string | null;
  sample_id?: string | null;
  dataset_domain?: "urban" | "animal" | null;
  has_ground_truth?: boolean;
  audit_match?: boolean | null;
};

export type AnalyticsDashboard = {
  total_predictions: number;
  avg_latency_ms: number | null;
  predictions_last_hour: number;
  low_confidence_count: number;
  unknown_count: number;
  latency_trend: Array<{ timestamp: string; latency_ms: number; label?: string }>;
  class_distribution: Array<{ name: string; count: number }>;
  mode_distribution: Array<{ name: string; count: number }>;
  source_distribution: Array<{ name: string; count: number }>;
  model_distribution: Array<{ name: string; count: number }>;
  urban_event_summary: Array<{ name: string; count: number }>;
  animal_event_summary: Array<{ name: string; count: number }>;
};

export type PendingAudio = {
  blob: Blob;
  source: "upload" | "microphone";
  filename?: string;
};

export async function predictAudio(params: {
  file: File | Blob;
  filename?: string;
  mode: ProcessingMode;
  modelName: string;
  inputSource: "upload" | "microphone";
  gradcam?: boolean;
}): Promise<PredictResult> {
  const form = new FormData();
  form.append("file", params.file, params.filename ?? "recording.wav");
  form.append("mode", params.mode);
  form.append("model_name", params.modelName);
  form.append("input_source", params.inputSource);
  form.append("gradcam", String(params.gradcam ?? true));
  form.append("save_to_db", "true");

  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: {
      "X-Session-Id": getSessionId(),
    },
    body: form,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Prediction request failed.");
  }

  return response.json();
}

export async function fetchBenchmarksFromApi(): Promise<ModelBenchmarkRow[]> {
  const response = await fetch(`${API_BASE}/api/models`);
  if (!response.ok) throw new Error("Failed to load model benchmarks.");
  return response.json();
}

export async function fetchHistoryFromApi(limit = 50): Promise<PredictionHistoryRow[]> {
  const response = await fetch(`${API_BASE}/api/predictions?limit=${limit}`, {
    headers: {
      "X-Session-Id": getSessionId(),
    },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Failed to load prediction history.");
  }
  return response.json();
}

export async function fetchDatasetOverview(): Promise<DatasetOverview[]> {
  const response = await fetch(`${API_BASE}/api/datasets`);
  if (!response.ok) throw new Error("Failed to load dataset overview.");
  return response.json();
}

export async function fetchCuratedSamples(domain: "urban" | "animal"): Promise<DatasetSample[]> {
  const response = await fetch(`${API_BASE}/api/datasets/${domain}/curated`);
  if (!response.ok) throw new Error("Failed to load curated samples.");
  return response.json();
}

export async function fetchDatasetSamples(domain: "urban" | "animal", label?: string): Promise<DatasetSample[]> {
  const params = new URLSearchParams({ limit: "30" });
  if (label) params.set("label", label);
  const response = await fetch(`${API_BASE}/api/datasets/${domain}/samples?${params}`);
  if (!response.ok) throw new Error("Failed to load dataset samples.");
  return response.json();
}

export async function predictFromSample(params: {
  domain: "urban" | "animal";
  sampleId: string;
  mode: ProcessingMode;
  modelName: string;
  gradcam?: boolean;
}): Promise<PredictResult> {
  const form = new FormData();
  form.append("domain", params.domain);
  form.append("sample_id", params.sampleId);
  form.append("mode", params.mode);
  form.append("model_name", params.modelName);
  form.append("gradcam", String(params.gradcam ?? true));
  form.append("save_to_db", "true");

  const response = await fetch(`${API_BASE}/api/predict/sample`, {
    method: "POST",
    headers: { "X-Session-Id": getSessionId() },
    body: form,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Dataset sample prediction failed.");
  }
  return response.json();
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) throw new Error("API unavailable.");
  return response.json();
}

export async function previewAudio(params: {
  file: Blob;
  filename?: string;
  inputSource: "upload" | "microphone";
}): Promise<AudioPreview> {
  const form = new FormData();
  form.append("file", params.file, params.filename ?? "recording.wav");
  form.append("input_source", params.inputSource);
  const response = await fetch(`${API_BASE}/api/audio/preview`, { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Audio preview failed.");
  }
  return response.json();
}

export async function compareModels(params: {
  file: Blob;
  filename?: string;
  mode: ProcessingMode;
}): Promise<ModelCompareResult> {
  const form = new FormData();
  form.append("file", params.file, params.filename ?? "recording.wav");
  form.append("mode", params.mode);
  const response = await fetch(`${API_BASE}/api/predict/compare`, { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Model comparison failed.");
  }
  return response.json();
}

export async function compareSampleModels(params: {
  domain: "urban" | "animal";
  sampleId: string;
  mode: ProcessingMode;
}): Promise<ModelCompareResult> {
  const form = new FormData();
  form.append("domain", params.domain);
  form.append("sample_id", params.sampleId);
  form.append("mode", params.mode);
  const response = await fetch(`${API_BASE}/api/predict/sample/compare`, { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Sample model comparison failed.");
  }
  return response.json();
}

export async function fetchAnalyticsDashboard(): Promise<AnalyticsDashboard> {
  const response = await fetch(`${API_BASE}/api/analytics/dashboard`, {
    headers: { "X-Session-Id": getSessionId() },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Failed to load analytics dashboard.");
  }
  return response.json();
}

export async function exportPredictionReport(result: PredictResult): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/reports/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(result),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Report export failed.");
  }
  return response.blob();
}

export async function exportSessionReport(limit = 100): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/reports/session-export?limit=${limit}`, {
    headers: { "X-Session-Id": getSessionId() },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Session export failed.");
  }
  return response.blob();
}

export function pngDataUrl(base64: string): string {
  return `data:image/png;base64,${base64}`;
}
