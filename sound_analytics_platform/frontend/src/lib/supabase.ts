import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabaseConfigured = Boolean(supabaseUrl && supabaseAnonKey);

export const supabase = supabaseConfigured
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null;

export type PredictionRow = {
  id: string;
  session_id: string;
  processing_mode: string;
  routed_domain: string | null;
  model_key: string;
  input_source: string;
  top_label: string;
  top_confidence: number;
  inference_ms: number | null;
  router_reason: string | null;
  gradcam_enabled: boolean;
  created_at: string;
};

export type ModelBenchmarkRow = {
  model_key: string;
  display_name: string;
  total_parameters: number;
  model_file_size_mb: number;
  inference_ms_mean: number;
  test_accuracy: number | null;
  test_macro_recall: number | null;
  test_macro_f1: number | null;
  is_deployed: boolean;
  notes: string | null;
};

export async function fetchHistoryFromSupabase(sessionId: string): Promise<PredictionRow[]> {
  if (!supabase) return [];
  const { data, error } = await supabase
    .from("predictions")
    .select("*")
    .eq("session_id", sessionId)
    .order("created_at", { ascending: false })
    .limit(25);
  if (error) throw error;
  return data ?? [];
}

export async function fetchBenchmarksFromSupabase(): Promise<ModelBenchmarkRow[]> {
  if (!supabase) return [];
  const { data, error } = await supabase
    .from("model_benchmarks")
    .select("*")
    .order("test_accuracy", { ascending: false });
  if (error) throw error;
  return data ?? [];
}
