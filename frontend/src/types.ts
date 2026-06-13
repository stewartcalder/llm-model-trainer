export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
  config: PipelineConfig;
  source_count: number;
  sample_count: number;
}

export interface PipelineConfig {
  chunking: { strategy: string; window_tokens: number; overlap_tokens: number };
  sample_types: string[];
  llm: {
    provider: string;
    model: string;
    base_url?: string;
    temperature: number;
    max_tokens: number;
    use_critic: boolean;
    api_key?: string;
  };
  concurrency: number;
  budget_usd: number;
  export: { format: string; train_split: number; include_statuses: string[] };
  prompt_templates?: Record<string, string>;
}

export interface Source {
  id: string;
  type: string;
  path_or_url: string;
  title: string;
  status: string;
  error?: string | null;
  chunk_count: number;
  sample_count: number;
  created_at: string;
}

export interface Chunk {
  id: string;
  source_id: string;
  text: string;
  token_count: number;
  page_or_section?: string | null;
  sample_count: number;
}

export interface Sample {
  id: string;
  chunk_id: string;
  source_id: string;
  source_title: string;
  type: string;
  instruction: string;
  input: string;
  output: string;
  quality: Record<string, number | boolean | string>;
  status: string;
  chunk_text: string;
  created_at: string;
  edited_at?: string | null;
}

export interface Run {
  id: string;
  status: string;
  stage: string;
  started_at: string;
  finished_at?: string | null;
  chunks_processed: number;
  chunks_total: number;
  samples_generated: number;
  tokens_used: number;
  cost_usd: number;
  log: string;
}

export interface DryRun {
  pending_sources: number;
  estimated_chunks: number;
  estimated_calls: number;
  estimated_tokens: number;
  estimated_cost_usd: number;
}

export interface Stats {
  sources: number;
  chunks: number;
  samples: number;
  by_type: Record<string, number>;
  by_status: Record<string, number>;
  approved_pct: number;
  avg_quality: Record<string, number>;
  token_histogram: { bucket: string; count: number }[];
}

export interface ExportResult {
  manifest: Record<string, unknown>;
  train_file: string;
  val_file: string;
  manifest_file: string;
  train_count: number;
  val_count: number;
}

export interface Meta {
  sample_types: { id: string; label: string }[];
  providers: string[];
  export_formats: string[];
}

export interface LLMStatus {
  ok: boolean;
  provider: string;
  model: string;
  latency_ms: number;
  detail: string;
}

export interface TrainingConfig {
  provider: "local" | "runpod";
  base_model: string;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  num_epochs: number;
  batch_size: number;
  learning_rate: number;
  max_seq_length: number;
  use_4bit: boolean;
  dataset_format: string;
  include_statuses: string[];
  // Local-only
  gguf_quantization: string;
  ollama_model_name: string;
}

export interface OllamaModelInfo {
  ollama_name: string;
  hf_model: string | null;
  cached: boolean;
  size_gb: number;
  mapped: boolean;
}

export interface LocalStatus {
  available: boolean;
  version: string | null;
  gpu: boolean;
  detail: string;
}

export interface TrainingJob {
  id: string;
  project_id: string;
  runpod_job_id: string | null;
  status: string;
  config: Partial<TrainingConfig>;
  log: string;
  model_path: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface RunPodStatus {
  configured: boolean;
  endpoint_id: string;
  health: Record<string, unknown>;
}

export interface ScreenSize {
  physical_width: number;
  physical_height: number;
  logical_width: number;
  logical_height: number;
  mon_left: number;
  mon_top: number;
}

export interface ScreenScraperStatus {
  available: boolean;
  missing: string[];
  detail: string;
  screen: ScreenSize | null;
}

export interface ScrapeStartRequest {
  project_id: string;
  title: string;
  region_left: number;
  region_top: number;
  region_width: number;
  region_height: number;
  click_x: number;
  click_y: number;
  pause_seconds: number;
  max_pages: number;
  change_threshold: number;
}

export interface ScrapeJob {
  id: string;
  project_id: string;
  source_id: string | null;
  title: string;
  status: string;
  config: Partial<ScrapeStartRequest>;
  pages: number;
  text: string;
  error: string | null;
  created_at: string;
  finished_at: string | null;
}
