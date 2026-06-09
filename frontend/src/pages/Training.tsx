import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { LocalStatus, Project, RunPodStatus, Stats, TrainingConfig, TrainingJob } from "../types";
import { Badge, useToast } from "../ui";

// ── Constants ─────────────────────────────────────────────────────────────────

const UNSLOTH_MODELS = [
  { id: "unsloth/Llama-3.2-1B-Instruct",   label: "Llama 3.2 1B — fastest, minimal VRAM" },
  { id: "unsloth/Llama-3.2-3B-Instruct",   label: "Llama 3.2 3B" },
  { id: "unsloth/Llama-3.1-8B-Instruct",   label: "Llama 3.1 8B — popular balance" },
  { id: "unsloth/Phi-3.5-mini-instruct",   label: "Phi-3.5 mini 3.8B — Microsoft" },
  { id: "unsloth/Qwen2.5-7B-Instruct",     label: "Qwen 2.5 7B — multilingual" },
  { id: "unsloth/mistral-7b-instruct-v0.3",label: "Mistral 7B" },
  { id: "unsloth/gemma-2-9b-it",           label: "Gemma 2 9B — Google" },
  { id: "__custom__",                       label: "Custom HuggingFace model ID…" },
];

const GGUF_QUANTS = [
  { id: "q4_k_m", label: "Q4_K_M — recommended (~4 GB)" },
  { id: "q5_k_m", label: "Q5_K_M — better quality (~5 GB)" },
  { id: "q8_0",   label: "Q8_0 — near-lossless (~8 GB)" },
  { id: "f16",    label: "F16 — no quantisation (very large)" },
];

const DEFAULT_CFG: TrainingConfig = {
  provider: "local",
  base_model: "unsloth/Llama-3.2-1B-Instruct",
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  num_epochs: 3,
  batch_size: 4,
  learning_rate: 2e-4,
  max_seq_length: 2048,
  use_4bit: true,
  dataset_format: "alpaca",
  include_statuses: ["approved"],
  gguf_quantization: "q4_k_m",
  ollama_model_name: "",
};

const ACTIVE_STATUSES = ["queued", "running", "IN_QUEUE", "IN_PROGRESS"];

// ── Small helpers ─────────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card" style={{ marginBottom: 14 }}>
      <h3>{title}</h3>
      {children}
    </div>
  );
}

function ProviderToggle({ value, onChange }: { value: "local" | "runpod"; onChange: (v: "local" | "runpod") => void }) {
  const btn = (id: "local" | "runpod", icon: string, label: string) => (
    <button
      className={`btn ${value === id ? "" : "ghost"}`}
      style={{ flex: 1 }}
      onClick={() => onChange(id)}
    >
      {icon} {label}
    </button>
  );
  return (
    <div className="row" style={{ marginBottom: 18 }}>
      {btn("local", "🖥", "Local (Unsloth)")}
      {btn("runpod", "☁", "RunPod (Cloud)")}
    </div>
  );
}

function LocalStatusPanel({ status }: { status: LocalStatus | null }) {
  if (!status) return <div className="muted">Checking…</div>;
  return (
    <div>
      <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
        <span className={`badge ${status.available ? "green" : "red"}`}>
          {status.available ? "✓ unsloth installed" : "✗ unsloth not installed"}
        </span>
        {status.available && (
          <span className={`badge ${status.gpu ? "green" : "amber"}`}>
            {status.gpu ? "GPU available" : "CPU only — very slow"}
          </span>
        )}
      </div>
      <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>{status.detail}</div>
      {!status.available && (
        <pre className="log" style={{ marginTop: 10, fontSize: 11 }}>
          pip install unsloth trl&gt;=0.12 datasets accelerate bitsandbytes peft safetensors
        </pre>
      )}
    </div>
  );
}

function RunPodStatusPanel({ status }: { status: RunPodStatus | null }) {
  if (!status) return <div className="muted">Checking…</div>;
  if (!status.configured) {
    return (
      <div>
        <span className="badge red">not configured</span>
        <p className="muted" style={{ marginTop: 10, fontSize: 13 }}>
          Add <code>RUNPOD_API_KEY</code> and <code>RUNPOD_ENDPOINT_ID</code> to{" "}
          <code>backend/.env</code>, then restart the server.
          See <strong>runpod_worker/README.md</strong> for setup instructions.
        </p>
      </div>
    );
  }
  const healthy = status.health.ok !== false;
  return (
    <div>
      <div className="row" style={{ flexWrap: "wrap", gap: 8, marginBottom: 6 }}>
        <span className="badge green">✓ endpoint configured</span>
        <span className={`badge ${healthy ? "green" : "red"}`}>
          {healthy ? "reachable" : "unreachable"}
        </span>
        <span className="mono muted" style={{ fontSize: 11 }}>{status.endpoint_id}</span>
      </div>
      {status.health.workers != null && (
        <div className="muted" style={{ fontSize: 12 }}>
          Workers: {JSON.stringify(status.health.workers)}
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Training({ project, stats }: { project: Project; stats: Stats | null }) {
  const [cfg, setCfg] = useState<TrainingConfig>(DEFAULT_CFG);
  const [customModel, setCustomModel] = useState("");
  const [localStatus, setLocalStatus] = useState<LocalStatus | null>(null);
  const [rpStatus, setRpStatus] = useState<RunPodStatus | null>(null);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null);
  const [starting, setStarting] = useState(false);
  const pollRef = useRef<number | null>(null);
  const toast = useToast();

  const provider = cfg.provider;

  // Derive the actual base_model for the config.
  const selectedPreset = UNSLOTH_MODELS.find(m => m.id === cfg.base_model) ?? { id: "__custom__", label: "" };
  const isCustom = selectedPreset.id === "__custom__" || !UNSLOTH_MODELS.slice(0, -1).find(m => m.id === cfg.base_model);

  const loadLocal = useCallback(() =>
    api.localStatus(project.id).then(setLocalStatus).catch(() => {}), [project.id]);
  const loadRunpod = useCallback(() =>
    api.runpodStatus(project.id).then(setRpStatus).catch(() => {}), [project.id]);
  const loadJobs = useCallback(() =>
    api.listTrainingJobs(project.id).then(j => { setJobs(j); return j; }), [project.id]);

  useEffect(() => {
    loadLocal();
    loadRunpod();
    loadJobs().then(j => { if (j.length) setActiveJob(j[0]); });
  }, [project.id]);

  // Poll the active job while running.
  useEffect(() => {
    const job = activeJob;
    if (!job || !ACTIVE_STATUSES.includes(job.status)) {
      if (pollRef.current) window.clearInterval(pollRef.current);
      return;
    }
    pollRef.current = window.setInterval(async () => {
      const updated = await api.getTrainingJob(project.id, job.id);
      setActiveJob(updated);
      setJobs(prev => prev.map(j => j.id === updated.id ? updated : j));
    }, 5000);
    return () => { if (pollRef.current) window.clearInterval(pollRef.current); };
  }, [activeJob?.id, activeJob?.status]);

  const handlePresetChange = (presetId: string) => {
    if (presetId === "__custom__") {
      setCfg(c => ({ ...c, base_model: customModel || "" }));
    } else {
      setCfg(c => ({ ...c, base_model: presetId }));
    }
  };

  const handleProviderSwitch = (p: "local" | "runpod") => {
    setCfg(c => ({
      ...c,
      provider: p,
      base_model: p === "local"
        ? "unsloth/Llama-3.2-1B-Instruct"
        : "meta-llama/Llama-3.2-1B",
    }));
  };

  const start = async () => {
    const effectiveCfg: TrainingConfig = {
      ...cfg,
      base_model: isCustom ? customModel : cfg.base_model,
    };
    if (!effectiveCfg.base_model.trim()) {
      toast("Enter a base model ID.", true); return;
    }
    setStarting(true);
    try {
      const job = await api.startTraining(project.id, effectiveCfg);
      setActiveJob(job);
      setJobs(prev => [job, ...prev]);
      toast(provider === "local"
        ? "Local training started — this will take a while."
        : "Training job submitted to RunPod.");
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setStarting(false);
    }
  };

  const cancel = async (job: TrainingJob) => {
    await api.cancelTraining(project.id, job.id);
    const updated = await api.getTrainingJob(project.id, job.id);
    setActiveJob(updated);
    setJobs(prev => prev.map(j => j.id === updated.id ? updated : j));
    toast(provider === "local" ? "Cancel signal sent — waiting for current step to finish." : "Cancellation requested.");
  };

  const approvedCount = stats?.by_status["approved"] ?? 0;
  const isActive = activeJob && ACTIVE_STATUSES.includes(activeJob.status);
  const canStart = !starting && !isActive && approvedCount > 0
    && (provider === "local" ? localStatus?.available : rpStatus?.configured);

  const pct = (activeJob && activeJob.config.num_epochs)
    ? Math.min(100, Math.round((activeJob.log.split("\n").length / (activeJob.config.num_epochs * 20)) * 100))
    : 0;

  return (
    <div>
      <h1 className="page-title">Fine-Tuning</h1>
      <p className="page-sub">
        Train a LoRA adapter on your approved samples and load the result into Ollama.
      </p>

      <ProviderToggle value={provider} onChange={handleProviderSwitch} />

      <div className="grid cols-2" style={{ alignItems: "start" }}>
        {/* ── Left: config ── */}
        <div>

          {/* Provider-specific connection panel */}
          <Section title={provider === "local" ? "Unsloth (local GPU)" : "RunPod connection"}>
            {provider === "local"
              ? <LocalStatusPanel status={localStatus} />
              : <RunPodStatusPanel status={rpStatus} />}
            <button className="btn small ghost" style={{ marginTop: 10 }}
              onClick={provider === "local" ? loadLocal : loadRunpod}>
              Refresh
            </button>
          </Section>

          {/* Base model */}
          <Section title="Base model">
            {provider === "local" ? (
              <>
                <label className="field">
                  <span>Model preset</span>
                  <select
                    value={isCustom ? "__custom__" : cfg.base_model}
                    onChange={e => handlePresetChange(e.target.value)}
                  >
                    {UNSLOTH_MODELS.map(m => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                  </select>
                </label>
                {isCustom && (
                  <label className="field">
                    <span>HuggingFace model ID</span>
                    <input type="text" value={customModel}
                      onChange={e => { setCustomModel(e.target.value); setCfg(c => ({ ...c, base_model: e.target.value })); }}
                      placeholder="username/model-name" />
                  </label>
                )}
                <div className="muted" style={{ fontSize: 11 }}>
                  Models are downloaded from HuggingFace on first use. Ensure you have accepted any licence agreements.
                </div>
              </>
            ) : (
              <label className="field">
                <span>HuggingFace model ID</span>
                <input type="text" value={cfg.base_model}
                  onChange={e => setCfg(c => ({ ...c, base_model: e.target.value }))}
                  placeholder="meta-llama/Llama-3.2-1B" />
              </label>
            )}
          </Section>

          {/* Local-only: GGUF export + Ollama */}
          {provider === "local" && (
            <Section title="Ollama export">
              <label className="field">
                <span>GGUF quantisation</span>
                <select value={cfg.gguf_quantization}
                  onChange={e => setCfg(c => ({ ...c, gguf_quantization: e.target.value }))}>
                  {GGUF_QUANTS.map(q => <option key={q.id} value={q.id}>{q.label}</option>)}
                </select>
              </label>
              <label className="field">
                <span>Ollama model name</span>
                <input type="text" value={cfg.ollama_model_name}
                  onChange={e => setCfg(c => ({ ...c, ollama_model_name: e.target.value }))}
                  placeholder="my-expert:latest" />
                <span className="muted" style={{ fontSize: 11, marginTop: 4, display: "block" }}>
                  After training, runs: <code>ollama create {cfg.ollama_model_name || "<name>"} -f Modelfile</code>.
                  Leave blank to skip Ollama registration and just download the GGUF.
                </span>
              </label>
            </Section>
          )}

          {/* Dataset */}
          <Section title="Dataset">
            <label className="field">
              <span>Format</span>
              <select value={cfg.dataset_format}
                onChange={e => setCfg(c => ({ ...c, dataset_format: e.target.value }))}>
                <option value="alpaca">Alpaca</option>
                <option value="sharegpt">ShareGPT</option>
                <option value="openai">OpenAI messages</option>
              </select>
            </label>
            <label className="field">
              <span>Include sample statuses</span>
              <div className="pill-row">
                {["approved", "edited", "pending_review"].map(s => (
                  <label key={s} className="checkbox">
                    <input type="checkbox"
                      checked={cfg.include_statuses.includes(s)}
                      onChange={e => setCfg(c => ({
                        ...c,
                        include_statuses: e.target.checked
                          ? [...c.include_statuses, s]
                          : c.include_statuses.filter(x => x !== s),
                      }))} />
                    {s.replace(/_/g, " ")}
                  </label>
                ))}
              </div>
              <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                {approvedCount} approved sample{approvedCount !== 1 ? "s" : ""} available
              </div>
            </label>
          </Section>

          {/* LoRA */}
          <Section title="LoRA parameters">
            <div className="grid cols-2">
              <label className="field">
                <span>Rank (r)</span>
                <input type="number" value={cfg.lora_r}
                  onChange={e => setCfg(c => ({ ...c, lora_r: +e.target.value }))} />
              </label>
              <label className="field">
                <span>Alpha</span>
                <input type="number" value={cfg.lora_alpha}
                  onChange={e => setCfg(c => ({ ...c, lora_alpha: +e.target.value }))} />
              </label>
              <label className="field">
                <span>Dropout</span>
                <input type="number" step="0.01" value={cfg.lora_dropout}
                  onChange={e => setCfg(c => ({ ...c, lora_dropout: +e.target.value }))} />
              </label>
              <label className="field">
                <span>Max seq length</span>
                <input type="number" value={cfg.max_seq_length}
                  onChange={e => setCfg(c => ({ ...c, max_seq_length: +e.target.value }))} />
              </label>
            </div>
            <label className="checkbox" style={{ marginTop: 4 }}>
              <input type="checkbox" checked={cfg.use_4bit}
                onChange={e => setCfg(c => ({ ...c, use_4bit: e.target.checked }))} />
              4-bit quantisation (QLoRA) — recommended for GPUs with &lt;24 GB VRAM
            </label>
          </Section>

          {/* Training hypers */}
          <Section title="Training">
            <div className="grid cols-3">
              <label className="field">
                <span>Epochs</span>
                <input type="number" value={cfg.num_epochs}
                  onChange={e => setCfg(c => ({ ...c, num_epochs: +e.target.value }))} />
              </label>
              <label className="field">
                <span>Batch size</span>
                <input type="number" value={cfg.batch_size}
                  onChange={e => setCfg(c => ({ ...c, batch_size: +e.target.value }))} />
              </label>
              <label className="field">
                <span>Learning rate</span>
                <input type="number" step="1e-5" value={cfg.learning_rate}
                  onChange={e => setCfg(c => ({ ...c, learning_rate: +e.target.value }))} />
              </label>
            </div>
          </Section>

          <div className="row" style={{ flexWrap: "wrap", gap: 10 }}>
            <button className="btn" onClick={start} disabled={!canStart}>
              {starting
                ? "Submitting…"
                : isActive
                  ? `${provider === "local" ? "🖥" : "☁"} Training in progress…`
                  : `${provider === "local" ? "🖥" : "☁"} Start training`}
            </button>
            {approvedCount === 0 && (
              <span className="muted" style={{ fontSize: 12 }}>
                No approved samples — approve some in Review first.
              </span>
            )}
            {provider === "local" && localStatus && !localStatus.available && (
              <span className="muted" style={{ fontSize: 12 }}>
                Install Unsloth to use local training.
              </span>
            )}
          </div>
        </div>

        {/* ── Right: active job + history ── */}
        <div>
          {activeJob && (
            <Section title={
              `${load_json_provider(activeJob) === "local" ? "🖥 Local" : "☁ RunPod"} job — ` +
              (activeJob.runpod_job_id?.slice(0, 10) ?? activeJob.id.slice(0, 8))
            }>
              <div className="row" style={{ marginBottom: 10, flexWrap: "wrap", gap: 8 }}>
                <Badge status={activeJob.status} />
                <span className="muted" style={{ fontSize: 12 }}>
                  {new Date(activeJob.created_at).toLocaleString()}
                </span>
                {isActive && (
                  <button className="btn small danger" onClick={() => cancel(activeJob)}>
                    Cancel
                  </button>
                )}
                {activeJob.status === "completed" && activeJob.model_path && (
                  <a className="btn small"
                    href={api.downloadModelUrl(project.id, activeJob.id)} download>
                    ⤓ Download zip
                  </a>
                )}
              </div>

              {isActive && (
                <div style={{ marginBottom: 10 }}>
                  <div className="progress"><div style={{ width: `${pct}%` }} /></div>
                  <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
                    <span className="spin">⟳</span> Polling every 5 s…
                    {load_json_provider(activeJob) === "local" &&
                      " Training runs in the server process — do not restart the backend."}
                  </div>
                </div>
              )}

              {activeJob.status === "completed" && activeJob.config.ollama_model_name && (
                <div className="badge green" style={{ marginBottom: 10, display: "inline-block" }}>
                  ✓ Available in Ollama as &ldquo;{activeJob.config.ollama_model_name}&rdquo;
                </div>
              )}

              <div className="log" style={{ maxHeight: 380 }}>
                {activeJob.log || "Waiting for first update…"}
              </div>

              {activeJob.config.base_model && (
                <div className="muted" style={{ fontSize: 11, marginTop: 8 }}>
                  {activeJob.config.base_model}
                  {" · "}LoRA r={activeJob.config.lora_r}
                  {" · "}{activeJob.config.num_epochs} epoch(s)
                  {activeJob.config.gguf_quantization
                    ? ` · GGUF ${activeJob.config.gguf_quantization}`
                    : ""}
                </div>
              )}
            </Section>
          )}

          {jobs.length > 1 && (
            <Section title="Job history">
              <table>
                <thead>
                  <tr><th>Started</th><th>Provider</th><th>Status</th><th>Model</th><th></th></tr>
                </thead>
                <tbody>
                  {jobs.slice(1).map(j => (
                    <tr key={j.id} onClick={() => setActiveJob(j)} style={{ cursor: "pointer" }}>
                      <td>{new Date(j.created_at).toLocaleString()}</td>
                      <td>{load_json_provider(j) === "local" ? "🖥 local" : "☁ RunPod"}</td>
                      <td><Badge status={j.status} /></td>
                      <td className="mono" style={{ fontSize: 11 }}>
                        {(j.config.base_model || "").split("/").pop()}
                      </td>
                      <td>
                        {j.status === "completed" && j.model_path && (
                          <a className="btn small ghost"
                            href={api.downloadModelUrl(project.id, j.id)} download>⤓</a>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
          )}

          {jobs.length === 0 && !starting && (
            <div className="card" style={{ background: "transparent", border: "1px dashed var(--border)" }}>
              <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                No training jobs yet. Configure the options on the left and click{" "}
                <strong>Start training</strong>.
                {" "}
                {provider === "local"
                  ? "After training, the merged GGUF will be registered in your local Ollama server."
                  : "The LoRA adapter will be downloadable as a zip when the RunPod job completes."}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function load_json_provider(job: TrainingJob): string {
  return (job.config as { provider?: string }).provider ?? "runpod";
}
