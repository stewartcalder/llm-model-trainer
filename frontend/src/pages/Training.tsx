import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Project, RunPodStatus, Stats, TrainingConfig, TrainingJob } from "../types";
import { Badge, useToast } from "../ui";

const DEFAULT_CFG: TrainingConfig = {
  base_model: "meta-llama/Llama-3.2-1B",
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
};

const ACTIVE = ["queued", "running", "IN_QUEUE", "IN_PROGRESS"];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card" style={{ marginBottom: 14 }}>
      <h3>{title}</h3>
      {children}
    </div>
  );
}

export default function Training({ project, stats }: { project: Project; stats: Stats | null }) {
  const [rpStatus, setRpStatus] = useState<RunPodStatus | null>(null);
  const [cfg, setCfg] = useState<TrainingConfig>(DEFAULT_CFG);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null);
  const [starting, setStarting] = useState(false);
  const timer = useRef<number | null>(null);
  const toast = useToast();

  const loadStatus = useCallback(() =>
    api.runpodStatus(project.id).then(setRpStatus).catch(() => {}), [project.id]);

  const loadJobs = useCallback(() =>
    api.listTrainingJobs(project.id).then(j => { setJobs(j); return j; }), [project.id]);

  useEffect(() => {
    loadStatus();
    loadJobs().then(j => { if (j.length > 0) setActiveJob(j[0]); });
  }, [project.id]);

  // Poll the active job while it's running.
  useEffect(() => {
    const job = activeJob;
    if (!job || !ACTIVE.includes(job.status)) {
      if (timer.current) window.clearInterval(timer.current);
      return;
    }
    timer.current = window.setInterval(async () => {
      const updated = await api.getTrainingJob(project.id, job.id);
      setActiveJob(updated);
      setJobs(prev => prev.map(j => j.id === updated.id ? updated : j));
    }, 5000);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [activeJob?.id, activeJob?.status]);

  const start = async () => {
    setStarting(true);
    try {
      const job = await api.startTraining(project.id, cfg);
      setActiveJob(job);
      setJobs(prev => [job, ...prev]);
      toast("Training job submitted to RunPod.");
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
    toast("Cancellation requested.");
  };

  const approvedCount = stats?.by_status["approved"] ?? 0;
  const isActive = activeJob && ACTIVE.includes(activeJob.status);
  const pct = activeJob && activeJob.config.num_epochs
    ? Math.min(
        100,
        // Rough estimate from log line count as a proxy.
        Math.round((activeJob.log.split("\n").length / (activeJob.config.num_epochs * 20)) * 100)
      )
    : 0;

  return (
    <div>
      <h1 className="page-title">GPU Fine-Tuning</h1>
      <p className="page-sub">
        Submit a LoRA fine-tuning job to your RunPod serverless endpoint and download the trained adapter.
      </p>

      {/* RunPod connectivity */}
      <Section title="RunPod connection">
        {!rpStatus ? (
          <div className="muted">Checking…</div>
        ) : rpStatus.configured ? (
          <div>
            <div className="row" style={{ marginBottom: 8 }}>
              <span className="badge green">✓ endpoint configured</span>
              <span className="mono muted">{rpStatus.endpoint_id}</span>
            </div>
            {rpStatus.health.ok !== false ? (
              <div className="row">
                <span className="badge green">endpoint reachable</span>
                {rpStatus.health.workers != null && (
                  <span className="muted" style={{ fontSize: 12 }}>
                    Workers: {JSON.stringify(rpStatus.health.workers)}
                  </span>
                )}
              </div>
            ) : (
              <div>
                <span className="badge red">endpoint unreachable</span>
                <span className="muted" style={{ fontSize: 12, marginLeft: 8 }}>
                  {String((rpStatus.health as any).detail || "")}
                </span>
              </div>
            )}
          </div>
        ) : (
          <div>
            <span className="badge red">not configured</span>
            <p className="muted" style={{ marginTop: 10, fontSize: 13 }}>
              Add <code>RUNPOD_API_KEY</code> and <code>RUNPOD_ENDPOINT_ID</code> to{" "}
              <code>backend/.env</code>, then restart the server.
              See <strong>runpod_worker/README.md</strong> to deploy the training handler.
            </p>
          </div>
        )}
        <button className="btn small ghost" style={{ marginTop: 10 }} onClick={loadStatus}>
          Refresh
        </button>
      </Section>

      <div className="grid cols-2" style={{ alignItems: "start" }}>
        {/* Training config */}
        <div>
          <Section title="Base model">
            <label className="field">
              <span>Hugging Face model ID</span>
              <input type="text" value={cfg.base_model}
                onChange={e => setCfg(c => ({ ...c, base_model: e.target.value }))}
                placeholder="meta-llama/Llama-3.2-1B" />
            </label>
            <label className="field">
              <span>Dataset format</span>
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
                          : c.include_statuses.filter(x => x !== s)
                      }))} />
                    {s.replace(/_/g, " ")}
                  </label>
                ))}
              </div>
              <div className="muted" style={{ marginTop: 6, fontSize: 12 }}>
                {approvedCount} approved sample{approvedCount !== 1 ? "s" : ""} available
              </div>
            </label>
          </Section>

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
              4-bit quantisation (QLoRA) — recommended for smaller GPUs
            </label>
          </Section>

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

          <div className="row">
            <button
              className="btn"
              onClick={start}
              disabled={starting || !!isActive || !rpStatus?.configured || approvedCount === 0}
            >
              {starting ? "Submitting…" : isActive ? "Training in progress…" : "⚡ Start training"}
            </button>
            {approvedCount === 0 && (
              <span className="muted" style={{ fontSize: 12 }}>
                No approved samples — approve some in Review first.
              </span>
            )}
          </div>
        </div>

        {/* Active job / history */}
        <div>
          {activeJob && (
            <Section title={`Job ${activeJob.runpod_job_id?.slice(0, 10) ?? activeJob.id.slice(0, 8)}`}>
              <div className="row" style={{ marginBottom: 12 }}>
                <Badge status={activeJob.status} />
                <span className="muted" style={{ fontSize: 12 }}>
                  {new Date(activeJob.created_at).toLocaleString()}
                </span>
                {isActive && (
                  <button className="btn small danger" onClick={() => cancel(activeJob)}>Cancel</button>
                )}
                {activeJob.status === "completed" && activeJob.model_path && (
                  <a
                    className="btn small"
                    href={api.downloadModelUrl(project.id, activeJob.id)}
                    download
                  >
                    ⤓ Download adapter
                  </a>
                )}
              </div>

              {isActive && (
                <div style={{ marginBottom: 12 }}>
                  <div className="progress"><div style={{ width: `${pct}%` }} /></div>
                  <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
                    <span className="spin">⟳</span> Polling every 5s…
                  </div>
                </div>
              )}

              <div className="log" style={{ maxHeight: 360 }}>
                {activeJob.log || "Waiting for first status update…"}
              </div>

              {activeJob.config.base_model && (
                <div className="muted" style={{ fontSize: 11, marginTop: 8 }}>
                  {activeJob.config.base_model} · LoRA r={activeJob.config.lora_r} · {activeJob.config.num_epochs} epoch(s)
                </div>
              )}
            </Section>
          )}

          {jobs.length > 1 && (
            <Section title="Job history">
              <table>
                <thead>
                  <tr>
                    <th>Started</th>
                    <th>Status</th>
                    <th>Model</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.slice(1).map(j => (
                    <tr key={j.id} onClick={() => setActiveJob(j)} style={{ cursor: "pointer" }}>
                      <td>{new Date(j.created_at).toLocaleString()}</td>
                      <td><Badge status={j.status} /></td>
                      <td className="mono" style={{ fontSize: 11 }}>
                        {(j.config.base_model || "").split("/").pop()}
                      </td>
                      <td>
                        {j.status === "completed" && j.model_path && (
                          <a className="btn small ghost" href={api.downloadModelUrl(project.id, j.id)} download>
                            ⤓
                          </a>
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
                No training jobs yet. Configure the parameters on the left and click <strong>Start training</strong>.
                The adapter weights (LoRA / QLoRA) will be downloaded automatically when the job completes.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
