import { createContext, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { api } from "./api";
import type { LLMStatus } from "./types";

// --- Toast ---
type Toast = { msg: string; err?: boolean } | null;
const ToastCtx = createContext<(msg: string, err?: boolean) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<Toast>(null);
  const show = (msg: string, err = false) => {
    setToast({ msg, err });
    setTimeout(() => setToast(null), 4000);
  };
  return (
    <ToastCtx.Provider value={show}>
      {children}
      {toast && <div className={`toast ${toast.err ? "err" : ""}`}>{toast.msg}</div>}
    </ToastCtx.Provider>
  );
}
export const useToast = () => useContext(ToastCtx);

// --- LLM status poller ---
export function useLLMStatus(projectId: string | null, intervalMs = 30_000) {
  const [status, setStatus] = useState<LLMStatus | null>(null);
  const timer = useRef<number | null>(null);

  const poll = () => {
    if (!projectId) return;
    api.llmStatus(projectId).then(setStatus).catch(() => {});
  };

  useEffect(() => {
    if (!projectId) { setStatus(null); return; }
    poll();
    timer.current = window.setInterval(poll, intervalMs);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [projectId]);

  return status;
}

// --- LLM status chip (sidebar) ---
export function LLMStatusChip({ status }: { status: LLMStatus | null }) {
  if (!status) return <div className="muted" style={{ fontSize: 11, padding: "0 8px" }}>Checking model…</div>;

  const dot = status.ok ? "🟢" : "🔴";
  const label = status.ok ? "connected" : "disconnected";

  return (
    <div style={{ padding: "8px", borderRadius: 8, background: "var(--panel-2)", border: "1px solid var(--border)" }}>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2 }}>
        {dot} {status.model || status.provider}
      </div>
      <div style={{ fontSize: 11, color: "var(--muted)" }}>
        {status.provider} · {label}
        {status.ok && status.latency_ms > 0 && ` · ${status.latency_ms}ms`}
      </div>
      {status.detail && !status.ok && (
        <div style={{ fontSize: 10, color: "var(--red)", marginTop: 2, wordBreak: "break-word" }}>
          {status.detail.slice(0, 80)}
        </div>
      )}
    </div>
  );
}

// --- Status badge ---
const STATUS_CLASS: Record<string, string> = {
  done: "green",
  approved: "green",
  processing: "blue",
  running: "blue",
  edited: "blue",
  pending: "amber",
  pending_review: "amber",
  queued: "amber",
  error: "red",
  rejected: "red",
  cancelled: "red",
};

export function Badge({ status }: { status: string }) {
  const cls = STATUS_CLASS[status] || "";
  return <span className={`badge ${cls}`}>{status.replace(/_/g, " ")}</span>;
}

export function fmtCost(n: number): string {
  return "$" + n.toFixed(n < 0.01 ? 5 : 4);
}

export function fmtNum(n: number): string {
  return n.toLocaleString();
}
