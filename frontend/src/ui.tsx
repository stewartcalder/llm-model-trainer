import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";

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
