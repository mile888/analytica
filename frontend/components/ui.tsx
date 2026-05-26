import type { ReactNode } from "react";
import { formatDateTime } from "@/lib/format";

function joinClasses(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function StatusBadge({ value }: { value: string }) {
  const tone =
    value === "failed" || value === "error" || value === "revoked"
      ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900/70 dark:bg-red-950/50 dark:text-red-300"
      : value === "completed" || value === "verified" || value === "approved" || value === "final"
        ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/70 dark:bg-emerald-950/50 dark:text-emerald-300"
        : value === "warning" || value === "needs_review" || value === "changes_requested"
          ? "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900/70 dark:bg-amber-950/50 dark:text-amber-300"
          : "border-slate-200 bg-slate-50 text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300";
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold leading-none ${tone}`}>
      {statusLabel(value)}
    </span>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <section
      className={joinClasses(
        "rounded-2xl border border-slate-200/80 bg-white/90 p-5 shadow-sm shadow-slate-200/60 backdrop-blur dark:border-slate-800 dark:bg-slate-950/85 dark:shadow-black/20",
        className
      )}
    >
      {children}
    </section>
  );
}

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={joinClasses(
        "rounded-xl border border-slate-200 bg-slate-50/70 p-4 dark:border-slate-800 dark:bg-slate-900/50",
        className
      )}
    >
      {children}
    </div>
  );
}

export function SectionHeader({
  eyebrow,
  title,
  description,
  action
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        {eyebrow ? <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">{eyebrow}</div> : null}
        <h2 className="text-base font-semibold tracking-tight text-slate-950 dark:text-slate-50">{title}</h2>
        {description ? <p className="mt-1 text-sm leading-6 text-slate-500 dark:text-slate-400">{description}</p> : null}
      </div>
      {action}
    </div>
  );
}

export function Button({
  children,
  variant = "primary",
  size = "md",
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
}) {
  const variants = {
    primary: "bg-slate-950 text-white hover:bg-slate-800 disabled:bg-slate-400 dark:bg-slate-100 dark:text-slate-950 dark:hover:bg-white",
    secondary: "border border-slate-300 bg-white text-slate-800 hover:bg-slate-50 disabled:text-slate-400 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:bg-slate-900",
    ghost: "text-slate-600 hover:bg-slate-100 disabled:text-slate-400 dark:text-slate-300 dark:hover:bg-slate-900",
    danger: "border border-red-200 bg-red-50 text-red-700 hover:bg-red-100 disabled:text-red-300 dark:border-red-900/70 dark:bg-red-950/40 dark:text-red-300"
  };
  const sizes = {
    sm: "px-2.5 py-1.5 text-xs",
    md: "px-3.5 py-2 text-sm"
  };
  return (
    <button
      {...props}
      className={joinClasses(
        "inline-flex items-center justify-center rounded-lg font-semibold transition disabled:cursor-not-allowed",
        sizes[size],
        variants[variant],
        className
      )}
    >
      {children}
    </button>
  );
}

export function Chip({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <span className={joinClasses("inline-flex items-center rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600 dark:bg-slate-900 dark:text-slate-300", className)}>
      {children}
    </span>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50/70 p-6 text-sm leading-6 text-slate-500 dark:border-slate-700 dark:bg-slate-900/40 dark:text-slate-400">
      {children}
    </div>
  );
}

export function PageSkeleton() {
  return (
    <div className="space-y-6">
      <div className="skeleton h-10 w-72" />
      <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <div className="skeleton h-72" />
        <div className="space-y-3">
          <div className="skeleton h-28" />
          <div className="skeleton h-28" />
          <div className="skeleton h-28" />
        </div>
      </div>
    </div>
  );
}

export function formatDate(value?: string | null) {
  return formatDateTime(value);
}

export function stageLabel(value?: string | null) {
  if (!value) return "Unknown";
  const labels: Record<string, string> = {
    preparing_data: "Preparing data",
    building_context: "Preparing context",
    running_analysis: "Analyzing dataset",
    validating_results: "Checking results",
    generating_report: "Preparing summary",
    completed: "Updated"
  };
  if (labels[value]) return labels[value];
  return value
    .split("_")
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(" ");
}

function statusLabel(value: string) {
  const labels: Record<string, string> = {
    draft: "draft",
    needs_review: "needs review",
    needs_more_analysis: "needs more analysis",
    running: "analyzing",
    queued: "queued",
    completed: "updated",
    failed: "needs attention",
    error: "needs attention",
    approved: "approved",
    final: "final",
    active: "active",
    stale: "stale",
    archived: "archived",
    not_started: "not started",
    ready: "ready",
    needs_attention: "needs attention",
    assistant: "assistant",
    message: "message",
    question: "question",
    analysis_failed: "needs attention",
    edited: "edited",
    changes_requested: "needs work",
    warning: "warning",
    verified: "verified",
    revoked: "revoked"
  };
  return labels[value] || value.replaceAll("_", " ");
}
