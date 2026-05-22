import type { ReactNode } from "react";

export function TechnicalDetails({
  title = "Advanced details",
  children
}: {
  title?: string;
  children: ReactNode;
}) {
  return (
    <details className="mt-3 rounded-xl border border-slate-200 bg-slate-50/60 p-3 dark:border-slate-800 dark:bg-slate-900/40">
      <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-slate-500 transition hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200">
        {title}
      </summary>
      <div className="mt-3 text-xs leading-5 text-slate-500 dark:text-slate-400">
        {children}
      </div>
    </details>
  );
}
