"use client";

import { useState } from "react";

export function CollapsibleSection({
  title,
  count,
  defaultOpen = true,
  children
}: {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="rounded-2xl border border-slate-200/80 bg-white/90 shadow-sm dark:border-slate-800 dark:bg-slate-950/85">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <span className="text-sm font-semibold text-slate-950 dark:text-slate-50">{title}</span>
        <span className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
          {typeof count === "number" ? <span>{count}</span> : null}
        </span>
      </button>
      {open ? <div className="border-t border-slate-100 p-4 dark:border-slate-800">{children}</div> : null}
    </section>
  );
}
