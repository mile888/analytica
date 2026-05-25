"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import type { InvestigationBranch } from "@/lib/api";
import { activateInvestigationBranch } from "@/lib/api";

export function applyActivatedBranch(branches: InvestigationBranch[], branchId: string): InvestigationBranch[] {
  return branches.map((branch) => ({ ...branch, is_active: branch.branch_id === branchId }));
}

export async function activateBranchSelection({
  investigationId,
  branch,
  busy,
  activateApi = activateInvestigationBranch
}: {
  investigationId: string;
  branch: InvestigationBranch;
  busy: string | null;
  activateApi?: typeof activateInvestigationBranch;
}) {
  if (branch.is_active || busy) return null;
  return activateApi(investigationId, branch.branch_id);
}

export function BranchSwitcher({
  investigationId,
  branches
}: {
  investigationId: string;
  branches: InvestigationBranch[];
}) {
  const router = useRouter();
  const [busy, setBusy] = useState<string | null>(null);
  const [localBranches, setLocalBranches] = useState(branches);
  useEffect(() => setLocalBranches(branches), [branches]);
  const visible = localBranches.slice(0, 5);

  async function activate(branch: InvestigationBranch) {
    if (branch.is_active || busy) return;
    setBusy(branch.branch_id);
    setLocalBranches((current) => applyActivatedBranch(current, branch.branch_id));
    try {
      const response = await activateInvestigationBranch(investigationId, branch.branch_id);
      if (response.branches?.length) {
        setLocalBranches(response.branches);
      }
      router.refresh();
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white/90 p-3 shadow-sm dark:border-slate-800 dark:bg-slate-950/85">
      <div className="mb-2 flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-950 dark:text-slate-50">Analysis threads</h2>
        {localBranches.length ? <span className="text-[11px] font-medium text-slate-400">{localBranches.length}</span> : null}
      </div>
      {visible.length ? (
        <div className="space-y-1.5">
          {visible.map((branch) => (
            <button
              key={branch.branch_id}
              type="button"
              onClick={() => activate(branch)}
              disabled={busy !== null}
              aria-pressed={branch.is_active}
              className={`w-full cursor-pointer rounded-xl border px-3 py-2 text-left transition disabled:cursor-wait disabled:opacity-80 ${
                branch.is_active
                  ? "border-blue-300 bg-blue-50 text-blue-950 dark:border-blue-800 dark:bg-blue-950/40 dark:text-blue-100"
                  : "border-slate-200 bg-white hover:border-blue-200 hover:bg-blue-50/50 dark:border-slate-800 dark:bg-slate-950 dark:hover:border-blue-900 dark:hover:bg-blue-950/20"
              }`}
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5 shrink-0 rounded-full border border-slate-200 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500 dark:border-slate-700 dark:text-slate-300">
                  {branchTypeLabel(branch.branch_type)}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-semibold">{branch.title}</span>
                  <span className="mt-0.5 block truncate text-xs text-slate-500 dark:text-slate-400">
                    {branch.subtitle || branch.branch_type}
                  </span>
                  {branch.dataset_ids?.length ? (
                    <span className="mt-0.5 block truncate text-[11px] text-slate-400">
                      Dataset: {branch.dataset_ids.join(", ")}
                    </span>
                  ) : null}
                </span>
                <span className={`mt-1 h-2 w-2 rounded-full ${branch.is_active ? "bg-blue-500" : "bg-slate-300 dark:bg-slate-700"}`} />
              </div>
            </button>
          ))}
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-slate-200 p-3 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
          No analysis threads yet
        </div>
      )}
    </section>
  );
}

export function branchTypeLabel(type: string): string {
  const normalized = String(type || "").toLowerCase();
  if (normalized.includes("shipping")) return "Ship";
  if (normalized.includes("temporal")) return "Trend";
  if (normalized.includes("distribution")) return "Dist";
  if (normalized.includes("quality")) return "Quality";
  if (normalized.includes("hypothesis")) return "Hyp";
  if (normalized.includes("customer")) return "Cust";
  if (normalized.includes("business")) return "Biz";
  if (normalized.includes("joinability")) return "Join";
  if (normalized.includes("warehouse")) return "WH";
  if (normalized.includes("multi_dataset")) return "Multi";
  if (normalized.includes("grouped")) return "Group";
  return "View";
}
