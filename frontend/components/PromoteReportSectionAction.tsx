"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { promoteReportSectionToMemory } from "@/lib/api";

export function PromoteReportSectionAction({
  reportId,
  sectionId
}: {
  reportId: string;
  sectionId: string;
}) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function promote() {
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      await promoteReportSectionToMemory(reportId, sectionId);
      setMessage("Promoted to decision memory.");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not promote section");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-3 flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={promote}
        disabled={busy}
        className="rounded-md border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:text-slate-400"
      >
        {busy ? "Saving..." : "Promote section to decision"}
      </button>
      {message ? <span className="text-xs text-emerald-600">{message}</span> : null}
      {error ? <span className="text-xs text-red-600">{error}</span> : null}
    </div>
  );
}
