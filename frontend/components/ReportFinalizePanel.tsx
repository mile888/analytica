"use client";

import { useState } from "react";
import { FinalReportSnapshot, ReportReadinessResult, approveReport, finalizeReport } from "@/lib/api";
import { formatDate } from "@/components/ui";

export function ReportFinalizePanel({
  reportId,
  initialApprovalStatus,
  initialSnapshots,
  readiness
}: {
  reportId: string;
  initialApprovalStatus: string;
  initialSnapshots: FinalReportSnapshot[];
  readiness?: ReportReadinessResult | null;
}) {
  const [snapshots, setSnapshots] = useState(initialSnapshots);
  const [approvalStatus, setApprovalStatus] = useState(initialApprovalStatus);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const [isApproving, setIsApproving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const latestSnapshot = snapshots[0];
  const blocked = Boolean(readiness && !readiness.is_ready);
  const approved = approvalStatus === "approved";

  async function approve() {
    setIsApproving(true);
    setError(null);
    try {
      const report = await approveReport(reportId);
      setApprovalStatus(report.approval_status);
    } catch (err) {
      setError(friendlyReportError(err, "Could not approve report"));
    } finally {
      setIsApproving(false);
    }
  }

  async function finalize() {
    setIsFinalizing(true);
    setError(null);
    try {
      const snapshot = await finalizeReport(reportId, false);
      setSnapshots((current) => [snapshot, ...current.filter((item) => item.snapshot_id !== snapshot.snapshot_id)]);
    } catch (err) {
      setError(friendlyReportError(err, "Could not create final version"));
    } finally {
      setIsFinalizing(false);
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="text-sm font-semibold text-slate-950">Finalize</h2>
      {latestSnapshot ? (
        <div className="mt-3 rounded-md border border-emerald-200 bg-emerald-50 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full bg-emerald-100 px-2.5 py-1 text-[11px] font-semibold text-emerald-800">
              Saved
            </span>
            <span className="text-xs text-emerald-700">{formatDate(latestSnapshot.created_at)}</span>
          </div>
          <p className="mt-2 text-xs text-emerald-800">Final version is saved for this report.</p>
        </div>
      ) : (
        <p className="mt-3 text-sm text-slate-500">No final version has been created yet.</p>
      )}

      {readiness && !readiness.is_ready ? (
        <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-900">
          This report has {readiness.blocking_count} blocking readiness issue(s). Resolve open comments and blocking
          checks before creating a final version.
        </div>
      ) : null}

      {!approved ? (
        <button
          type="button"
          onClick={approve}
          disabled={isApproving || blocked}
          className="mt-3 w-full rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {isApproving ? "Approving..." : blocked ? "Resolve checks before approving" : "Approve report"}
        </button>
      ) : (
        <button
          type="button"
          onClick={finalize}
          disabled={isFinalizing || blocked}
          className="mt-3 w-full rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {isFinalizing ? "Finalizing..." : blocked ? "Resolve checks before finalizing" : "Create final version"}
        </button>
      )}
      {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
    </div>
  );
}

function friendlyReportError(err: unknown, fallback: string) {
  const message = err instanceof Error ? err.message : fallback;
  if (message.includes("report is not approved")) {
    return "Approve the report before creating the final version.";
  }
  if (message.includes("Cannot create final snapshot")) {
    return message.replace("Cannot create final snapshot", "Cannot create final version");
  }
  return message;
}
