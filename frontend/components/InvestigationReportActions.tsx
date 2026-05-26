"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import {
  FinalReportSnapshot,
  ShareableReport,
  createShareableReport
} from "@/lib/api";
import { Button, Card, SectionHeader, formatDate } from "@/components/ui";

type ReportTemplate = "executive_summary" | "product_decision_memo";

const templateLabels: Record<ReportTemplate, string> = {
  executive_summary: "Executive Summary",
  product_decision_memo: "Product Decision Memo"
};

const templateDescriptions: Record<ReportTemplate, string> = {
  executive_summary: "Short, decision-ready summary of the main findings and charts.",
  product_decision_memo: "Structured memo for a product or business decision, with recommendation and rationale."
};

export function InvestigationReportActions({
  investigationId,
  initialReports,
  initialSnapshotsByReport
}: {
  investigationId: string;
  initialReports: ShareableReport[];
  initialSnapshotsByReport: Record<string, FinalReportSnapshot[]>;
}) {
  const router = useRouter();
  const [reports, setReports] = useState<ShareableReport[]>(initialReports);
  const [snapshotsByReport, setSnapshotsByReport] = useState(initialSnapshotsByReport);
  const [template, setTemplate] = useState<ReportTemplate>("executive_summary");
  const [includeTechnical, setIncludeTechnical] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const latestReport = reports.find((report) => report.is_latest !== false) || reports[0];
  const latestSnapshots = latestReport ? snapshotsByReport[latestReport.report_id] || [] : [];
  const latestSnapshot = latestSnapshots[0];

  async function createReport() {
    setIsCreating(true);
    setError(null);
    try {
      const report = await createShareableReport(investigationId, { template, include_technical: includeTechnical });
      setReports((current) => [report, ...current.filter((item) => item.report_id !== report.report_id)]);
      setSnapshotsByReport((current) => ({ ...current, [report.report_id]: [] }));
      router.refresh();
      router.push(`/reports/${report.report_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create report");
    } finally {
      setIsCreating(false);
    }
  }

  return (
    <Card>
      <SectionHeader
        title="Report actions"
        description="Turn reviewed findings and charts into a polished report."
      />

      {!latestReport ? (
        <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50/70 p-4 dark:border-slate-700 dark:bg-slate-900/40">
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto]">
            <select
              value={template}
              onChange={(event) => setTemplate(event.target.value as ReportTemplate)}
              title={templateDescriptions[template]}
              className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-500/10 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
            >
              {Object.entries(templateLabels).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
            <Button
              type="button"
              onClick={createReport}
              disabled={isCreating}
              size="sm"
            >
              {isCreating ? "Creating..." : "Create shareable report"}
            </Button>
          </div>
          <p className="mt-2 text-xs leading-5 text-slate-500 dark:text-slate-400">
            {templateDescriptions[template]}
          </p>
          <label className="mt-3 flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400">
            <input
              type="checkbox"
              checked={includeTechnical}
              onChange={(event) => setIncludeTechnical(event.target.checked)}
            />
            <span title="Adds technical artifacts and implementation detail for reviewers who need to audit the analysis.">
              Include technical details
            </span>
          </label>
        </div>
      ) : (
        <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50/70 p-4 dark:border-slate-800 dark:bg-slate-900/40">
          <div className="space-y-3">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <div className="font-semibold text-slate-950 dark:text-slate-50">{latestReport.title}</div>
                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600 dark:bg-slate-900 dark:text-slate-300">
                  {reportTemplateLabel(latestReport.template)}
                </span>
              </div>
              <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                Version {latestReport.version} · Updated {formatDate(latestReport.updated_at)}
                {latestSnapshot ? ` · Final version saved ${formatDate(latestSnapshot.created_at)}` : ""}
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <Link className="inline-flex rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:bg-slate-900" href={`/reports/${latestReport.report_id}`}>
                View report
              </Link>
              <Button
                type="button"
                onClick={createReport}
                disabled={isCreating}
                variant="secondary"
                size="sm"
              >
                {isCreating ? "Creating..." : "Create updated report"}
              </Button>
            </div>
          </div>
        </div>
      )}

      {reports.length > 1 ? (
        <div className="mt-4 border-t border-slate-100 pt-4 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
          Earlier report versions are kept for history and hidden from the main report list.
        </div>
      ) : null}

      {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
    </Card>
  );
}

function reportTemplateLabel(value: string) {
  if (value === "product_decision_memo") return "Decision memo";
  return "Executive summary";
}
