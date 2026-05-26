import Link from "next/link";
import { notFound } from "next/navigation";
import {
  ReportReadinessResult,
  getReport,
  getReportReadiness,
  listFinalSnapshotsForReport,
  listReportComments,
  reportPdfDownloadUrl
} from "@/lib/api";
import { ReportFinalizePanel } from "@/components/ReportFinalizePanel";
import { ReportReviewSummary } from "@/components/ReportReviewSummary";
import { ReportViewModeToggle } from "@/components/ReportViewModeToggle";
import { Card, StatusBadge, formatDate } from "@/components/ui";

export const dynamic = "force-dynamic";
export const revalidate = 0;
export const fetchCache = "force-no-store";

export default async function ReportDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let report;
  try {
    report = await getReport(id);
  } catch {
    notFound();
  }
  const [snapshots, readiness, comments] = await Promise.all([
    listFinalSnapshotsForReport(id).catch(() => []),
    getReportReadiness(id).catch(() => null),
    listReportComments(id).catch(() => [])
  ]);

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
      <main className="space-y-6">
        <header className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <Link href={`/investigations/${report.investigation_id}`} className="text-sm font-medium text-slate-500 hover:text-slate-900">
            Back to investigation
          </Link>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-semibold tracking-tight text-ink">{report.title}</h1>
          </div>
          <p className="mt-2 text-sm text-slate-600">
            Report draft generated from reviewed investigation findings and charts.
          </p>
          <a
            href={reportPdfDownloadUrl(report.report_id)}
            className="mt-4 inline-flex rounded-md bg-slate-950 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-950 dark:hover:bg-slate-200"
          >
            Download PDF
          </a>
        </header>

        <ReportViewModeToggle report={report} comments={comments} />
      </main>

      <aside className="space-y-4">
        <Card>
          <h2 className="text-sm font-semibold text-slate-950">Report details</h2>
          <div className="mt-3 space-y-3 text-sm">
            <MetadataItem label="Type" value={reportTemplateLabel(report.template)} />
            <MetadataItem label="Version" value={`Version ${report.version}`} />
            <MetadataItem label="Updated" value={formatDate(report.updated_at)} />
            <MetadataItem label="Source findings" value={String(report.source_finding_ids.length)} />
            <MetadataItem label="Charts and tables" value={String(report.source_artifact_ids.length)} />
          </div>
        </Card>

        <details className="rounded-2xl border border-slate-200/80 bg-white/90 p-5 shadow-sm shadow-slate-200/60 dark:border-slate-800 dark:bg-slate-950/85 dark:shadow-black/20">
          <summary className="cursor-pointer text-sm font-semibold text-slate-950">Review details</summary>
          <div className="mt-4 space-y-4">
            <ReportReviewSummary report={report} comments={comments} readiness={readiness} />
            <ReadinessCard readiness={readiness} />
          </div>
        </details>

        <ReportFinalizePanel
          reportId={report.report_id}
          initialApprovalStatus={report.approval_status}
          initialSnapshots={snapshots}
          readiness={readiness}
        />

      </aside>
    </div>
  );
}

function reportTemplateLabel(value: string) {
  if (value === "product_decision_memo") return "Decision memo";
  if (value === "technical_appendix") return "Technical details";
  return "Executive summary";
}

function MetadataItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-slate-500">{label}</div>
      <div className="mt-1 break-words font-medium text-slate-900">{value}</div>
    </div>
  );
}

function ReadinessCard({
  readiness
}: {
  readiness: ReportReadinessResult | null;
}) {
  const blockingChecks =
    readiness?.checks?.filter((check) => check.blocking === true || check.severity === "error") || [];
  const warningChecks =
    readiness?.checks?.filter((check) => check.severity === "warning" && check.blocking !== true) || [];

  return (
    <Card>
      <h2 className="text-sm font-semibold text-slate-950">Readiness</h2>
      {readiness ? (
        <div className="mt-3">
          <StatusBadge value={readiness.is_ready ? "ready" : "needs_attention"} />
          <p className="mt-3 text-sm text-slate-600">
            {readiness.blocking_count} blocking checks
            <br />
            {readiness.warning_count} warnings
          </p>
          {blockingChecks.length ? (
            <div className="mt-4">
              <div className="text-xs font-medium text-red-700">Blocking</div>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-red-700">
                {blockingChecks.slice(0, 5).map((check, index) => (
                  <li key={String(check.id || check.name || index)}>{String(check.message || check.name || "Readiness check failed")}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {warningChecks.length ? (
            <div className="mt-4">
              <div className="text-xs font-medium text-amber-700">Warnings</div>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-amber-700">
                {warningChecks.slice(0, 4).map((check, index) => (
                  <li key={String(check.id || check.name || index)}>{String(check.message || check.name || "Readiness warning")}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : (
        <p className="mt-3 text-sm text-slate-500">Readiness is unavailable.</p>
      )}
    </Card>
  );
}
