import { ReportComment, ReportReadinessResult, ShareableReport } from "@/lib/api";
import { Card, StatusBadge } from "@/components/ui";

export interface ReportReviewSummaryCounts {
  totalSections: number;
  approvedSections: number;
  draftSections: number;
  changesRequestedSections: number;
  needsReviewSections: number;
  openComments: number;
  resolvedComments: number;
  blockingChecks: number;
  warningChecks: number;
  overallStatus: "Ready to finalize" | "Needs attention" | "Changes requested" | "Draft";
  firstOpenCommentSectionId?: string;
  firstChangesRequestedSectionId?: string;
}

export function buildReportReviewSummary(
  report: ShareableReport,
  comments: ReportComment[],
  readiness: ReportReadinessResult | null
): ReportReviewSummaryCounts {
  const totalSections = report.sections.length;
  const approvedSections = report.sections.filter((section) => section.review_status === "approved").length;
  const changesRequestedSections = report.sections.filter(
    (section) => section.review_status === "changes_requested"
  ).length;
  const needsReviewSections = report.sections.filter((section) => section.review_status === "needs_review").length;
  const draftSections = report.sections.filter((section) => section.review_status === "draft").length;
  const openComments = comments.filter((comment) => comment.status === "open").length;
  const resolvedComments = comments.filter((comment) => comment.status === "resolved").length;
  const blockingChecks = readiness?.blocking_count || 0;
  const warningChecks = readiness?.warning_count || 0;
  const firstOpenCommentSectionId = comments.find((comment) => comment.status === "open")?.section_id;
  const firstChangesRequestedSectionId = report.sections.find(
    (section) => section.review_status === "changes_requested"
  )?.section_id;

  let overallStatus: ReportReviewSummaryCounts["overallStatus"] = "Draft";
  if (changesRequestedSections > 0 || report.approval_status === "changes_requested") {
    overallStatus = "Changes requested";
  } else if (blockingChecks > 0 || openComments > 0) {
    overallStatus = "Needs attention";
  } else if (readiness?.is_ready) {
    overallStatus = "Ready to finalize";
  }

  return {
    totalSections,
    approvedSections,
    draftSections,
    changesRequestedSections,
    needsReviewSections,
    openComments,
    resolvedComments,
    blockingChecks,
    warningChecks,
    overallStatus,
    firstOpenCommentSectionId,
    firstChangesRequestedSectionId
  };
}

export function ReportReviewSummary({
  report,
  comments,
  readiness
}: {
  report: ShareableReport;
  comments: ReportComment[];
  readiness: ReportReadinessResult | null;
}) {
  const summary = buildReportReviewSummary(report, comments, readiness);

  return (
    <Card>
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-950">Report Review</h2>
          <p className="mt-1 text-xs text-slate-500">
            {summary.approvedSections} / {summary.totalSections} sections approved
          </p>
        </div>
        <StatusBadge value={summary.overallStatus} />
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-2 text-sm">
        <SummaryMetric label="Approved" value={summary.approvedSections} />
        <SummaryMetric label="Draft" value={summary.draftSections} />
        <SummaryMetric label="Needs review" value={summary.needsReviewSections} />
        <SummaryMetric label="Changes requested" value={summary.changesRequestedSections} />
        <SummaryMetric label="Open comments" value={summary.openComments} />
        <SummaryMetric label="Resolved comments" value={summary.resolvedComments} />
        <SummaryMetric label="Blocking issues" value={summary.blockingChecks} tone={summary.blockingChecks ? "danger" : "default"} />
        <SummaryMetric label="Warnings" value={summary.warningChecks} tone={summary.warningChecks ? "warning" : "default"} />
      </dl>

      <div className="mt-4 space-y-2 text-xs">
        {summary.firstOpenCommentSectionId ? (
          <a
            href={`#section-${summary.firstOpenCommentSectionId}`}
            className="block rounded-md border border-slate-200 px-3 py-2 font-medium text-slate-700 hover:bg-slate-50"
          >
            Jump to first open comment
          </a>
        ) : null}
        {summary.firstChangesRequestedSectionId ? (
          <a
            href={`#section-${summary.firstChangesRequestedSectionId}`}
            className="block rounded-md border border-slate-200 px-3 py-2 font-medium text-slate-700 hover:bg-slate-50"
          >
            Jump to changes requested
          </a>
        ) : null}
      </div>
    </Card>
  );
}

function SummaryMetric({
  label,
  value,
  tone = "default"
}: {
  label: string;
  value: number;
  tone?: "default" | "warning" | "danger";
}) {
  const valueTone =
    tone === "danger" ? "text-red-700" : tone === "warning" ? "text-amber-700" : "text-slate-950";
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className={`mt-1 text-lg font-semibold ${valueTone}`}>{value}</dd>
    </div>
  );
}
