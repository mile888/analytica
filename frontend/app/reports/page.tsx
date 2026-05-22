import Link from "next/link";
import { listFinalSnapshotsForReport, listInvestigations, listReportsForInvestigation, type Investigation, type ShareableReport } from "@/lib/api";
import { Card, formatDate } from "@/components/ui";

export const dynamic = "force-dynamic";

export default async function ReportsPage() {
  const investigations = await listInvestigations();
  const reportsByInvestigation = await Promise.all(
    investigations.map(async (investigation) => ({
      investigation,
      reports: await listReportsForInvestigation(investigation.investigation_id).catch(() => [])
    }))
  );
  const reports = reportsByInvestigation
    .flatMap(({ investigation, reports }) => reports.map((report) => ({ report, investigation })))
    .filter(({ report }) => report.is_latest !== false)
    .reduce<Array<{ report: ShareableReport; investigation: Investigation }>>((items, item) => {
      const key = `${item.report.investigation_id}:${item.report.template}`;
      const existingIndex = items.findIndex((candidate) => `${candidate.report.investigation_id}:${candidate.report.template}` === key);
      if (existingIndex === -1) {
        items.push(item);
      } else if (item.report.updated_at > items[existingIndex].report.updated_at) {
        items[existingIndex] = item;
      }
      return items;
    }, [])
    .sort((a, b) => b.report.updated_at.localeCompare(a.report.updated_at));

  const finalCounts = Object.fromEntries(
    await Promise.all(
      reports.map(async ({ report }) => [
        report.report_id,
        (await listFinalSnapshotsForReport(report.report_id).catch(() => [])).length
      ])
    )
  );

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Reports</h1>
        <p className="mt-1 text-sm text-muted">Reports created from reviewed investigation findings and charts.</p>
      </header>

      {reports.length ? (
        <div className="space-y-3">
          {reports.map(({ report, investigation }) => (
            <Link key={report.report_id} href={`/reports/${report.report_id}`}>
              <Card className="transition hover:border-slate-300 hover:shadow">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="font-semibold text-slate-950">{report.title}</h2>
                    </div>
                    <p className="mt-2 text-sm text-slate-600">From {investigation.title}</p>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                      <span>{reportTemplateLabel(report.template)}</span>
                      <span>Updated {formatDate(report.updated_at)}</span>
                      {finalCounts[report.report_id] ? <span>Final version saved</span> : null}
                    </div>
                  </div>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      ) : (
        <Card>
          <div className="py-8 text-center">
            <h2 className="text-base font-semibold text-slate-950 dark:text-slate-50">No reports yet.</h2>
            <p className="mx-auto mt-2 max-w-md text-sm text-slate-500 dark:text-slate-400">
              Create a report from an investigation after reviewing findings and charts.
            </p>
            <Link
              href="/investigations"
              className="mt-4 inline-flex rounded-md bg-slate-950 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-950 dark:hover:bg-slate-200"
            >
              Open investigations
            </Link>
          </div>
        </Card>
      )}
    </div>
  );
}

function reportTemplateLabel(value: string) {
  if (value === "product_decision_memo") return "Decision memo";
  if (value === "technical_appendix") return "Technical details";
  return "Executive summary";
}
