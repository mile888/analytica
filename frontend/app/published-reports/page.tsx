import { listFinalReports } from "@/lib/api";
import { Card, EmptyState, StatusBadge, formatDate } from "@/components/ui";

export const dynamic = "force-dynamic";

export default async function PublishedReportsPage({
  searchParams
}: {
  searchParams: Promise<{ search?: string; decision_status?: string }>;
}) {
  const params = await searchParams;
  const reports = await listFinalReports({
    search: params.search,
    decision_status: params.decision_status && params.decision_status !== "all" ? params.decision_status : undefined
  });

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Published Reports</h1>
        <p className="mt-1 text-sm text-muted">
          Final deliverables with decision metadata, readiness snapshot and approval state.
        </p>
      </header>

      <form className="grid gap-3 rounded-lg border border-slate-200 bg-white p-4 sm:grid-cols-[1fr_220px_auto]">
        <input
          name="search"
          defaultValue={params.search || ""}
          placeholder="Search title, owner, area, tags"
          className="rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500"
        />
        <select
          name="decision_status"
          defaultValue={params.decision_status || "all"}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500"
        >
          <option value="all">All decision statuses</option>
          <option value="proposed">Proposed</option>
          <option value="accepted">Accepted</option>
          <option value="rejected">Rejected</option>
          <option value="superseded">Superseded</option>
          <option value="unknown">Unknown</option>
        </select>
        <button className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white">Filter</button>
      </form>

      {reports.length === 0 ? (
        <EmptyState>No published reports match this view.</EmptyState>
      ) : (
        <div className="grid gap-3">
          {reports.map((report) => {
            const metadata = report.decision_metadata;
            return (
              <Card key={report.snapshot_id}>
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="font-semibold text-slate-950">{report.title}</h2>
                      <StatusBadge value={report.status} />
                      <StatusBadge value={metadata?.decision_status || "unknown"} />
                    </div>
                    <p className="mt-2 text-sm text-slate-600">
                      {metadata?.short_description || report.investigation_title || "No short description yet."}
                    </p>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                      <span>Created {formatDate(report.created_at)}</span>
                      <span>Version {report.report_version}</span>
                      <span>Approval {report.approval_status}</span>
                      <span>Owner {metadata?.owner || "unassigned"}</span>
                      <span>Area {metadata?.business_area || "unset"}</span>
                      <span>
                        Readiness {report.readiness_is_ready ? "ready" : "needs attention"}
                        {typeof report.readiness_warning_count === "number"
                          ? ` · ${report.readiness_warning_count} warnings`
                          : ""}
                      </span>
                    </div>
                    {metadata?.tags?.length ? (
                      <div className="mt-3 flex flex-wrap gap-1">
                        {metadata.tags.map((tag) => (
                          <span key={tag} className="rounded bg-slate-100 px-2 py-1 text-xs text-slate-600">{tag}</span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
