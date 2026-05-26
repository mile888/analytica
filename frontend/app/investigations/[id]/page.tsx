import { notFound } from "next/navigation";
import {
  getDataSource,
  getDataSourceProfile,
  getInvestigation,
  getInvestigationBranches,
  getInvestigationRuns,
  getInvestigationSuggestedQuestions,
  getUsageContext,
  listFinalSnapshotsForReport,
  listInvestigationMessages,
  listReportsForInvestigation
} from "@/lib/api";
import { ChatWorkspace } from "@/components/ChatWorkspace";
import { InvestigationDataContextItem } from "@/components/InvestigationDataContext";
import { InvestigationKnowledgeSidebar } from "@/components/InvestigationKnowledgeSidebar";
import { InvestigationRunRail } from "@/components/InvestigationRunRail";
import { DeleteInvestigationAction } from "@/components/DeleteActions";

export const dynamic = "force-dynamic";
export const revalidate = 0;
export const fetchCache = "force-no-store";

export default async function InvestigationDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let investigation;
  try {
    investigation = await getInvestigation(id);
  } catch {
    notFound();
  }
  const runs = await getInvestigationRuns(id);
  const messages = await listInvestigationMessages(id);
  const suggestedQuestions = await getInvestigationSuggestedQuestions(id, 8).catch(() => ({ suggestions: [] }));
  const reports = await listReportsForInvestigation(id).catch(() => []);
  const branches = await getInvestigationBranches(id).catch(() => []);
  const snapshotsByReport = Object.fromEntries(
    await Promise.all(
      reports.map(async (report) => [
        report.report_id,
        await listFinalSnapshotsForReport(report.report_id).catch(() => [])
      ])
    )
  );
  const linkedDataSourceIds = investigation.linked_data_source_ids || [];
  const dataContextItems = (
    await Promise.all(
      linkedDataSourceIds.map(async (sourceId): Promise<InvestigationDataContextItem | null> => {
        try {
          const [source, profile, usageContext] = await Promise.all([
            getDataSource(sourceId),
            getDataSourceProfile(sourceId).catch(() => null),
            getUsageContext(sourceId).catch(() => null)
          ]);
          return { source, profile, usageContext };
        } catch {
          return null;
        }
      })
    )
  ).filter((item): item is InvestigationDataContextItem => Boolean(item));
  const latestRun = runs[0];

  return (
    <div className="relative -mx-4 -my-6 min-h-[calc(100vh-4rem)] bg-slate-100/60 p-3 dark:bg-slate-950 sm:-mx-6 sm:p-4 lg:-mx-8">
      <div className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-72 bg-blue-500/5 blur-3xl dark:bg-blue-400/10" />
      <div className="grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)_360px]">
        <div className="order-2 xl:order-1">
          <InvestigationKnowledgeSidebar
            investigationId={investigation.investigation_id}
            linkedDataSourceIds={linkedDataSourceIds}
            findings={investigation.findings || []}
            artifacts={investigation.artifacts || []}
            reports={reports}
            snapshotsByReport={snapshotsByReport}
          />
          <details className="mt-3 rounded-2xl border border-red-200/70 bg-white/90 p-4 shadow-sm dark:border-red-950/70 dark:bg-slate-950/85">
            <summary className="cursor-pointer text-sm font-semibold text-red-700 dark:text-red-300">Danger zone</summary>
            <p className="mt-2 text-xs leading-5 text-slate-500 dark:text-slate-400">
              Permanently delete this investigation.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <DeleteInvestigationAction investigationId={investigation.investigation_id} />
            </div>
          </details>
        </div>

        <main className="order-1 min-w-0 xl:order-2">
          <ChatWorkspace
            investigation={investigation}
            latestRun={latestRun}
            messages={messages}
            findings={investigation.findings || []}
            artifacts={(investigation.artifacts || []).filter((artifact) => artifact.visibility !== "hidden")}
            suggestions={suggestedQuestions.suggestions}
            branches={branches}
            dataSources={dataContextItems.map((item) => item.source)}
          />
        </main>

        <div className="order-3">
          <InvestigationRunRail
            investigationId={investigation.investigation_id}
            linkedDataSourceIds={linkedDataSourceIds}
            dataContextItems={dataContextItems}
            suggestions={suggestedQuestions.suggestions}
            artifacts={(investigation.artifacts || []).filter((artifact) => artifact.visibility !== "hidden")}
            reports={reports}
            branches={branches}
          />
        </div>
      </div>
    </div>
  );
}
