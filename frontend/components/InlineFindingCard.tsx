import type { Artifact, DataSource, Finding, InvestigationMemoryItem, ShareableReport } from "@/lib/api";
import { FindingEvidenceActions } from "@/components/FindingEvidenceActions";
import type { InvestigationDataContextItem } from "@/components/InvestigationDataContext";
import { StatusBadge } from "@/components/ui";
import { formatFindingForUser } from "@/lib/display";

export function InlineFindingCard({
  investigationId,
  finding,
  artifacts,
  dataSources,
  dataSourceContexts,
  memoryItems,
  reports
}: {
  investigationId: string;
  finding: Finding;
  artifacts: Artifact[];
  dataSources: DataSource[];
  dataSourceContexts: InvestigationDataContextItem[];
  memoryItems: InvestigationMemoryItem[];
  reports: ShareableReport[];
}) {
  const display = formatFindingForUser(finding);
  return (
    <article className="rounded-2xl border border-blue-100 bg-blue-50/70 p-4 shadow-sm dark:border-blue-900/60 dark:bg-blue-950/20">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <h4 className="text-sm font-semibold text-slate-950 dark:text-slate-50">{display.title}</h4>
        <StatusBadge value={finding.status} />
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-300">{display.summary}</p>
      <FindingEvidenceActions
        investigationId={investigationId}
        finding={finding}
        artifacts={artifacts}
        dataSources={dataSources}
        dataSourceContexts={dataSourceContexts}
        memoryItems={memoryItems}
        reports={reports}
      />
    </article>
  );
}
