import { notFound } from "next/navigation";
import { getInvestigation, getInvestigationRuns } from "@/lib/api";
import { RunInvestigationPanel } from "@/components/RunInvestigationPanel";
import { Card, StatusBadge, formatDate } from "@/components/ui";

export const dynamic = "force-dynamic";

export default async function InvestigationDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let investigation;
  try {
    investigation = await getInvestigation(id);
  } catch {
    notFound();
  }
  const runs = await getInvestigationRuns(id);

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_380px]">
      <div className="space-y-6">
        <header className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight text-ink">{investigation.title}</h1>
            <StatusBadge value={investigation.status} />
          </div>
          <p className="max-w-4xl text-sm text-slate-600">{investigation.user_question}</p>
          <div className="flex flex-wrap gap-2 text-xs text-slate-500">
            <span>Updated {formatDate(investigation.updated_at)}</span>
            <span>{investigation.linked_data_source_ids?.length || 0} linked data sources</span>
          </div>
        </header>

        <Card>
          <h2 className="text-sm font-semibold text-slate-950">Findings</h2>
          {investigation.findings?.length ? (
            <div className="mt-3 grid gap-3">
              {investigation.findings.slice(0, 6).map((finding) => (
                <div key={finding.finding_id} className="rounded-md border border-slate-200 p-3">
                  <div className="flex items-center gap-2">
                    <strong className="text-sm">{finding.title || "Finding"}</strong>
                    <StatusBadge value={finding.status} />
                  </div>
                  <p className="mt-2 text-sm text-slate-600">{finding.text}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="mt-3 text-sm text-slate-500">No findings yet.</p>
          )}
        </Card>

        <Card>
          <h2 className="text-sm font-semibold text-slate-950">Artifacts</h2>
          {investigation.artifacts?.length ? (
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {investigation.artifacts.slice(0, 8).map((artifact) => (
                <div key={artifact.artifact_id} className="rounded-md border border-slate-200 p-3 text-sm">
                  <div className="font-medium text-slate-800">{artifact.title}</div>
                  <div className="mt-1 text-xs text-slate-500">{artifact.type || artifact.artifact_type} · {artifact.visibility}</div>
                </div>
              ))}
            </div>
          ) : (
            <p className="mt-3 text-sm text-slate-500">No artifacts yet.</p>
          )}
        </Card>
      </div>

      <aside className="space-y-4">
        <RunInvestigationPanel
          investigationId={investigation.investigation_id}
          linkedDataSourceIds={investigation.linked_data_source_ids || []}
          initialRuns={runs}
        />
      </aside>
    </div>
  );
}
