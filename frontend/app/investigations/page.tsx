import Link from "next/link";
import { getInvestigationRuns, listDataSources, listInvestigations } from "@/lib/api";
import { DeleteInvestigationAction } from "@/components/DeleteActions";
import { CreateInvestigationForm } from "@/components/CreateInvestigationForm";
import { Button, Card, EmptyState, StatusBadge, formatDate } from "@/components/ui";
import { productInvestigations } from "@/lib/investigations";

export const dynamic = "force-dynamic";

export default async function InvestigationsPage() {
  const [investigations, dataSources] = await Promise.all([
    listInvestigations(),
    listDataSources()
  ]);
  const visibleInvestigations = productInvestigations(investigations);
  const runCounts = await Promise.all(
    visibleInvestigations.map(async (item) => {
      try {
        const runs = await getInvestigationRuns(item.investigation_id);
        return [item.investigation_id, runs.length] as const;
      } catch {
        return [item.investigation_id, 0] as const;
      }
    })
  );
  const runCountById = Object.fromEntries(runCounts);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Investigations</h1>
        <p className="mt-1 text-sm text-muted">
          Persistent analytical workspaces organized around questions, insights, charts and reports.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <CreateInvestigationForm dataSources={dataSources} />

        <div className="space-y-3">
          {visibleInvestigations.length === 0 ? (
            <EmptyState>No investigations yet. Create one from the Start new analysis panel.</EmptyState>
          ) : (
            visibleInvestigations.map((item) => (
              <Card key={item.investigation_id} className="transition hover:border-slate-300 hover:shadow">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <h2 className="font-semibold text-slate-950 dark:text-slate-50">{item.title}</h2>
                      <StatusBadge value={item.status} />
                    </div>
                    <p className="mt-2 max-w-3xl text-sm text-slate-600 dark:text-slate-300">{item.user_question}</p>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500 dark:text-slate-400">
                      <span>Created {formatDate(item.created_at)}</span>
                      <span>{item.linked_data_source_ids?.length || 0} datasets</span>
                      <span>{runCountById[item.investigation_id] || 0} analyses</span>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center justify-end gap-2">
                    <Link href={`/investigations/${item.investigation_id}`}>
                      <Button variant="secondary" size="sm">Open workspace</Button>
                    </Link>
                    <DeleteInvestigationAction investigationId={item.investigation_id} />
                  </div>
                </div>
              </Card>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
