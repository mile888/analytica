import Link from "next/link";
import { getInvestigationRuns, listDataSources, listInvestigations } from "@/lib/api";
import { CreateInvestigationForm } from "@/components/CreateInvestigationForm";
import { Card, EmptyState, StatusBadge, formatDate } from "@/components/ui";

export const dynamic = "force-dynamic";

export default async function InvestigationsPage() {
  const [investigations, dataSources] = await Promise.all([
    listInvestigations(),
    listDataSources()
  ]);
  const runCounts = await Promise.all(
    investigations.map(async (item) => {
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
          Analytical workspaces organized around questions, runs, artifacts and reviewed outputs.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <CreateInvestigationForm dataSources={dataSources} />

        <div className="space-y-3">
          {investigations.length === 0 ? (
            <EmptyState>No investigations yet. Create one from the Start new analysis panel.</EmptyState>
          ) : (
            investigations.map((item) => (
              <Link key={item.investigation_id} href={`/investigations/${item.investigation_id}`}>
                <Card className="transition hover:border-slate-300 hover:shadow">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <div className="flex items-center gap-2">
                        <h2 className="font-semibold text-slate-950">{item.title}</h2>
                        <StatusBadge value={item.status} />
                      </div>
                      <p className="mt-2 max-w-3xl text-sm text-slate-600">{item.user_question}</p>
                      <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                        <span>Created {formatDate(item.created_at)}</span>
                        <span>{item.linked_data_source_ids?.length || 0} data sources</span>
                        <span>{runCountById[item.investigation_id] || 0} backend runs</span>
                      </div>
                    </div>
                  </div>
                </Card>
              </Link>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
