import Link from "next/link";
import { listDataSources } from "@/lib/api";
import { DeleteDataSourceAction } from "@/components/DeleteActions";
import { UploadCsvDataSourceForm } from "@/components/UploadCsvDataSourceForm";
import { Button, Card, EmptyState, StatusBadge, formatDate } from "@/components/ui";

export const dynamic = "force-dynamic";

export default async function DataSourcesPage() {
  const sources = await listDataSources();

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Datasets</h1>
        <p className="mt-1 text-sm text-muted">
          Upload a CSV, let Analytica profile it, then start an investigation from real data.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <div className="space-y-4">
          <UploadCsvDataSourceForm />
        </div>

        <div className="space-y-3">
          {sources.length === 0 ? (
            <EmptyState>No datasets yet. Upload a CSV to start an analytical investigation.</EmptyState>
          ) : (
            sources.map((source) => (
              <Card key={source.data_source_id} className="transition hover:border-slate-300 hover:shadow">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <h2 className="font-semibold text-slate-950 dark:text-slate-50">{source.name}</h2>
                      <StatusBadge value={source.status} />
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-500 dark:text-slate-400">
                      <span>{source.data_source_type}</span>
                      <span>Updated {formatDate(source.updated_at)}</span>
                      <span>{source.linked_investigation_ids.length} investigations</span>
                    </div>
                    {source.description ? (
                      <p className="mt-2 max-w-3xl text-sm text-slate-600 dark:text-slate-300">{source.description}</p>
                    ) : null}
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <Link href={`/data-sources/${source.data_source_id}`}>
                      <Button variant="secondary" size="sm">Open</Button>
                    </Link>
                    <DeleteDataSourceAction dataSourceId={source.data_source_id} />
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
