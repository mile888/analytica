import Link from "next/link";
import { listDataSources } from "@/lib/api";
import { CreateDataSourceForm } from "@/components/CreateDataSourceForm";
import { UploadCsvDataSourceForm } from "@/components/UploadCsvDataSourceForm";
import { Card, EmptyState, StatusBadge, formatDate } from "@/components/ui";

export const dynamic = "force-dynamic";

export default async function DataSourcesPage() {
  const sources = await listDataSources();

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Data Sources</h1>
        <p className="mt-1 text-sm text-muted">
          Productized data sources with profiles, usage context and human semantic notes.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <div className="space-y-4">
          <UploadCsvDataSourceForm />
          <CreateDataSourceForm />
        </div>

        <div className="space-y-3">
          {sources.length === 0 ? (
            <EmptyState>No data sources yet. Create a metadata source to start the workflow.</EmptyState>
          ) : (
            sources.map((source) => (
              <Link key={source.data_source_id} href={`/data-sources/${source.data_source_id}`}>
                <Card className="transition hover:border-slate-300 hover:shadow">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <div className="flex items-center gap-2">
                        <h2 className="font-semibold text-slate-950">{source.name}</h2>
                        <StatusBadge value={source.status} />
                      </div>
                      <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-500">
                        <span>{source.data_source_type}</span>
                        <span>Updated {formatDate(source.updated_at)}</span>
                        <span>{source.linked_investigation_ids.length} linked investigations</span>
                      </div>
                      {source.description ? (
                        <p className="mt-2 max-w-3xl text-sm text-slate-600">{source.description}</p>
                      ) : null}
                      {source.tags.length ? (
                        <div className="mt-3 flex flex-wrap gap-1">
                          {source.tags.map((tag) => (
                            <span key={tag} className="rounded bg-slate-100 px-2 py-1 text-xs text-slate-600">{tag}</span>
                          ))}
                        </div>
                      ) : null}
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
