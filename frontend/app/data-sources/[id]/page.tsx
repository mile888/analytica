import { notFound } from "next/navigation";
import { getDataSource, getDataSourceProfile, getUsageContext } from "@/lib/api";
import { Card, EmptyState, StatusBadge, formatDate } from "@/components/ui";

export const dynamic = "force-dynamic";

export default async function DataSourceDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let source;
  try {
    source = await getDataSource(id);
  } catch {
    notFound();
  }
  const [profile, context] = await Promise.all([
    getDataSourceProfile(id).catch(() => null),
    getUsageContext(id).catch(() => null)
  ]);

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_380px]">
      <div className="space-y-6">
        <header className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight text-ink">{source.name}</h1>
            <StatusBadge value={source.status} />
          </div>
          <p className="text-sm text-slate-600">{source.description || "No description yet."}</p>
          <div className="flex flex-wrap gap-2 text-xs text-slate-500">
            <span>{source.data_source_type}</span>
            {source.location ? <span>{source.location}</span> : null}
            <span>Updated {formatDate(source.updated_at)}</span>
            <span>{source.linked_investigation_ids.length} linked investigations</span>
          </div>
        </header>

        <Card>
          <h2 className="text-sm font-semibold text-slate-950">Profile summary</h2>
          {profile ? (
            <>
              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Metric label="Rows" value={profile.row_count} />
                <Metric label="Columns" value={profile.column_count} />
                <Metric label="Numeric" value={Object.keys(profile.numeric_summary).length} />
                <Metric label="Categorical" value={Object.keys(profile.categorical_summary).length} />
              </div>
              <div className="mt-4 overflow-hidden rounded-md border border-slate-200">
                <table className="w-full text-left text-sm">
                  <thead className="bg-slate-50 text-xs text-slate-500">
                    <tr>
                      <th className="px-3 py-2">Column</th>
                      <th className="px-3 py-2">Type</th>
                      <th className="px-3 py-2">Nullable</th>
                      <th className="px-3 py-2">Unique</th>
                    </tr>
                  </thead>
                  <tbody>
                    {profile.columns.slice(0, 12).map((column) => (
                      <tr key={column.name} className="border-t border-slate-100">
                        <td className="px-3 py-2 font-medium">{column.name}</td>
                        <td className="px-3 py-2 text-slate-600">{column.dtype}</td>
                        <td className="px-3 py-2 text-slate-600">{column.nullable ? "yes" : "no"}</td>
                        <td className="px-3 py-2 text-slate-600">{column.unique_count ?? "n/a"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {profile.sampled_rows.length ? (
                <div className="mt-4 overflow-auto rounded-md border border-slate-200">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 text-slate-500">
                      <tr>
                        {Object.keys(profile.sampled_rows[0]).slice(0, 8).map((key) => (
                          <th key={key} className="px-3 py-2">{key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {profile.sampled_rows.slice(0, 5).map((row, index) => (
                        <tr key={index} className="border-t border-slate-100">
                          {Object.keys(profile.sampled_rows[0]).slice(0, 8).map((key) => (
                            <td key={key} className="max-w-[180px] truncate px-3 py-2 text-slate-600">
                              {String(row[key] ?? "")}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </>
          ) : (
            <EmptyState>No profile stored for this source.</EmptyState>
          )}
        </Card>

        <Card>
          <h2 className="text-sm font-semibold text-slate-950">Usage context</h2>
          {context ? (
            <div className="mt-3 space-y-4">
              {context.description ? <p className="text-sm text-slate-600">{context.description}</p> : null}
              <div className="grid gap-2">
                {context.column_summaries.slice(0, 14).map((column) => (
                  <div key={column.name} className="rounded-md border border-slate-200 p-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-medium text-slate-900">{column.name}</span>
                      <StatusBadge value={column.inferred_role} />
                      <span className="text-xs text-slate-500">{column.dtype}</span>
                    </div>
                    {column.notes.length ? (
                      <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-slate-600">
                        {column.notes.slice(0, 3).map((note) => <li key={note}>{note}</li>)}
                      </ul>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <EmptyState>No usage context available.</EmptyState>
          )}
        </Card>
      </div>

      <aside className="space-y-4">
        <Card>
          <h2 className="text-sm font-semibold text-slate-950">Caveats</h2>
          {context?.caveats.length ? (
            <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-slate-600">
              {context.caveats.map((caveat) => <li key={caveat}>{caveat}</li>)}
            </ul>
          ) : (
            <p className="mt-3 text-sm text-slate-500">No caveats captured.</p>
          )}
        </Card>
        <Card>
          <h2 className="text-sm font-semibold text-slate-950">Previous questions</h2>
          {context?.previous_questions.length ? (
            <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-slate-600">
              {context.previous_questions.map((question) => <li key={question}>{question}</li>)}
            </ul>
          ) : (
            <p className="mt-3 text-sm text-slate-500">No linked questions yet.</p>
          )}
        </Card>
      </aside>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="mt-1 text-xl font-semibold text-slate-950">{value}</div>
    </div>
  );
}
