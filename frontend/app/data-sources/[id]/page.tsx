import { notFound } from "next/navigation";
import { getDataSource, getDataSourceProfile, getUsageContext } from "@/lib/api";
import { DeleteDataSourceAction } from "@/components/DeleteActions";
import { Card, EmptyState, StatusBadge, formatDate } from "@/components/ui";

export const dynamic = "force-dynamic";
export const revalidate = 0;
export const fetchCache = "force-no-store";

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
          </div>
          {source.description ? <p className="text-sm text-slate-600">{source.description}</p> : null}
          <div className="flex flex-wrap gap-2 text-xs text-slate-500">
            <span>Created {formatDate(source.created_at)}</span>
            <span>{source.linked_investigation_ids.length} linked investigations</span>
          </div>
        </header>

        <Card>
          <h2 className="text-sm font-semibold text-slate-950">Profile summary</h2>
          {profile ? (
            <>
              {mayBeSingleColumnParse(profile) ? (
                <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                  This dataset may be parsed as a single column. Check delimiter or re-upload with correct CSV settings.
                </div>
              ) : null}
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
                <div className="mt-4">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Preview</h3>
                  <div className="mt-2 overflow-auto rounded-md border border-slate-200">
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
                </div>
              ) : null}
            </>
          ) : (
            <EmptyState>No profile stored for this source.</EmptyState>
          )}
        </Card>

        <Card>
          <details>
            <summary className="cursor-pointer text-sm font-semibold text-slate-950">Dataset details</summary>
            {context ? (
              <div className="mt-4 space-y-5">
                {context.description ? <p className="text-sm text-slate-600">{context.description}</p> : null}
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Column roles</h3>
                  <div className="mt-2 grid gap-2">
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
                {context.caveats.length ? (
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Caveats</h3>
                    <ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-slate-600">
                      {context.caveats.map((caveat) => <li key={caveat}>{caveat}</li>)}
                    </ul>
                  </div>
                ) : null}
                {context.previous_questions.length ? (
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Previous questions</h3>
                    <ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-slate-600">
                      {context.previous_questions.map((question) => <li key={question}>{question}</li>)}
                    </ul>
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="mt-4">
                <EmptyState>No additional dataset details are available.</EmptyState>
              </div>
            )}
          </details>
        </Card>
      </div>

      <aside className="space-y-4">
        <Card>
          <h2 className="text-sm font-semibold text-slate-950">Danger zone</h2>
          <p className="mt-2 text-sm text-slate-500">Permanently delete this dataset.</p>
          <div className="mt-3">
            <DeleteDataSourceAction dataSourceId={source.data_source_id} />
          </div>
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

function mayBeSingleColumnParse(profile: {
  column_count: number;
  columns: Array<{ name: string; sample_values: unknown[] }>;
  sampled_rows: Record<string, unknown>[];
}) {
  if (profile.column_count !== 1) return false;
  const columnName = profile.columns[0]?.name || "";
  const columnSamples = profile.columns[0]?.sample_values?.map((value) => String(value ?? "")) || [];
  const sampleValues = [
    columnName,
    ...columnSamples,
    ...profile.sampled_rows.slice(0, 3).flatMap((row) => Object.values(row).map((value) => String(value ?? "")))
  ];
  return sampleValues.some((value) => /[,;\t|]/.test(value));
}
