import Link from "next/link";
import { DataSource, DataSourceProfile, DataSourceUsageContext } from "@/lib/api";
import { datasetExecutionStatus } from "@/lib/display";
import { Card, Chip, EmptyState, SectionHeader } from "@/components/ui";

export interface InvestigationDataContextItem {
  source: DataSource;
  profile?: DataSourceProfile | null;
  usageContext?: DataSourceUsageContext | null;
}

export function InvestigationDataContext({ items }: { items: InvestigationDataContextItem[] }) {
  return (
    <Card>
      <SectionHeader
        title="Dataset"
        description="Linked data and detected analysis fields."
      />

      {items.length ? (
        <div className="mt-4 space-y-3">
          {items.map(({ source, profile, usageContext }) => {
            const keyColumns = usageContext?.column_summaries
              ?.filter((column) => ["metric", "dimension", "timestamp", "identifier", "target"].includes(column.inferred_role))
              .slice(0, 5);
            const caveats = usageContext?.caveats?.slice(0, 2) || [];
            return (
              <div key={source.data_source_id} className="rounded-xl border border-slate-200 bg-slate-50/70 p-3 transition hover:border-slate-300 dark:border-slate-800 dark:bg-slate-900/50">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <Link href={`/data-sources/${source.data_source_id}`} className="font-semibold text-slate-900 hover:underline dark:text-slate-100">
                    {source.name}
                  </Link>
                  <DatasetStateChip source={source} />
                </div>
                <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-500 dark:text-slate-400">
                  {profile ? <Chip>{profile.row_count} rows · {profile.column_count} columns</Chip> : <Chip>Profile pending</Chip>}
                </div>
                {keyColumns?.length ? (
                  <div className="mt-3 flex flex-wrap gap-1">
                    {keyColumns.map((column) => (
                      <span key={column.name} className="rounded-full bg-white px-2 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 dark:bg-slate-950 dark:text-slate-300 dark:ring-slate-800">
                        {column.name}
                      </span>
                    ))}
                  </div>
                ) : null}
                {usageContext?.description ? (
                  <p className="mt-3 text-xs leading-5 text-slate-600 dark:text-slate-400">
                    {usageContext.description}
                  </p>
                ) : null}
                {caveats.length ? (
                  <ul className="mt-3 list-disc space-y-1 pl-5 text-xs leading-5 text-slate-600 dark:text-slate-400" aria-label="Dataset watchouts">
                    {caveats.map((caveat) => <li key={caveat}>{caveat}</li>)}
                  </ul>
                ) : null}
                {(usageContext?.column_summaries?.length || 0) > 5 ? (
                  <div className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                    +{usageContext!.column_summaries.length - 5} more detected columns
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="mt-4"><EmptyState>No dataset is linked yet.</EmptyState></div>
      )}
    </Card>
  );
}

function DatasetStateChip({ source }: { source: DataSource }) {
  const label = datasetExecutionStatus(source);
  const tone = label === "Parsing failed"
    ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300"
    : label === "Ready for analysis"
      ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-300"
      : "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-300";
  return <span className={`rounded-full border px-2 py-1 text-[11px] font-semibold ${tone}`}>{label}</span>;
}
