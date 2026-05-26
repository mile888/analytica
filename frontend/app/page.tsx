import Link from "next/link";
import type { ReactNode } from "react";
import {
  Investigation,
  listDataSources,
  listInvestigations
} from "@/lib/api";
import { Card, EmptyState, StatusBadge, formatDate } from "@/components/ui";
import { isHighConfidenceKeyFinding } from "@/lib/display";
import { productInvestigations } from "@/lib/investigations";

export const dynamic = "force-dynamic";
export const revalidate = 0;
export const fetchCache = "force-no-store";

export default async function DashboardPage() {
  const [investigations, dataSources] = await Promise.all([
    listInvestigations().catch(() => []),
    listDataSources().catch(() => [])
  ]);

  const visibleInvestigations = productInvestigations(investigations);
  const recentInvestigations = [...visibleInvestigations]
    .sort((a, b) => Date.parse(b.updated_at || b.created_at) - Date.parse(a.updated_at || a.created_at))
    .slice(0, 5);

  const recentDataSources = [...dataSources]
    .sort((a, b) => Date.parse(b.updated_at || b.created_at) - Date.parse(a.updated_at || a.created_at))
    .slice(0, 5);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">Analytica workspace</h1>
        <p className="mt-1 text-sm text-muted">
          Start with a dataset, continue an investigation, or turn reviewed findings into a report.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2">
        <DashboardMetric label="Investigations" value={visibleInvestigations.length} />
        <DashboardMetric label="Datasets" value={dataSources.length} />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <main>
          <DashboardSection title="Recent investigations" empty="No investigations yet.">
            {recentInvestigations.map((investigation) => (
              <DashboardRow
                key={investigation.investigation_id}
                href={`/investigations/${investigation.investigation_id}`}
                title={investigation.title}
                meta={`${keyFindingCount(investigation)} key findings · updated ${formatDate(investigation.updated_at)}`}
                badge={investigation.status}
              />
            ))}
          </DashboardSection>
        </main>

        <aside>
          <DashboardSection title="Recent datasets" empty="No data sources yet.">
            {recentDataSources.map((source) => (
              <DashboardRow
                key={source.data_source_id}
                href={`/data-sources/${source.data_source_id}`}
                title={source.name}
                meta={`Updated ${formatDate(source.updated_at)} · ${source.linked_investigation_ids.length} investigations`}
                badge={source.status}
              />
            ))}
          </DashboardSection>
        </aside>
      </div>
    </div>
  );
}

function keyFindingCount(investigation: Investigation) {
  return (investigation.findings || []).filter(isHighConfidenceKeyFinding).length;
}

function DashboardMetric({ label, value }: { label: string; value: number }) {
  return (
    <Card>
      <div className="text-2xl font-semibold text-slate-950 dark:text-slate-50">{value}</div>
      <div className="mt-1 text-sm text-slate-500 dark:text-slate-400">{label}</div>
    </Card>
  );
}

function DashboardSection({
  title,
  empty,
  children
}: {
  title: string;
  empty: string;
  children: ReactNode;
}) {
  const items = Array.isArray(children) ? children.filter(Boolean) : children;
  const hasItems = Array.isArray(items) ? items.length > 0 : Boolean(items);
  return (
    <Card>
      <h2 className="text-sm font-semibold text-slate-950 dark:text-slate-50">{title}</h2>
      <div className="mt-3 space-y-2">
        {hasItems ? items : <EmptyState>{empty}</EmptyState>}
      </div>
    </Card>
  );
}

function DashboardRow({
  href,
  title,
  meta,
  badge
}: {
  href: string;
  title: string;
  meta: string;
  badge: string;
}) {
  return (
    <Link
      href={href}
      className="block rounded-md border border-slate-200 p-3 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-900"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium text-slate-950 dark:text-slate-50">{title}</div>
          <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{meta}</div>
        </div>
        <StatusBadge value={badge} />
      </div>
    </Link>
  );
}
