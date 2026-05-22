"use client";

import Link from "next/link";
import {
  Artifact,
  DataSource,
  InvestigationMemoryItem,
  ShareableReport
} from "@/lib/api";
import { EvidenceReference } from "@/lib/evidence";
import { InvestigationDataContextItem } from "@/components/InvestigationDataContext";
import { TechnicalDetails } from "@/components/TechnicalDetails";
import { Button, StatusBadge, formatDate } from "@/components/ui";
import { formatArtifactForUser, truncateText } from "@/lib/display";

function previewText(value: unknown) {
  if (value === null || value === undefined || value === "") return "No preview content available.";
  if (typeof value === "string") return truncateText(value, 900);
  try {
    return truncateText(JSON.stringify(value, null, 2), 900);
  } catch {
    return truncateText(String(value), 900);
  }
}

function findReportSection(reports: ShareableReport[], sectionId: string) {
  for (const report of reports) {
    const section = report.sections?.find((candidate) => candidate.section_id === sectionId);
    if (section) return { report, section };
  }
  return null;
}

export function EvidenceInspector({
  evidence,
  artifacts,
  dataSourceContexts,
  memoryItems,
  reports,
  onClose
}: {
  evidence: EvidenceReference | null;
  artifacts: Artifact[];
  dataSourceContexts: InvestigationDataContextItem[];
  memoryItems: InvestigationMemoryItem[];
  reports: ShareableReport[];
  onClose: () => void;
}) {
  if (!evidence) return null;

  const artifact = evidence.type === "artifact" ? artifacts.find((item) => item.artifact_id === evidence.id) : undefined;
  const dataSourceContext = evidence.type === "data_source"
    ? dataSourceContexts.find((item) => item.source.data_source_id === evidence.id)
    : undefined;
  const memoryItem = evidence.type === "memory_item" ? memoryItems.find((item) => item.memory_id === evidence.id) : undefined;
  const reportSection = evidence.type === "report_section" ? findReportSection(reports, evidence.id) : null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/30 p-3 backdrop-blur-sm sm:p-6" role="dialog" aria-modal="true">
      <div className="flex h-full w-full max-w-xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-800 dark:bg-slate-950">
        <header className="border-b border-slate-200 p-5 dark:border-slate-800">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <StatusBadge value={evidence.type} />
              </div>
              <h2 className="mt-3 text-lg font-semibold tracking-tight text-slate-950 dark:text-slate-50">
                {evidence.label}
              </h2>
              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                Preview linked evidence without leaving the investigation.
              </p>
            </div>
            <Button type="button" variant="secondary" size="sm" onClick={onClose}>Close</Button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-5">
          {artifact ? <ArtifactPreview artifact={artifact} /> : null}
          {dataSourceContext ? <DataSourcePreview item={dataSourceContext} /> : null}
          {memoryItem ? <MemoryPreview item={memoryItem} /> : null}
          {reportSection ? <ReportSectionPreview report={reportSection.report} section={reportSection.section} /> : null}
          {!artifact && !dataSourceContext && !memoryItem && !reportSection ? (
            <FallbackPreview evidence={evidence} />
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ArtifactPreview({ artifact }: { artifact: Artifact }) {
  const content = artifact.content ?? artifact.metadata?.preview ?? artifact.metadata?.summary;
  const display = formatArtifactForUser(artifact);
  return (
    <section className="space-y-4">
      <PreviewHeader title={display.title} badges={[display.typeLabel]} />
      <p className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm leading-6 text-slate-700 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300">
        {display.description}
      </p>
      <TechnicalDetails title="Artifact payload">
        <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words">
          {previewText(content)}
        </pre>
      </TechnicalDetails>
    </section>
  );
}

function DataSourcePreview({ item }: { item: InvestigationDataContextItem }) {
  return (
    <section className="space-y-4">
      <PreviewHeader title={item.source.name} badges={[item.source.data_source_type, item.source.status]} />
      <div className="grid gap-3 sm:grid-cols-2">
        <Fact label="Rows" value={item.profile ? String(item.profile.row_count) : "Unknown"} />
        <Fact label="Columns" value={item.profile ? String(item.profile.column_count) : "Unknown"} />
      </div>
      {item.usageContext?.caveats?.length ? (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Caveats</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600 dark:text-slate-400">
            {item.usageContext.caveats.slice(0, 5).map((caveat) => <li key={caveat}>{caveat}</li>)}
          </ul>
        </div>
      ) : null}
      <Link href={`/data-sources/${item.source.data_source_id}`} className="inline-flex text-sm font-semibold text-blue-600 hover:text-blue-700 dark:text-blue-300">
        Open data source
      </Link>
    </section>
  );
}

function MemoryPreview({ item }: { item: InvestigationMemoryItem }) {
  return (
    <section className="space-y-4">
      <PreviewHeader title={item.title || item.memory_type} badges={[item.memory_type, item.status]} />
      <p className="whitespace-pre-wrap rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm leading-6 text-slate-700 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300">
        {item.content}
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        <Fact label="Created" value={formatDate(item.created_at)} />
        <Fact label="Updated" value={formatDate(item.updated_at)} />
      </div>
      {item.metadata?.source_type ? <Fact label="Origin" value={`${item.metadata.source_type} · ${item.metadata.source_id || "unknown"}`} /> : null}
    </section>
  );
}

function ReportSectionPreview({
  report,
  section
}: {
  report: ShareableReport;
  section: ShareableReport["sections"][number];
}) {
  return (
    <section className="space-y-4">
      <PreviewHeader title={section.title} badges={["report section", section.review_status]} />
      <p className="whitespace-pre-wrap rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm leading-6 text-slate-700 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300">
        {section.content || "No content yet."}
      </p>
      <Link href={`/reports/${report.report_id}#section-${section.section_id}`} className="inline-flex text-sm font-semibold text-blue-600 hover:text-blue-700 dark:text-blue-300">
        Open report section
      </Link>
    </section>
  );
}

function FallbackPreview({ evidence }: { evidence: EvidenceReference }) {
  return (
    <section className="space-y-4">
      <PreviewHeader title={evidence.label} badges={[evidence.type]} />
      <p className="rounded-xl border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400">
        This evidence reference exists, but no previewable object was loaded for it.
      </p>
      {evidence.href ? <Link href={evidence.href} className="inline-flex text-sm font-semibold text-blue-600">Open linked object</Link> : null}
    </section>
  );
}

function PreviewHeader({ title, badges }: { title: string; badges: Array<string | undefined> }) {
  return (
    <div>
      <div className="flex flex-wrap gap-2">
        {badges.filter(Boolean).map((badge) => <StatusBadge key={badge} value={badge || "unknown"} />)}
      </div>
      <h3 className="mt-3 text-base font-semibold text-slate-950 dark:text-slate-50">{title}</h3>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-900">
      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-sm text-slate-800 dark:text-slate-200">{value}</div>
    </div>
  );
}
