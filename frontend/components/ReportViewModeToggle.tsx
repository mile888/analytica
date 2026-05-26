"use client";

import { useState } from "react";
import { ReportSectionEditor } from "@/components/ReportSectionEditor";
import { ReportComment, ShareableReport } from "@/lib/api";
import { reportPreviewRows, reportTableColumns, reportTableRows, reportTranscriptItems, reportVisualArtifactSnapshots } from "@/lib/reportPreview";

export function ReportViewModeToggle({
  report,
  comments = []
}: {
  report: ShareableReport;
  comments?: ReportComment[];
}) {
  const [mode, setMode] = useState<"preview" | "edit">("preview");
  const orderedSections = [...report.sections].sort((a, b) => a.order - b.order);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-950">Sections</h2>
        <div className="inline-flex rounded-md border border-slate-200 bg-slate-50 p-1">
          <button
            type="button"
            onClick={() => setMode("preview")}
            className={`rounded px-3 py-1.5 text-xs font-medium ${mode === "preview" ? "bg-white text-slate-950 shadow-sm" : "text-slate-500"}`}
          >
            Preview
          </button>
          <button
            type="button"
            onClick={() => setMode("edit")}
            className={`rounded px-3 py-1.5 text-xs font-medium ${mode === "edit" ? "bg-white text-slate-950 shadow-sm" : "text-slate-500"}`}
          >
            Edit
          </button>
        </div>
      </div>

      {mode === "edit" ? (
        <div className="mt-4">
          <ReportSectionEditor report={report} comments={comments} />
        </div>
      ) : orderedSections.length ? (
        <div className="mt-4 grid gap-4 lg:grid-cols-[180px_minmax(0,1fr)]">
          <nav className="hidden self-start rounded-md border border-slate-200 bg-slate-50 p-2 text-xs lg:block">
            {orderedSections.map((section) => (
              <a key={section.section_id} href={`#section-${section.section_id}`} className="block rounded px-2 py-1.5 text-slate-600 hover:bg-white hover:text-slate-950">
                {section.title}
              </a>
            ))}
          </nav>
          <div className="space-y-4">
          {orderedSections.map((section) => (
            <section id={`section-${section.section_id}`} key={section.section_id} className="rounded-md border border-slate-200 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-medium text-slate-950">{section.title}</h3>
                {section.section_type ? <span className="rounded-full bg-slate-100 px-2 py-1 text-[11px] font-medium text-slate-500">{section.section_type.replaceAll("_", " ")}</span> : null}
              </div>
              {section.section_type === "transcript" ? (
                <TranscriptPreview content={section.content || ""} />
              ) : section.section_type === "visual_analysis" && reportVisualArtifactSnapshots(section).length ? (
                <p className="mt-3 text-sm leading-6 text-slate-600">Selected report artifacts are rendered below.</p>
              ) : (
                <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-600">
                  {section.content || "No content yet."}
                </p>
              )}
              {section.section_type === "visual_analysis" ? <VisualArtifacts section={section} /> : null}
            </section>
          ))}
          </div>
        </div>
      ) : (
        <p className="mt-3 text-sm text-slate-500">No sections yet.</p>
      )}
    </div>
  );
}

function VisualArtifacts({ section }: { section: ShareableReport["sections"][number] }) {
  const snapshots = reportVisualArtifactSnapshots(section);
  if (!snapshots.length) return null;
  return (
    <div className="mt-4 grid gap-3">
      {snapshots.map((raw, index) => {
        const item = raw as Record<string, unknown>;
        const datasetIds = Array.isArray(item.dataset_ids) ? item.dataset_ids.map(String) : [];
        return (
          <article key={String(item.artifact_id || index)} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-wide text-blue-600">{String(item.chart_type || item.artifact_type || "artifact")}</div>
                <h4 className="mt-1 text-sm font-semibold text-slate-950">{String(item.title || "Selected visual")}</h4>
              </div>
              {datasetIds.length ? <span className="rounded-full bg-white px-2 py-1 text-[11px] text-slate-500">Dataset: {datasetIds.join(", ")}</span> : null}
            </div>
            {item.description ? <p className="mt-2 text-xs leading-5 text-slate-600">{String(item.description)}</p> : null}
            {item.artifact_type === "table" ? <SnapshotTablePreview item={item} /> : <SnapshotChartPreview item={item} />}
            {item.image_path ? <p className="mt-2 text-[11px] text-emerald-700">Chart image stored for PDF export.</p> : null}
          </article>
        );
      })}
    </div>
  );
}

function TranscriptPreview({ content }: { content: string }) {
  const items = reportTranscriptItems(content);
  if (!items.length) {
    return <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-600">{content || "No content yet."}</p>;
  }
  return (
    <div className="mt-3 space-y-3">
      {items.map((item) => (
        <article key={item.index} className="rounded-md border border-slate-200 bg-slate-50 p-3">
          <div className="mb-2 inline-flex h-6 min-w-6 items-center justify-center rounded-full bg-slate-900 px-2 text-xs font-semibold text-white">
            {item.index}
          </div>
          {item.question ? (
            <div className="text-sm leading-6">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Question</div>
              <p className="mt-1 whitespace-pre-wrap text-slate-800">{item.question}</p>
            </div>
          ) : null}
          {item.answer ? (
            <div className="mt-3 border-t border-slate-200 pt-3 text-sm leading-6">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-blue-600">Answer</div>
              <p className="mt-1 whitespace-pre-wrap text-slate-700">{item.answer}</p>
            </div>
          ) : null}
        </article>
      ))}
    </div>
  );
}

function SnapshotChartPreview({ item }: { item: Record<string, unknown> }) {
  const values = reportPreviewRows(item);
  const maxValue = Math.max(...values.map((row) => Math.abs(row.value)), 1);
  if (!values.length) return null;

  return (
    <div className="mt-3 rounded-md border border-slate-200 bg-white p-3" aria-label="Chart preview">
      <div className="space-y-2">
        {values.map((row, index) => (
          <div key={`${row.label}-${index}`} className="grid grid-cols-[minmax(72px,140px)_1fr_auto] items-center gap-2 text-[11px] text-slate-600">
            <span className="truncate" title={row.label}>{row.label}</span>
            <span className="h-2 overflow-hidden rounded-full bg-slate-100">
              <span className="block h-full rounded-full bg-blue-500" style={{ width: `${Math.max(4, (Math.abs(row.value) / maxValue) * 100)}%` }} />
            </span>
            <span className="font-medium tabular-nums text-slate-700">{formatPreviewValue(row.value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SnapshotTablePreview({ item }: { item: Record<string, unknown> }) {
  const rows = reportTableRows(item);
  const columns = reportTableColumns(rows);
  if (!rows.length || !columns.length) return null;

  return (
    <div className="mt-3 overflow-hidden rounded-md border border-slate-200 bg-white" aria-label="Table preview">
      <div className="max-h-80 overflow-auto">
        <table className="min-w-full text-left text-xs">
          <thead className="sticky top-0 bg-slate-100 text-[11px] uppercase tracking-wide text-slate-500">
            <tr>
              {columns.map((column) => (
                <th key={column} scope="col" className="border-b border-slate-200 px-3 py-2 font-semibold">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex} className={rowIndex % 2 ? "bg-slate-50/60" : "bg-white"}>
                {columns.map((column) => (
                  <td key={column} className="max-w-64 truncate px-3 py-2 text-slate-700" title={row[column]}>
                    {row[column]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatPreviewValue(value: number) {
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}
