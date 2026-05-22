"use client";

import type { Artifact } from "@/lib/api";

export function TableArtifactCard({ artifact, compact = false }: { artifact: Artifact; compact?: boolean }) {
  const rows = tableRows(artifact);
  if (!rows.length) return null;
  const columns = tableColumns(rows).slice(0, compact ? 5 : 8);
  const visibleRows = rows.slice(0, compact ? 4 : 5);

  return (
    <article className={`rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-950 ${compact ? "p-3" : "p-4"}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-300">Table</div>
          <h3 className="mt-1 line-clamp-2 text-sm font-semibold text-slate-950 dark:text-slate-50">{artifact.title || "Table preview"}</h3>
        </div>
        <span className="shrink-0 rounded-full bg-slate-100 px-2 py-1 text-[11px] font-medium text-slate-500 dark:bg-slate-900 dark:text-slate-400">
          {rows.length} rows
        </span>
      </div>
      <div className="mt-3 overflow-hidden rounded-xl border border-slate-200 dark:border-slate-800">
        <div className="overflow-x-auto">
          <table className="w-full min-w-max text-left text-xs">
            <thead className="bg-slate-50 text-[11px] uppercase tracking-wide text-slate-500 dark:bg-slate-900 dark:text-slate-400">
              <tr>
                {columns.map((column) => (
                  <th key={column} className="px-3 py-2 font-semibold">{column}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {visibleRows.map((row, rowIndex) => (
                <tr key={rowIndex} className="bg-white dark:bg-slate-950">
                  {columns.map((column) => (
                    <td key={column} className="max-w-48 truncate px-3 py-2 text-slate-700 dark:text-slate-200" title={formatCell(row[column])}>
                      {formatCell(row[column])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </article>
  );
}

function tableRows(artifact: Artifact): Array<Record<string, unknown>> {
  const content = artifact.content;
  if (Array.isArray(content)) {
    return content.filter((row): row is Record<string, unknown> => typeof row === "object" && row !== null && !Array.isArray(row));
  }
  if (typeof content === "object" && content !== null && !Array.isArray(content)) {
    const nested = content as Record<string, unknown>;
    const rows = nested.rows || nested.content || nested.data;
    if (Array.isArray(rows)) {
      return rows.filter((row): row is Record<string, unknown> => typeof row === "object" && row !== null && !Array.isArray(row));
    }
  }
  return [];
}

function tableColumns(rows: Array<Record<string, unknown>>): string[] {
  const seen = new Set<string>();
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      seen.add(key);
    }
  }
  return Array.from(seen);
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}
