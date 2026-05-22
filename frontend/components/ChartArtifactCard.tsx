"use client";

import { useRouter } from "next/navigation";
import { Fragment } from "react";
import { useState } from "react";
import type { Artifact } from "@/lib/api";
import { activateInvestigationBranch, createInvestigationMessage, runInvestigation, selectArtifactForReport } from "@/lib/api";
import { chartSummary, chartTypeLabel, formatNumber, normalizeChartArtifact, type ChartPreviewData } from "@/lib/charts";
import { useToast } from "@/components/ToastProvider";

export function ChartArtifactCard({
  artifact,
  investigationId,
  linkedDataSourceIds = [],
  compact = false
}: {
  artifact: Artifact;
  investigationId: string;
  linkedDataSourceIds?: string[];
  compact?: boolean;
}) {
  const chart = normalizeChartArtifact(artifact);
  const router = useRouter();
  const { showToast } = useToast();
  const [busy, setBusy] = useState<"explain" | "report" | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [selectedForReport, setSelectedForReport] = useState(Boolean(artifact.metadata?.selected_for_report || artifact.metadata?.use_in_report));

  if (!chart) {
    const artifactType = artifact.type || artifact.artifact_type;
    if (artifactType !== "chart") return null;
    return (
      <article className={`rounded-2xl border border-amber-200 bg-amber-50 text-amber-900 shadow-sm dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-100 ${compact ? "p-3" : "p-4"}`}>
        <div className="text-[11px] font-semibold uppercase tracking-wide">Artifact validation error</div>
        <h3 className="mt-1 text-sm font-semibold">{artifact.title || "Chart artifact"}</h3>
        <p className="mt-2 text-xs leading-5">
          This chart payload could not be rendered safely because its bins, counts, row totals, or chart metadata are inconsistent.
        </p>
      </article>
    );
  }

  async function explainChart() {
    setBusy("explain");
    try {
      const branchId = artifactBranchId(artifact);
      if (branchId) {
        await activateInvestigationBranch(investigationId, branchId);
      }
      const message = await createInvestigationMessage(investigationId, {
        role: "user",
        type: "follow_up",
        content: `Explain this chart.`,
        metadata: {
          action: "explain_artifact",
          artifact_id: artifact.artifact_id,
          branch_id: branchId
        }
      });
      await runInvestigation(investigationId, {
        message_id: message.message_id,
        data_source_ids: linkedDataSourceIds
      });
      showToast("Chart explanation started", "success");
      router.refresh();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Could not explain chart", "error");
    } finally {
      setBusy(null);
    }
  }

  async function useInReport() {
    setBusy("report");
    try {
      await selectArtifactForReport(investigationId, artifact.artifact_id, true);
      setSelectedForReport(true);
      showToast("Chart added to the next report update", "success");
      router.refresh();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Could not add chart to report", "error");
    } finally {
      setBusy(null);
    }
  }

  return (
    <article className={`rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-950 ${compact ? "p-3" : "p-4"}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-blue-600 dark:text-blue-300">
            {chartTypeLabel(chart.type)}
          </div>
          <h3 className="mt-1 line-clamp-2 text-sm font-semibold text-slate-950 dark:text-slate-50">{chart.title}</h3>
          {!compact ? <p className="mt-1 text-xs leading-5 text-slate-500 dark:text-slate-400">{chartSummary(chart)}</p> : null}
        </div>
      </div>

      <div className="mt-3">
        <ChartPreview chart={chart} compact={compact} />
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={explainChart}
          disabled={busy !== null}
          className="rounded-full border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-50 dark:border-slate-800 dark:text-slate-200 dark:hover:border-blue-900 dark:hover:bg-blue-950/40"
        >
          {busy === "explain" ? "Starting..." : "Explain this chart"}
        </button>
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="rounded-full border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 dark:border-slate-800 dark:text-slate-200 dark:hover:border-blue-900 dark:hover:bg-blue-950/40"
        >
          Expand
        </button>
        <button
          type="button"
          onClick={useInReport}
          disabled={busy !== null}
          className="rounded-full bg-slate-950 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-slate-800 disabled:opacity-50 dark:bg-slate-100 dark:text-slate-950 dark:hover:bg-white"
        >
          {busy === "report" ? "Adding..." : selectedForReport ? "In report" : "Use in report"}
        </button>
      </div>

      {expanded ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 p-4 backdrop-blur-sm">
          <div className="max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-3xl border border-slate-200 bg-white p-5 shadow-2xl dark:border-slate-800 dark:bg-slate-950">
            <div className="mb-4 flex items-start justify-between gap-4">
              <div>
                <div className="text-xs font-semibold uppercase tracking-wide text-blue-600 dark:text-blue-300">Chart preview</div>
                <h2 className="mt-1 text-lg font-semibold text-slate-950 dark:text-slate-50">{chart.title}</h2>
                <p className="mt-1 text-sm leading-6 text-slate-500 dark:text-slate-400">{chartSummary(chart)}</p>
              </div>
              <button
                type="button"
                onClick={() => setExpanded(false)}
                className="rounded-full border border-slate-200 px-3 py-1.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 dark:border-slate-800 dark:text-slate-200 dark:hover:bg-slate-900"
              >
                Close
              </button>
            </div>
            <ChartPreview chart={chart} />
          </div>
        </div>
      ) : null}
    </article>
  );
}

function artifactBranchId(artifact: Artifact): string {
  const metadata = artifact.metadata || {};
  const nested = typeof metadata.metadata === "object" && metadata.metadata !== null && !Array.isArray(metadata.metadata)
    ? metadata.metadata as Record<string, unknown>
    : {};
  const branchId = String(metadata.branch_id || nested.branch_id || "");
  return branchId === "global" || branchId === "unknown" ? "" : branchId;
}

function ChartPreview({ chart, compact = false }: { chart: ChartPreviewData; compact?: boolean }) {
  if (chart.type === "line") return <LineChart chart={chart} compact={compact} />;
  if (chart.type === "scatter") return <ScatterChart chart={chart} compact={compact} />;
  if (chart.type === "heatmap") return <HeatmapChart chart={chart} compact={compact} />;
  if (chart.type === "histogram") return <HistogramChart chart={chart} compact={compact} />;
  return <BarLikeChart chart={chart} compact={compact} />;
}

function HistogramChart({ chart, compact }: { chart: ChartPreviewData; compact?: boolean }) {
  const maxValue = Math.max(...chart.points.map((point) => Math.max(point.value, 0)), 1);
  const height = compact ? 96 : 150;
  return (
    <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/70">
      <div className="flex items-end gap-0.5 border-b border-slate-300/70 pb-1 dark:border-slate-700" style={{ height }}>
        {chart.points.map((point, index) => (
          <div key={`${point.label}-${index}`} className="flex min-w-0 flex-1 flex-col items-center justify-end">
            <div
              title={`${point.label}: ${formatNumber(point.value)}`}
              className="w-full rounded-t bg-blue-500 dark:bg-blue-400"
              style={{ height: `${Math.max(3, (Math.max(point.value, 0) / maxValue) * (height - 20))}px` }}
            />
          </div>
        ))}
      </div>
      <div className="mt-2 flex justify-between gap-3 text-[10px] text-slate-500 dark:text-slate-400">
        <span className="truncate">{chart.points[0]?.label}</span>
        <span className="truncate text-right">{chart.points[chart.points.length - 1]?.label}</span>
      </div>
      {!compact ? <div className="mt-1 text-[10px] font-medium uppercase tracking-wide text-slate-400">Ordered bins · record count</div> : null}
    </div>
  );
}

function HeatmapChart({ chart, compact }: { chart: ChartPreviewData; compact?: boolean }) {
  const rows = Array.from(new Set(chart.points.map((point) => point.row || ""))).filter(Boolean);
  const columns = Array.from(new Set(chart.points.map((point) => point.column || ""))).filter(Boolean);
  const maxValue = Math.max(...chart.points.map((point) => point.value), 1);
  const valueFor = (row: string, column: string) => chart.points.find((point) => point.row === row && point.column === column)?.value ?? 0;
  return (
    <div className="overflow-x-auto rounded-xl bg-slate-50 p-3 dark:bg-slate-900/70">
      <div
        className="grid gap-1 text-[10px]"
        style={{ gridTemplateColumns: `minmax(42px, 0.7fr) repeat(${columns.length}, minmax(${compact ? 20 : 28}px, 1fr))` }}
      >
        <div />
        {columns.map((column) => (
          <div key={column} className="truncate text-center font-semibold text-slate-500 dark:text-slate-400">{column}</div>
        ))}
        {rows.map((row) => (
          <Fragment key={row}>
            <div key={`${row}-label`} className="truncate pr-1 text-right font-semibold text-slate-500 dark:text-slate-400">{row}</div>
            {columns.map((column) => {
              const value = valueFor(row, column);
              const opacity = Math.max(0.12, Math.min(1, value / maxValue));
              return (
                <div
                  key={`${row}-${column}`}
                  title={`${row} / ${column}: ${formatNumber(value)}`}
                  className="grid aspect-square place-items-center rounded text-[9px] font-semibold text-blue-950 dark:text-blue-50"
                  style={{ backgroundColor: `rgba(59, 130, 246, ${opacity})` }}
                >
                  {compact ? "" : formatNumber(value)}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function BarLikeChart({ chart, compact }: { chart: ChartPreviewData; compact?: boolean }) {
  const maxValue = Math.max(...chart.points.map((point) => Math.max(point.value, 0)), 1);
  return (
    <div className={compact ? "space-y-1.5" : "space-y-2"}>
      {chart.points.slice(0, compact ? 6 : 10).map((point) => (
        <div key={`${point.label}-${point.value}`} className="grid grid-cols-[minmax(70px,0.8fr)_1fr_auto] items-center gap-2 text-xs">
          <div className="truncate text-slate-500 dark:text-slate-400" title={point.label}>{point.label || "Value"}</div>
          <div className="h-2 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
            <div
              className="h-full rounded-full bg-blue-500 dark:bg-blue-400"
              style={{ width: `${Math.max(3, (Math.max(point.value, 0) / maxValue) * 100)}%` }}
            />
          </div>
          <div className="text-right font-medium text-slate-700 dark:text-slate-200">{formatNumber(point.value)}</div>
        </div>
      ))}
    </div>
  );
}

function LineChart({ chart, compact }: { chart: ChartPreviewData; compact?: boolean }) {
  const width = 320;
  const height = compact ? 90 : 130;
  const values = chart.points.map((point) => point.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = chart.points.map((point, index) => {
    const x = chart.points.length === 1 ? width / 2 : (index / (chart.points.length - 1)) * width;
    const y = height - ((point.value - min) / span) * (height - 16) - 8;
    return `${x},${y}`;
  });
  return (
    <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/70">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-28 w-full overflow-visible">
        <polyline fill="none" stroke="currentColor" strokeWidth="3" points={points.join(" ")} className="text-blue-500 dark:text-blue-300" />
        {chart.points.map((point, index) => {
          const [x, y] = points[index].split(",").map(Number);
          return <circle key={`${point.label}-${index}`} cx={x} cy={y} r="3.5" className="fill-blue-500 dark:fill-blue-300" />;
        })}
      </svg>
      <div className="mt-1 flex justify-between text-[11px] text-slate-500 dark:text-slate-400">
        <span>{chart.points[0]?.label}</span>
        <span>{chart.points[chart.points.length - 1]?.label}</span>
      </div>
    </div>
  );
}

function ScatterChart({ chart, compact }: { chart: ChartPreviewData; compact?: boolean }) {
  const width = 320;
  const height = compact ? 90 : 130;
  const xs = chart.points.map((point) => point.x ?? 0);
  const ys = chart.points.map((point) => point.y ?? point.value);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  return (
    <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-900/70">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-28 w-full">
        {chart.points.map((point, index) => {
          const x = (((point.x ?? 0) - minX) / spanX) * (width - 20) + 10;
          const y = height - (((point.y ?? point.value) - minY) / spanY) * (height - 20) - 10;
          return <circle key={`${point.label}-${index}`} cx={x} cy={y} r="3" className="fill-blue-500/80 dark:fill-blue-300/80" />;
        })}
      </svg>
    </div>
  );
}
