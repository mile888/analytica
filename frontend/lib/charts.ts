import type { Artifact } from "@/lib/api";

export type ChartType = "bar" | "line" | "histogram" | "scatter" | "heatmap";

export interface ChartPoint {
  label: string;
  value: number;
  x?: number;
  y?: number;
  row?: string;
  column?: string;
  left?: number;
  right?: number;
}

export interface ChartPreviewData {
  type: ChartType;
  title: string;
  xLabel?: string;
  yLabel?: string;
  points: ChartPoint[];
  rowCount?: number;
  comparisonGroups?: number;
}

export function normalizeChartArtifact(artifact: Artifact): ChartPreviewData | null {
  const artifactType = artifact.type || artifact.artifact_type;
  if (artifactType !== "chart") return null;

  const content = asRecord(artifact.content);
  if (!content) return null;

  const rawType = String(content.chart_type || content.type || "").toLowerCase();
  const type: ChartType = rawType.includes("line")
    ? "line"
    : rawType.includes("heatmap")
      ? "heatmap"
    : rawType.includes("hist")
      ? "histogram"
      : rawType.includes("scatter")
        ? "scatter"
        : "bar";

  const rows = Array.isArray(content.rows) ? content.rows.filter(isRecord) : [];
  const xKey = stringValue(content.x) || stringValue(content.x_field) || "label";
  const yKey = stringValue(content.y) || stringValue(content.y_field) || "value";
  const title = artifact.title || chartTitle(type);

  if (type === "histogram") {
    const declaredRowCount = numberValue(content.row_count);
    const comparisonGroups = Array.isArray(content.comparison_groups) ? content.comparison_groups.filter(isRecord) : [];
    if (comparisonGroups.length) {
      const groupPoints = comparisonGroups.flatMap((group) => {
        const groupLabel = stringValue(group.label) || stringValue(group.group) || "";
        const groupRowCount = numberValue(group.row_count) ?? numberValue(group.record_count) ?? numberValue(group.n);
        const bins = Array.isArray(group.bins) ? group.bins.filter(isRecord) : [];
        if (!groupLabel || !bins.length || groupRowCount === null || groupRowCount <= 0) return [];
        const points = bins
          .map((bin) => ({
            label: `${groupLabel}: ${stringValue(bin.label) || stringValue(bin.bin)}`,
            value: numberValue(bin.count) ?? 0,
            left: numberValue(bin.left) ?? undefined,
            right: numberValue(bin.right) ?? undefined,
          }))
          .filter((point) => point.label && Number.isFinite(point.value));
        const binTotal = points.reduce((sum, point) => sum + point.value, 0);
        return binTotal === groupRowCount ? points : [];
      });
      const computedRowCount = groupPoints.reduce((sum, point) => sum + point.value, 0);
      if (declaredRowCount !== null && declaredRowCount !== computedRowCount) return null;
      const rowCount = declaredRowCount ?? computedRowCount;
      return groupPoints.length && rowCount > 0 && groupPoints.some((point) => point.value > 0)
        ? { type, title, xLabel: stringValue(content.series) || xKey, yLabel: "record count", points: groupPoints.slice(0, 24), rowCount, comparisonGroups: comparisonGroups.length }
        : null;
    }
    const binCounts = Array.isArray(content.bin_counts) ? content.bin_counts.map(numberValue) : [];
    const binEdges = Array.isArray(content.bin_edges) ? content.bin_edges : [];
    if (binCounts.some((value) => value !== null) && binEdges.length === binCounts.length + 1) {
      const points = binCounts
        .map((value, index) => {
          const left = numberValue(binEdges[index]);
          const right = numberValue(binEdges[index + 1]);
          const label = left !== null && right !== null ? `${formatNumber(left)}-${formatNumber(right)}` : `Bin ${index + 1}`;
          return { label, value: value ?? 0, left: left ?? undefined, right: right ?? undefined };
        })
        .filter((point) => point.label || Number.isFinite(point.value));
      const rowCount = declaredRowCount ?? points.reduce((sum, point) => sum + point.value, 0);
      const binTotal = points.reduce((sum, point) => sum + point.value, 0);
      return points.length && rowCount > 0 && binTotal === rowCount && points.some((point) => point.value > 0)
        ? { type, title, xLabel: xKey, yLabel: "count", points: points.slice(0, 24), rowCount }
        : null;
    }
    const histogramRows = (Array.isArray(content.bins) ? content.bins.filter(isRecord) : rows);
    const points = histogramRows
      .map((row) => ({
        label: stringValue(row.label) || stringValue(row.bin) || stringValue(row[xKey]) || "",
        value: numberValue(row.count) ?? numberValue(row.value) ?? numberValue(row[yKey]) ?? 0,
        left: numberValue(row.left) ?? undefined,
        right: numberValue(row.right) ?? undefined,
      }))
      .filter((point) => point.label || Number.isFinite(point.value));
    const rowCount = declaredRowCount ?? points.reduce((sum, point) => sum + point.value, 0);
    const binTotal = points.reduce((sum, point) => sum + point.value, 0);
    return points.length && rowCount > 0 && binTotal === rowCount && points.some((point) => point.value > 0)
      ? { type, title, xLabel: xKey, yLabel: "count", points: points.slice(0, 24), rowCount }
      : null;
  }

  if (type === "scatter") {
    const xMetric = stringValue(content.x_metric) || stringValue(artifact.metadata?.x_metric) || xKey;
    const yMetric = stringValue(content.y_metric) || stringValue(artifact.metadata?.y_metric) || yKey;
    const labelKey = stringValue(content.label) || stringValue(content.label_field) || stringValue(content.grouping) || "label";
    const scatterRows = Array.isArray(content.points) ? content.points.filter(isRecord) : rows;
    const points: ChartPoint[] = [];
    scatterRows.forEach((row, index) => {
      const x = numberValue(row.x) ?? numberValue(row[xKey]) ?? numberValue(row[xMetric]);
      const y = numberValue(row.y) ?? numberValue(row[yKey]) ?? numberValue(row[yMetric]);
      if (x === null || y === null) return;
      const label = stringValue(row.label) || stringValue(row[labelKey]) || stringValue(row.group) || `Point ${index + 1}`;
      points.push({ label, value: y, x, y });
    });
    return points.length ? { type, title, xLabel: xMetric, yLabel: yMetric, points: points.slice(0, 80) } : null;
  }

  if (type === "heatmap") {
    const heatRows = rows.length ? rows :
      (Array.isArray(content.points) ? (content.points as unknown[]).filter(isRecord) : []);
    const valueKey = stringValue(content.value) || yKey || "value";
    const points = heatRows
      .map((row) => ({
        label: `${stringValue(row[yKey]) || stringValue(row.year) || stringValue(row.row)}-${stringValue(row[xKey]) || stringValue(row.month) || stringValue(row.column)}`,
        value: numberValue(row[valueKey]) ?? numberValue(row.mean) ?? numberValue(row.value) ?? 0,
        row: stringValue(row[yKey]) || stringValue(row.year) || stringValue(row.row),
        column: stringValue(row[xKey]) || stringValue(row.month) || stringValue(row.column),
      }))
      .filter((point) => point.row && point.column);
    return points.length ? { type, title, xLabel: xKey, yLabel: yKey, points: points.slice(0, 120) } : null;
  }

  const points = rows
    .map((row) => ({
      label: stringValue(row[xKey]) || stringValue(row.period) || stringValue(row.label) || "",
      value: numberValue(row[yKey]) ?? numberValue(row.mean) ?? numberValue(row.value) ?? numberValue(row.count) ?? 0
    }))
    .filter((point) => point.label || Number.isFinite(point.value));

  return points.length ? { type, title, xLabel: xKey, yLabel: yKey, points: points.slice(0, 24) } : null;
}

export function chartSummary(data: ChartPreviewData): string {
  if (!data.points.length) return "Chart preview is not available yet.";
  const sorted = [...data.points].sort((a, b) => b.value - a.value);
  const top = sorted[0];
  const bottom = sorted[sorted.length - 1];
  if (data.type === "line") {
    const first = data.points[0];
    const last = data.points[data.points.length - 1];
    const direction = last.value > first.value ? "upward" : last.value < first.value ? "downward" : "flat";
    return `Line chart with a ${direction} movement from ${formatNumber(first.value)} to ${formatNumber(last.value)}.`;
  }
  if (data.type === "scatter") {
    return `Scatter plot with ${data.points.length} plotted observations.`;
  }
  if (data.type === "heatmap") {
    return `Heatmap with ${data.points.length} cells; strongest cell is ${top.label} (${formatNumber(top.value)}).`;
  }
  if (data.type === "histogram") {
    const total = data.rowCount ?? data.points.reduce((sum, point) => sum + point.value, 0);
    if (data.comparisonGroups) {
      return `Histogram comparison with ${data.comparisonGroups} groups and ${formatNumber(total)} records.`;
    }
    return `Histogram with ${data.points.length} ordered bins and ${formatNumber(total)} records.`;
  }
  return `${top.label} is highest (${formatNumber(top.value)}); ${bottom.label} is lowest (${formatNumber(bottom.value)}).`;
}

export function chartTypeLabel(type: ChartType): string {
  const labels: Record<ChartType, string> = {
    bar: "Bar chart",
    line: "Trend chart",
    histogram: "Histogram",
    scatter: "Scatter plot",
    heatmap: "Heatmap",
  };
  return labels[type];
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
}

function chartTitle(type: ChartType): string {
  const labels: Record<ChartType, string> = {
    bar: "Bar chart",
    line: "Line chart",
    histogram: "Distribution chart",
    scatter: "Scatter plot",
    heatmap: "Heatmap"
  };
  return labels[type];
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function stringValue(value: unknown): string {
  return typeof value === "string" || typeof value === "number" ? String(value) : "";
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}
