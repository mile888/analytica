import type { Artifact, ShareableReport } from "./api";
import { normalizeChartArtifact } from "./charts";

export function isChartArtifact(artifact: Artifact): boolean {
  return (artifact.type || artifact.artifact_type) === "chart";
}

export function isTechnicalArtifact(artifact: Artifact): boolean {
  return ["python_code", "sql", "validation", "trace", "unknown"].includes(artifact.type || artifact.artifact_type || "unknown");
}

export function isMeaningfulVisualTable(artifact: Artifact): boolean {
  const type = artifact.type || artifact.artifact_type;
  const title = String(artifact.title || "").toLowerCase();
  return type === "table" && !/result preview|column summary|evidence review|run trace|validation/.test(title);
}

export function visualAnalysisArtifacts(artifacts: Artifact[]): { charts: Artifact[]; tables: Artifact[]; technical: Artifact[] } {
  const visible = artifacts.filter((artifact) => artifact.visibility !== "hidden");
  return {
    charts: uniqueChartArtifacts(visible.filter(isChartArtifact)),
    tables: visible.filter(isMeaningfulVisualTable),
    technical: visible.filter(isTechnicalArtifact)
  };
}

export function artifactBranchId(artifact: Artifact): string {
  const metadata = (artifact.metadata || {}) as Record<string, unknown>;
  const nested = typeof metadata.metadata === "object" && metadata.metadata !== null
    ? metadata.metadata as Record<string, unknown>
    : {};
  return String(metadata.branch_id || nested.branch_id || "");
}

export function prioritizeArtifactsForBranch(artifacts: Artifact[], branchId?: string | null): Artifact[] {
  if (!branchId) return artifacts;
  const selected = artifacts.filter((artifact) => artifactBranchId(artifact) === branchId);
  const rest = artifacts.filter((artifact) => artifactBranchId(artifact) !== branchId);
  return [...selected, ...rest];
}

export function artifactsForBranch(artifacts: Artifact[], branchId?: string | null): Artifact[] {
  if (!branchId) return [];
  return artifacts.filter((artifact) => artifactBranchId(artifact) === branchId);
}

export function latestChartOrReport(artifacts: Artifact[], reports: ShareableReport[]): { chart?: Artifact; report?: ShareableReport } {
  const chart = uniqueChartArtifacts(artifacts.filter((artifact) => artifact.visibility !== "hidden" && isChartArtifact(artifact))).at(-1);
  return chart ? { chart } : { report: reports[0] };
}

export function uniqueChartArtifacts(artifacts: Artifact[]): Artifact[] {
  const seen = new Set<string>();
  const unique: Artifact[] = [];
  for (const artifact of [...artifacts].reverse()) {
    const key = chartFingerprint(artifact);
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(artifact);
  }
  return unique.reverse();
}

function chartFingerprint(artifact: Artifact): string {
  const chart = normalizeChartArtifact(artifact);
  if (!chart) return String(artifact.artifact_id || artifact.title || "");
  const points = chart.points
    .slice(0, 24)
    .map((point) => `${point.label}:${Number(point.value).toFixed(4)}:${point.x ?? ""}:${point.y ?? ""}`)
    .join("|");
  return [chart.type, chart.title.toLowerCase(), chart.xLabel || "", chart.yLabel || "", points].join("::");
}
