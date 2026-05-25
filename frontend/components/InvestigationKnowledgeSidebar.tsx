"use client";

import { useState } from "react";
import type { ReactNode } from "react";
import type {
  Artifact,
  Finding,
  FinalReportSnapshot,
  ShareableReport
} from "@/lib/api";
import { ChartArtifactCard } from "@/components/ChartArtifactCard";
import { TableArtifactCard } from "@/components/TableArtifactCard";
import { CollapsibleSection } from "@/components/CollapsibleSection";
import { InvestigationReportActions } from "@/components/InvestigationReportActions";
import { Chip, EmptyState } from "@/components/ui";
import { formatFindingForUser, isHighConfidenceKeyFinding } from "@/lib/display";
import { visualAnalysisArtifacts } from "@/lib/visualArtifacts";

export function InvestigationKnowledgeSidebar({
  investigationId,
  linkedDataSourceIds,
  findings,
  artifacts,
  reports,
  snapshotsByReport
}: {
  investigationId: string;
  linkedDataSourceIds: string[];
  findings: Finding[];
  artifacts: Artifact[];
  reports: ShareableReport[];
  snapshotsByReport: Record<string, FinalReportSnapshot[]>;
}) {
  const [showAllFindings, setShowAllFindings] = useState(false);
  const visibleArtifacts = artifacts.filter((artifact) => artifact.visibility !== "hidden");
  const visualArtifacts = visualAnalysisArtifacts(visibleArtifacts);
  const visualCount = visualArtifacts.charts.length + visualArtifacts.tables.length;
  const uniqueFindings = dedupeFindings(findings.filter(isHighConfidenceKeyFinding)).slice(-5);
  const visibleFindings = showAllFindings ? uniqueFindings : uniqueFindings.slice(0, 3);

  return (
    <aside className="space-y-3 lg:pr-1">
      <CollapsibleSection title="Key findings" count={uniqueFindings.length}>
        {uniqueFindings.length ? (
          <div className="space-y-2">
            {visibleFindings.map((finding) => {
              const display = formatFindingForUser(finding);
              const summary = fullFindingSummary(finding, display.summary);
              return (
                <article
                  key={finding.finding_id}
                  className="rounded-2xl border border-slate-200 bg-white/80 p-3 shadow-sm dark:border-slate-800 dark:bg-slate-950/70"
                >
                  <h3 className="text-sm font-semibold leading-5 text-slate-950 dark:text-slate-50">
                    {findingTitle(finding, display.title, summary)}
                  </h3>
                  <p className="mt-2 whitespace-pre-line text-xs leading-5 text-slate-600 dark:text-slate-300">
                    {summary}
                  </p>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    <InfoChip title="How strongly this finding is supported by the available computed evidence.">
                      {display.confidenceLabel} confidence
                    </InfoChip>
                    {recordCountChip(summary) ? (
                      <InfoChip title="Number of dataset rows used for this specific finding or group.">
                        {recordCountChip(summary)}
                      </InfoChip>
                    ) : null}
                    {filterChip(summary) ? <Chip>{filterChip(summary)}</Chip> : null}
                  </div>
                  {warningText(display) ? (
                    <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50/70 px-3 py-2 text-xs text-amber-900 dark:border-amber-950 dark:bg-amber-950/20 dark:text-amber-100">
                      {warningText(display)}
                    </div>
                  ) : null}
                </article>
              );
            })}
            {uniqueFindings.length > 3 ? (
              <button
                type="button"
                onClick={() => setShowAllFindings((current) => !current)}
                className="text-xs font-semibold text-blue-600 hover:text-blue-700 dark:text-blue-300"
              >
                {showAllFindings ? "Show fewer findings" : `Show all ${uniqueFindings.length} findings`}
              </button>
            ) : null}
          </div>
        ) : (
          <EmptyState>No findings yet. Ask about metrics, relationships, joinability, or semantic overlap.</EmptyState>
        )}
      </CollapsibleSection>

      <CollapsibleSection title="Visual analysis" count={visualCount} defaultOpen={Boolean(visualCount)}>
        <ArtifactGroups
          artifacts={visibleArtifacts}
          investigationId={investigationId}
          linkedDataSourceIds={linkedDataSourceIds}
        />
      </CollapsibleSection>

      <CollapsibleSection title="Turn into report" count={reports.length} defaultOpen={false}>
        <InvestigationReportActions
          investigationId={investigationId}
          initialReports={reports}
          initialSnapshotsByReport={snapshotsByReport}
        />
      </CollapsibleSection>
    </aside>
  );
}

function recordCountChip(summary: string): string {
  const match = summary.match(/across\s+([\d,]+)\s+(records|rows)/i) || summary.match(/n\s*=\s*([\d,]+)/i);
  return match ? `${match[1].replace(/,+$/, "")} records` : "";
}

function filterChip(summary: string): string {
  const filter = summary.match(/`[^`]+`\s*=\s*`([^`]+)`/i) || summary.match(/\bin\s+`?([A-Z][A-Za-z ]+)`?,/);
  return filter ? filter[1] : "";
}

function warningText(display: ReturnType<typeof formatFindingForUser>): string {
  const text = `${display.summary} ${display.limitation} ${display.uncertaintyNote}`.toLowerCase();
  if (text.includes("skew")) return "Right-skewed distribution";
  if (text.includes("sparse") || text.includes("small sample")) return "Small sample";
  if (text.includes("outlier") || text.includes("extreme")) return "Outlier-sensitive";
  return "";
}

function ArtifactGroups({
  artifacts,
  investigationId,
  linkedDataSourceIds
}: {
  artifacts: Artifact[];
  investigationId: string;
  linkedDataSourceIds: string[];
}) {
  if (!artifacts.length) return <EmptyState>No charts yet. Analyze a chart, comparison, trend, or distribution.</EmptyState>;
  const { charts: chartArtifacts, tables: meaningfulTables } = visualAnalysisArtifacts(artifacts);

  return (
    <div className="space-y-3">
      {chartArtifacts.length ? (
        <div className="space-y-2">
          {chartArtifacts.slice(0, 4).map((artifact) => (
            <ChartArtifactCard
              key={artifact.artifact_id}
              artifact={artifact}
              investigationId={investigationId}
              linkedDataSourceIds={linkedDataSourceIds}
              compact
            />
          ))}
        </div>
      ) : (
        <EmptyState>No charts yet. Analyze a chart, comparison, trend, or distribution.</EmptyState>
      )}

      {meaningfulTables.length ? (
        <div className="space-y-2">
          {meaningfulTables.slice(0, 2).map((artifact) => (
            <TableArtifactCard key={artifact.artifact_id} artifact={artifact} investigationId={investigationId} compact />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function fullFindingSummary(finding: Finding, fallback: string): string {
  const metadata = finding.metadata || {};
  const conclusion = typeof metadata.conclusion === "string" ? metadata.conclusion.trim() : "";
  return cleanDisplayText(conclusion || finding.text || fallback);
}

function cleanDisplayText(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function findingTitle(finding: Finding, fallbackTitle: string, summary: string): string {
  const metadata = finding.metadata || {};
  const metric = typeof metadata.metric === "string" ? metadata.metric : "";
  const dimension = typeof metadata.dimension === "string" ? metadata.dimension : "";
  const text = cleanDisplayText(summary || fallbackTitle).replace(/`([^`]+)` is treated as `([^`]+)`\.\s*/i, "");
  const ledBy = text.match(/`([^`]+)`\s+by\s+`([^`]+)`\s+is led by\s+`([^`]+)`/i);
  if (ledBy) return `${ledBy[3]} leads ${ledBy[1]} by ${ledBy[2]}`;
  const dist = text.match(/`([^`]+)`\s+distribution(?:\s+in\s+([^.`]+))?/i);
  if (dist) return dist[2] ? `${dist[1]} distribution in ${dist[2]}` : `${dist[1]} distribution`;
  const comparison = text.match(/`([^`]+)`\s+.*compared against\s+`([^`]+)`/i);
  if (comparison) return `${comparison[1]} comparison vs ${comparison[2]}`;
  if (metric && dimension) return `${metric} by ${dimension}`;
  return cleanDisplayText(fallbackTitle).replace(/`([^`]+)` is treated as `([^`]+)`\.\s*/i, "");
}

function InfoChip({ children, title }: { children: ReactNode; title: string }) {
  return (
    <span
      title={title}
      className="inline-flex items-center rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600 dark:bg-slate-900 dark:text-slate-300"
    >
      {children}
    </span>
  );
}

function dedupeFindings(findings: Finding[]): Finding[] {
  const seen = new Set<string>();
  const deduped: Finding[] = [];
  for (const finding of findings) {
    const display = formatFindingForUser(finding);
    const key = `${display.title} ${display.summary}`.toLowerCase().replace(/\s+/g, " ").trim();
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(finding);
  }
  return deduped;
}
