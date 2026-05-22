"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import {
  Artifact,
  DataSource,
  EvidenceLinkPayload,
  Finding,
  InvestigationMemoryItem,
  ShareableReport,
  linkFindingEvidence,
  promoteFindingToMemory
} from "@/lib/api";
import { Button } from "@/components/ui";
import { EvidenceInspector } from "@/components/EvidenceInspector";
import { InvestigationDataContextItem } from "@/components/InvestigationDataContext";
import { EvidenceReference, normalizeEvidenceReferences } from "@/lib/evidence";

type LinkableEvidenceReference = EvidenceReference & {
  type: "artifact" | "data_source" | "memory_item" | "report_section";
};

export function FindingEvidenceActions({
  investigationId,
  finding,
  artifacts,
  dataSources,
  dataSourceContexts = [],
  memoryItems,
  reports = []
}: {
  investigationId: string;
  finding: Finding;
  artifacts: Artifact[];
  dataSources: DataSource[];
  dataSourceContexts?: InvestigationDataContextItem[];
  memoryItems: InvestigationMemoryItem[];
  reports?: ShareableReport[];
}) {
  const router = useRouter();
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const evidenceOptions = useMemo(() => {
    const options: LinkableEvidenceReference[] = [];
    for (const artifact of artifacts) {
      options.push({
        key: `artifact:${artifact.artifact_id}`,
        type: "artifact",
        id: artifact.artifact_id,
        label: artifact.title || artifact.artifact_id,
        metadata: {}
      });
    }
    for (const source of dataSources) {
      options.push({
        key: `data_source:${source.data_source_id}`,
        type: "data_source",
        id: source.data_source_id,
        label: source.name,
        href: `/data-sources/${source.data_source_id}`,
        metadata: {}
      });
    }
    for (const item of memoryItems.filter((candidate) => candidate.status !== "archived")) {
      options.push({
        key: `memory_item:${item.memory_id}`,
        type: "memory_item",
        id: item.memory_id,
        label: item.title || item.content.slice(0, 48) || item.memory_id,
        metadata: {}
      });
    }
    for (const report of reports) {
      for (const section of report.sections || []) {
        options.push({
          key: `report_section:${section.section_id}`,
          type: "report_section",
          id: section.section_id,
          label: `${report.title}: ${section.title}`,
          href: `/reports/${report.report_id}#section-${section.section_id}`,
          metadata: { report_id: report.report_id }
        });
      }
    }
    return options;
  }, [artifacts, dataSources, memoryItems, reports]);

  const linkedEvidence = normalizeEvidenceReferences(finding.metadata?.linked_evidence);
  const [selectedEvidenceKey, setSelectedEvidenceKey] = useState(evidenceOptions[0]?.key || "");
  const [inspectedEvidence, setInspectedEvidence] = useState<EvidenceReference | null>(null);

  async function promote(type: "assumption" | "risk" | "decision") {
    setBusy(`promote-${type}`);
    setError(null);
    setMessage(null);
    try {
      await promoteFindingToMemory(investigationId, finding.finding_id, type);
      setMessage("Saved.");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not promote finding");
    } finally {
      setBusy(null);
    }
  }

  async function linkEvidence() {
    const selected = evidenceOptions.find((option) => option.key === selectedEvidenceKey);
    if (!selected) return;
    setBusy("link-evidence");
    setError(null);
    setMessage(null);
    try {
      await linkFindingEvidence(investigationId, finding.finding_id, selected);
      setMessage("Evidence attached.");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not link evidence");
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="mt-3 space-y-3 border-t border-slate-200 pt-3 dark:border-slate-800">
      <div>
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">Save finding as</div>
        <div className="mt-2 flex flex-wrap gap-2">
          {(["assumption", "risk", "decision"] as const).map((type) => (
            <Button
              key={type}
              type="button"
              onClick={() => promote(type)}
              disabled={busy === `promote-${type}`}
              variant="secondary"
              size="sm"
            >
              {busy === `promote-${type}` ? "Saving..." : memoryActionLabel(type)}
            </Button>
          ))}
        </div>
      </div>

      <div>
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">Evidence</div>
        {linkedEvidence.length ? (
          <div className="mt-2 flex flex-wrap gap-2">
            {linkedEvidence.map((item) =>
              (
                <button
                  key={item.key}
                  type="button"
                  onClick={() => setInspectedEvidence(item)}
                  className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700 transition hover:bg-blue-100 hover:text-blue-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-blue-950/50 dark:hover:text-blue-300"
                >
                  {item.label}
                </button>
              )
            )}
          </div>
        ) : (
          <p className="mt-1 text-xs text-slate-500">No evidence attached yet.</p>
        )}

        {evidenceOptions.length ? (
          <div className="mt-2 flex flex-wrap gap-2">
            <select
              value={selectedEvidenceKey}
              onChange={(event) => setSelectedEvidenceKey(event.target.value)}
              className="min-w-0 flex-1 rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700 outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-500/10 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200"
            >
              {evidenceOptions.map((option) => (
                <option key={option.key} value={option.key}>
                  {option.type.replace("_", " ")} · {option.label}
                </option>
              ))}
            </select>
            <Button
              type="button"
              onClick={linkEvidence}
              disabled={!selectedEvidenceKey || busy === "link-evidence"}
              size="sm"
            >
              {busy === "link-evidence" ? "Attaching..." : "Attach"}
            </Button>
          </div>
        ) : null}
      </div>

      {message ? <p className="text-xs text-emerald-600">{message}</p> : null}
      {error ? <p className="text-xs text-red-600">{error}</p> : null}
      <EvidenceInspector
        evidence={inspectedEvidence}
        artifacts={artifacts}
        dataSourceContexts={dataSourceContexts.length ? dataSourceContexts : dataSources.map((source) => ({ source }))}
        memoryItems={memoryItems}
        reports={reports}
        onClose={() => setInspectedEvidence(null)}
      />
    </div>
  );
}

function memoryActionLabel(type: "assumption" | "risk" | "decision") {
  const labels = {
    assumption: "Working belief",
    risk: "Risk",
    decision: "Conclusion"
  };
  return labels[type];
}
