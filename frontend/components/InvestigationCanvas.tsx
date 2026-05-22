import {
  Artifact,
  DataSource,
  FinalReportSnapshot,
  Finding,
  Investigation,
  InvestigationMemoryItem,
  InvestigationRun,
  ShareableReport
} from "@/lib/api";
import { FindingEvidenceActions } from "@/components/FindingEvidenceActions";
import { InvestigationDataContextItem } from "@/components/InvestigationDataContext";
import { InvestigationReportActions } from "@/components/InvestigationReportActions";
import { TechnicalDetails } from "@/components/TechnicalDetails";
import { Card, Chip, EmptyState, Panel, SectionHeader, StatusBadge } from "@/components/ui";
import { formatArtifactForUser, formatFindingForUser, looksTechnical } from "@/lib/display";

export function InvestigationCanvas({
  investigation,
  latestRun,
  reports,
  snapshotsByReport,
  dataSources = [],
  dataSourceContexts = [],
  memoryItems = []
}: {
  investigation: Investigation;
  latestRun?: InvestigationRun;
  reports: ShareableReport[];
  snapshotsByReport: Record<string, FinalReportSnapshot[]>;
  dataSources?: DataSource[];
  dataSourceContexts?: InvestigationDataContextItem[];
  memoryItems?: InvestigationMemoryItem[];
}) {
  const findings = investigation.findings || [];
  const artifacts = investigation.artifacts || [];
  const visibleArtifacts = artifacts.filter((artifact) => artifact.visibility !== "hidden");

  return (
    <div className="space-y-6">
      <Card className="p-5 lg:p-6">
        <SectionHeader
          eyebrow="Workspace"
          title="Investigation canvas"
          description="Review analytical outputs, connect evidence, and move the work toward a shareable report."
          action={
            <div className="grid grid-cols-3 gap-2 text-center">
              <CanvasMetric label="Findings" value={findings.length} />
              <CanvasMetric label="Artifacts" value={visibleArtifacts.length} />
              <CanvasMetric label="Runs" value={investigation.runs?.length || (latestRun ? 1 : 0)} />
            </div>
          }
        />
      </Card>

      <FindingsSection
        investigationId={investigation.investigation_id}
        findings={findings}
        artifacts={visibleArtifacts}
        dataSources={dataSources}
        dataSourceContexts={dataSourceContexts}
        memoryItems={memoryItems}
        reports={reports}
      />
      <ArtifactsSection artifacts={visibleArtifacts} />
      <ReportSummary artifacts={artifacts} findings={findings} latestRun={latestRun} />
      <InvestigationReportActions
        investigationId={investigation.investigation_id}
        initialReports={reports}
        initialSnapshotsByReport={snapshotsByReport}
      />
    </div>
  );
}

function CanvasMetric({ label, value }: { label: string; value: number }) {
  return (
    <div className="min-w-20 rounded-xl border border-slate-200 bg-slate-50/80 px-3 py-2 dark:border-slate-800 dark:bg-slate-900/70">
      <div className="text-lg font-semibold text-slate-950 dark:text-slate-50">{value}</div>
      <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">{label}</div>
    </div>
  );
}

function FindingsSection({
  investigationId,
  findings,
  artifacts,
  dataSources,
  dataSourceContexts,
  memoryItems,
  reports
}: {
  investigationId: string;
  findings: Finding[];
  artifacts: Artifact[];
  dataSources: DataSource[];
  dataSourceContexts: InvestigationDataContextItem[];
  memoryItems: InvestigationMemoryItem[];
  reports: ShareableReport[];
}) {
  return (
    <Card>
      <SectionHeader
        title="Findings"
        description="Analytical claims that can become memory, evidence, or report material."
        action={<Chip>{findings.length} total</Chip>}
      />
      {findings.length ? (
        <div className="mt-4 grid gap-3">
          {findings.slice(0, 6).map((finding) => (
            <FindingCard
              key={finding.finding_id}
              investigationId={investigationId}
              finding={finding}
              artifacts={artifacts}
              dataSources={dataSources}
              dataSourceContexts={dataSourceContexts}
              memoryItems={memoryItems}
              reports={reports}
            />
          ))}
          {findings.length > 6 ? (
            <TechnicalDetails title="Additional findings">
              <ul className="space-y-2">
                {findings.slice(6).map((finding) => {
                  const display = formatFindingForUser(finding);
                  return <li key={finding.finding_id}>{display.title}: {display.summary}</li>;
                })}
              </ul>
            </TechnicalDetails>
          ) : null}
        </div>
      ) : (
        <div className="mt-4">
          <EmptyState>Run this investigation to generate findings.</EmptyState>
        </div>
      )}
    </Card>
  );
}

function FindingCard({
  investigationId,
  finding,
  artifacts,
  dataSources,
  dataSourceContexts,
  memoryItems,
  reports
}: {
  investigationId: string;
  finding: Finding;
  artifacts: Artifact[];
  dataSources: DataSource[];
  dataSourceContexts: InvestigationDataContextItem[];
  memoryItems: InvestigationMemoryItem[];
  reports: ShareableReport[];
}) {
  const display = formatFindingForUser(finding);
  const hasTechnicalText = display.isTechnicalSummary || looksTechnical(finding.title);
  return (
    <Panel className="transition hover:border-slate-300 hover:bg-white dark:hover:border-slate-700 dark:hover:bg-slate-950">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <strong className="text-sm font-semibold text-slate-950 dark:text-slate-50">{display.title}</strong>
        <StatusBadge value={finding.status} />
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">{display.summary}</p>
      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <MiniInsightMetric label="Confidence" value={display.confidenceLabel} />
        <MiniInsightMetric label="Evidence" value={display.evidenceStrength} />
        <MiniInsightMetric label="Impact" value={display.businessImpact} />
      </div>
      {display.businessImplication ? (
        <p className="mt-3 rounded-xl bg-blue-50 p-3 text-xs leading-5 text-blue-900 dark:bg-blue-950/30 dark:text-blue-100">
          {display.businessImplication}
        </p>
      ) : null}
      {display.limitation || display.recommendedValidation ? (
        <TechnicalDetails title="View details">
          <dl className="space-y-2">
            {display.limitation ? (
              <div>
                <dt className="font-semibold text-slate-600 dark:text-slate-300">Limitation</dt>
                <dd>{display.limitation}</dd>
              </div>
            ) : null}
            {display.recommendedValidation ? (
              <div>
                <dt className="font-semibold text-slate-600 dark:text-slate-300">Recommended validation</dt>
                <dd>{display.recommendedValidation}</dd>
              </div>
            ) : null}
          </dl>
        </TechnicalDetails>
      ) : null}
      <FindingEvidenceActions
        investigationId={investigationId}
        finding={finding}
        artifacts={artifacts}
        dataSources={dataSources}
        dataSourceContexts={dataSourceContexts}
        memoryItems={memoryItems}
        reports={reports}
      />
      {hasTechnicalText ? (
        <TechnicalDetails>
          <dl className="space-y-2">
            <div>
              <dt className="font-semibold text-slate-600 dark:text-slate-300">Original title</dt>
              <dd className="break-words">{finding.title}</dd>
            </div>
            <div>
              <dt className="font-semibold text-slate-600 dark:text-slate-300">Original text</dt>
              <dd className="break-words">{finding.text}</dd>
            </div>
          </dl>
        </TechnicalDetails>
      ) : null}
    </Panel>
  );
}

function MiniInsightMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-2 py-1 dark:border-slate-800 dark:bg-slate-900">
      <div className="text-[10px] uppercase tracking-wide text-slate-400">{label}</div>
      <div className="truncate font-semibold text-slate-700 dark:text-slate-200">{value}</div>
    </div>
  );
}

function ArtifactsSection({ artifacts }: { artifacts: Artifact[] }) {
  const formatted = artifacts.map((artifact) => ({ artifact, display: formatArtifactForUser(artifact) }));
  const userArtifacts = formatted.filter((item) => !item.display.isTechnical);
  const technicalArtifacts = formatted.filter((item) => item.display.isTechnical);
  return (
    <Card>
      <SectionHeader
        title="Artifacts"
        description="Readable outputs created during investigation runs. Technical outputs are collapsed."
        action={<Chip>{userArtifacts.length} user-facing</Chip>}
      />
      {formatted.length ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {userArtifacts.slice(0, 6).map(({ artifact, display }) => (
            <div key={artifact.artifact_id} className="rounded-xl border border-slate-200 bg-white p-3 text-sm transition hover:-translate-y-0.5 hover:shadow-sm dark:border-slate-800 dark:bg-slate-950">
              <div className="flex items-center justify-between gap-2">
                <div className="font-medium text-slate-900 dark:text-slate-100">{display.title}</div>
                <Chip>{display.typeLabel}</Chip>
              </div>
              <p className="mt-2 text-xs leading-5 text-slate-500 dark:text-slate-400">{display.description}</p>
            </div>
          ))}
          {technicalArtifacts.length ? (
            <div className="sm:col-span-2">
              <TechnicalDetails title={`Technical artifacts (${technicalArtifacts.length})`}>
                <div className="grid gap-2 sm:grid-cols-2">
                  {technicalArtifacts.map(({ artifact, display }) => (
                    <div key={artifact.artifact_id} className="rounded-lg border border-slate-200 bg-white p-2 dark:border-slate-800 dark:bg-slate-950">
                      <div className="font-medium text-slate-700 dark:text-slate-200">{display.title}</div>
                      <div className="mt-1 text-slate-500">{display.typeLabel} · {artifact.visibility}</div>
                    </div>
                  ))}
                </div>
              </TechnicalDetails>
            </div>
          ) : null}
          {userArtifacts.length === 0 && technicalArtifacts.length ? (
            <div className="sm:col-span-2">
              <EmptyState>This run produced technical artifacts only. Open Technical artifacts for details.</EmptyState>
            </div>
          ) : null}
          {userArtifacts.length > 6 ? (
            <div className="sm:col-span-2">
              <TechnicalDetails title={`More user-facing artifacts (${userArtifacts.length - 6})`}>
                <ul className="space-y-1">
                  {userArtifacts.slice(6).map(({ artifact, display }) => (
                    <li key={artifact.artifact_id}>{display.typeLabel}: {display.title}</li>
                  ))}
                </ul>
              </TechnicalDetails>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="mt-4">
          <EmptyState>Artifacts from analysis runs will appear here.</EmptyState>
        </div>
      )}
    </Card>
  );
}

function ReportSummary({
  artifacts,
  findings,
  latestRun
}: {
  artifacts: Artifact[];
  findings: Finding[];
  latestRun?: InvestigationRun;
}) {
  const reportArtifacts = artifacts.filter((artifact) => (artifact.type || artifact.artifact_type) === "report");
  return (
    <Card>
      <SectionHeader
        title="Report path"
        description="Reports and final deliverables come after findings and artifacts are reviewed."
        action={latestRun ? <StatusBadge value={latestRun.status} /> : <StatusBadge value="not_started" />}
      />
      <div className="mt-4 grid gap-3 text-sm sm:grid-cols-3">
        <ReportStep title={findings.length ? "Review findings" : "Generate findings"} detail={`${findings.length} findings available`} />
        <ReportStep title={reportArtifacts.length ? "Report artifact exists" : "Create report"} detail={`${reportArtifacts.length} report artifacts`} />
        <ReportStep title="Finalize report" detail="Create a final version after review" />
      </div>
    </Card>
  );
}

function ReportStep({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3 dark:border-slate-800 dark:bg-slate-900/50">
      <div className="font-semibold text-slate-900 dark:text-slate-100">{title}</div>
      <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{detail}</div>
    </div>
  );
}
