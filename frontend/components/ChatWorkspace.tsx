"use client";

import { useEffect, useRef, useState } from "react";
import type {
  Artifact,
  DataSource,
  Finding,
  Investigation,
  InvestigationBranch,
  InvestigationMessage,
  InvestigationRun,
} from "@/lib/api";
import { ChartArtifactCard } from "@/components/ChartArtifactCard";
import { FollowUpComposer } from "@/components/FollowUpComposer";
import { TableArtifactCard } from "@/components/TableArtifactCard";
import { Chip } from "@/components/ui";
import { assistantAnswerForRun, cleanText, dedupeConsecutiveMessages, formatMessageForUser } from "@/lib/display";
import { artifactsForBranch, prioritizeArtifactsForBranch, visualAnalysisArtifacts } from "@/lib/visualArtifacts";

export function ChatWorkspace({
  investigation,
  latestRun,
  messages,
  findings,
  artifacts,
  suggestions,
  branches = [],
  dataSources = []
}: {
  investigation: Investigation;
  latestRun?: InvestigationRun;
  messages: InvestigationMessage[];
  findings: Finding[];
  artifacts: Artifact[];
  suggestions: string[];
  branches?: InvestigationBranch[];
  dataSources?: DataSource[];
}) {
  const [optimisticFollowUps, setOptimisticFollowUps] = useState<Array<{ id: string; question: string }>>([]);
  const [initialRunPending, setInitialRunPending] = useState(false);
  const threadEndRef = useRef<HTMLDivElement | null>(null);
  const visibleMessages = dedupeConsecutiveMessages(messages).filter((message) => {
    const messageType = message.message_type || message.type;
    return !(messageType === "question" && cleanText(message.content) === cleanText(investigation.user_question));
  });
  const confirmedUserMessageIds = new Set(
    visibleMessages
      .filter((message) => {
        const messageType = message.message_type || message.type;
        return message.role === "user" || messageType === "question" || messageType === "follow_up";
      })
      .map((message) => message.message_id)
  );
  const confirmedUserQuestions = new Set(
    visibleMessages
      .filter(isUserLikeMessage)
      .map((message) => cleanText(message.content).toLowerCase())
  );
  const pendingFollowUps = optimisticFollowUps.filter(
    (item) => !confirmedUserMessageIds.has(item.id) && !confirmedUserQuestions.has(cleanText(item.question).toLowerCase())
  );
  const hasAssistantMessage = visibleMessages.some(isAssistantLikeMessage);
  const lastVisibleMessage = visibleMessages[visibleMessages.length - 1];
  const lastMessageNeedsAnswer = Boolean(lastVisibleMessage && isUserLikeMessage(lastVisibleMessage));
  const canRunCurrentQuestion = !hasAssistantMessage && !findings.length && !latestRun;
  const syntheticAssistant = hasAssistantMessage && !lastMessageNeedsAnswer
    ? null
    : assistantAnswerForRun(
        latestRun,
        findings.length,
        artifacts.filter((artifact) => artifact.visibility !== "hidden").length
      );
  const visibleArtifacts = artifacts.filter((artifact) => artifact.visibility !== "hidden");
  const activeBranch = branches.find((branch) => branch.is_active);
  const showDatasetDetails = dataSources.length > 1;
  const dataSourceNames = new Map(dataSources.map((source) => [source.data_source_id, source.name]));
  const selectedBranchArtifacts = artifactsForBranch(visibleArtifacts, activeBranch?.branch_id);
  const relevantVisualArtifacts = visualAnalysisArtifacts(prioritizeArtifactsForBranch(visibleArtifacts, activeBranch?.branch_id));
  const relevantChartArtifacts = relevantVisualArtifacts.charts.slice(-2);
  const relevantTableArtifacts = relevantVisualArtifacts.tables.slice(-2);
  const hasRelevantVisualArtifacts = Boolean(relevantChartArtifacts.length || relevantTableArtifacts.length);

  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [visibleMessages.length, pendingFollowUps.length, initialRunPending, relevantChartArtifacts.length, relevantTableArtifacts.length]);

  return (
    <section className="flex min-h-[calc(100vh-8rem)] flex-col rounded-3xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-950">
      <header className="border-b border-slate-200 bg-white/90 px-5 py-4 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90 sm:px-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <Chip>{investigation.linked_data_source_ids.length} dataset{investigation.linked_data_source_ids.length === 1 ? "" : "s"}</Chip>
              {showDatasetDetails
                ? dataSources.slice(0, 4).map((source) => (
                    <Chip key={source.data_source_id}>{source.name}</Chip>
                  ))
                : null}
              {showDatasetDetails && activeBranch?.dataset_ids?.length ? (
                <Chip>Active: {activeBranch.dataset_ids.map((id) => dataSourceNames.get(id) || id).join(", ")}</Chip>
              ) : null}
            </div>
            <h1 className="line-clamp-2 text-xl font-semibold tracking-tight text-slate-950 dark:text-slate-50 sm:text-2xl">
              {investigation.title}
            </h1>
          </div>
        </div>
      </header>
      <div className="flex-1 bg-[radial-gradient(circle_at_top,rgba(59,130,246,0.06),transparent_30%)] px-4 py-6 dark:bg-[radial-gradient(circle_at_top,rgba(59,130,246,0.10),transparent_30%)] sm:px-6">
        <div className="mx-auto max-w-3xl space-y-5">
          <ChatBubble
            label="You"
            tone="user"
            content={investigation.user_question}
          />

          {visibleMessages.map((message) => {
            const display = formatMessageForUser(message);
            return (
              <ChatBubble
                key={message.message_id}
                label={display.label}
                tone={display.tone}
                content={display.content}
                status={display.status}
              />
            );
          })}

          {syntheticAssistant ? (
            <ChatBubble
              label={syntheticAssistant.label}
              tone={syntheticAssistant.tone}
              content={syntheticAssistant.content}
              status={syntheticAssistant.status}
            />
          ) : null}

          {activeBranch && !selectedBranchArtifacts.length && !hasRelevantVisualArtifacts ? (
            <div className="rounded-2xl border border-dashed border-slate-200 bg-white/70 p-4 text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-950/70 dark:text-slate-400">
              No visual outputs yet for this branch.
            </div>
          ) : hasRelevantVisualArtifacts ? (
            <div className="space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">Visual analysis</div>
              {relevantChartArtifacts.map((artifact) => (
                <ChartArtifactCard
                  key={artifact.artifact_id}
                  artifact={artifact}
                  investigationId={investigation.investigation_id}
                  linkedDataSourceIds={investigation.linked_data_source_ids}
                  datasetLabel={artifactDatasetLabel(artifact, dataSourceNames, showDatasetDetails)}
                />
              ))}
              {relevantTableArtifacts.map((artifact) => (
                <TableArtifactCard
                  key={artifact.artifact_id}
                  artifact={artifact}
                  investigationId={investigation.investigation_id}
                  datasetLabel={artifactDatasetLabel(artifact, dataSourceNames, showDatasetDetails)}
                />
              ))}
            </div>
          ) : null}

          {pendingFollowUps.map(({ id, question }) => (
            <div key={id} className="space-y-3">
              <ChatBubble
                label="You"
                tone="user"
                content={question}
                pending
              />
              <ChatBubble
                label="Analytica"
                tone="assistant"
                content="Analytica is analyzing this question..."
                pending
              />
            </div>
          ))}

          {initialRunPending ? (
            <ChatBubble
              label="Analytica"
              tone="assistant"
              content="Analytica is analyzing this question..."
              pending
            />
          ) : null}

          {!findings.length && !syntheticAssistant && !hasAssistantMessage && !initialRunPending ? (
            <div className="rounded-3xl border border-dashed border-slate-300 bg-white/70 p-6 text-center dark:border-slate-700 dark:bg-slate-950/70">
              <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-blue-50 text-xl dark:bg-blue-950/40">↗</div>
              <h2 className="text-sm font-semibold text-slate-950 dark:text-slate-50">No assistant answer yet</h2>
              <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500 dark:text-slate-400">
                Click Analyze to run the investigation. Key findings, charts, and report-ready conclusions will collect on the left.
              </p>
            </div>
          ) : null}
          <div ref={threadEndRef} aria-hidden="true" />
        </div>
      </div>

      <FollowUpComposer
        investigationId={investigation.investigation_id}
        linkedDataSourceIds={investigation.linked_data_source_ids}
        canRunCurrentQuestion={canRunCurrentQuestion}
        suggestions={suggestions}
        onCurrentQuestionStarted={() => setInitialRunPending(true)}
        onCurrentQuestionSettled={() => setInitialRunPending(false)}
        onStarted={(question, messageId) =>
          setOptimisticFollowUps((current) => {
            return current.some((item) => item.id === messageId) ? current : [...current, { id: messageId, question }];
          })
        }
        onSettled={(_, messageId) =>
          setOptimisticFollowUps((current) => {
            return current.filter((item) => item.id !== messageId);
          })
        }
      />
    </section>
  );
}

function artifactDatasetLabel(artifact: Artifact, names: Map<string, string>, enabled: boolean): string {
  if (!enabled) return "";
  const metadata = artifact.metadata || {};
  const ids = Array.isArray(metadata.dataset_ids) ? metadata.dataset_ids.map(String).filter(Boolean) : [];
  const single = typeof metadata.dataset_id === "string" && metadata.dataset_id ? [metadata.dataset_id] : [];
  const selected = ids.length ? ids : single;
  return selected.map((id) => names.get(id) || id).join(", ");
}

function isUserLikeMessage(message: InvestigationMessage): boolean {
  const messageType = message.message_type || message.type;
  return message.role === "user" || messageType === "question" || messageType === "follow_up";
}

function isAssistantLikeMessage(message: InvestigationMessage): boolean {
  const messageType = message.message_type || message.type;
  return message.role === "assistant" || messageType === "run_summary" || messageType === "answer" || messageType === "error";
}

function ChatBubble({
  label,
  tone,
  content,
  status,
  pending = false
}: {
  label: string;
  tone: "user" | "assistant" | "system" | "error";
  content: string;
  status?: string;
  pending?: boolean;
}) {
  const isUser = tone === "user";
  return (
    <article className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[92%] overflow-visible rounded-3xl px-4 py-3 shadow-sm sm:max-w-[88%] ${
          isUser
            ? "bg-slate-950 text-white dark:bg-slate-100 dark:text-slate-950"
            : tone === "error"
              ? "border border-red-200 bg-red-50 text-red-900 dark:border-red-900/70 dark:bg-red-950/40 dark:text-red-100"
              : "border border-slate-200 bg-white text-slate-800 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-100"
        }`}
      >
        <div className={`mb-1 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wide ${isUser ? "text-white/65 dark:text-slate-700" : "text-slate-400"}`}>
          <span>{label}</span>
          {pending ? <span className="animate-pulse">Thinking</span> : null}
        </div>
        <div className="max-h-none whitespace-pre-wrap break-words text-sm leading-6 [overflow-wrap:anywhere]">{content}</div>
        {status ? <div className="mt-2 text-xs font-medium opacity-70">{status}</div> : null}
      </div>
    </article>
  );
}
