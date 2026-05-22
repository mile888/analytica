"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  InvestigationMessage,
  createInvestigationMessage,
  listInvestigationMessages,
  runInvestigation
} from "@/lib/api";
import { Button, Card, EmptyState, StatusBadge, formatDate } from "@/components/ui";
import { dedupeConsecutiveMessages, formatMessageForUser } from "@/lib/display";

export function InvestigationConversationPanel({
  investigationId,
  initialQuestion,
  initialMessages,
  linkedDataSourceIds
}: {
  investigationId: string;
  initialQuestion: string;
  initialMessages: InvestigationMessage[];
  linkedDataSourceIds: string[];
}) {
  const router = useRouter();
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const [messages, setMessages] = useState<InvestigationMessage[]>(initialMessages);
  const [content, setContent] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const visibleMessages = dedupeConsecutiveMessages(messages);

  async function analyzeFollowUp() {
    const trimmed = content.trim();
    if (!trimmed) {
      inputRef.current?.focus();
      setError("Enter a question to analyze.");
      return;
    }
    setIsSubmitting(true);
    setError(null);
    try {
      const message = await createInvestigationMessage(investigationId, {
        content: trimmed,
        type: "follow_up"
      });
      setMessages((current) => [...current, message]);
      setContent("");
      await runInvestigation(investigationId, {
        data_source_ids: linkedDataSourceIds,
        message_id: message.message_id
      });
      setMessages(await listInvestigationMessages(investigationId));
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not analyze follow-up");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Card>
      <h2 className="text-base font-semibold text-slate-950 dark:text-slate-50">Conversation</h2>
      <p className="mt-1 text-sm leading-6 text-slate-500 dark:text-slate-400">
        Continue analysis without leaving the investigation workspace.
      </p>

      <div className="mt-4 space-y-3">
        <div className="rounded-xl border border-slate-200 bg-slate-50/80 p-3 dark:border-slate-800 dark:bg-slate-900/50">
          <div className="flex items-center gap-2">
            <StatusBadge value="question" />
            <span className="text-xs text-slate-500 dark:text-slate-400">Initial question</span>
          </div>
          <p className="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-300">{initialQuestion}</p>
        </div>

        {visibleMessages.map((message) => {
          const display = formatMessageForUser(message);
          return (
            <div
              key={message.message_id}
              className={`rounded-xl border p-3 shadow-sm ${
                display.tone === "assistant"
                  ? "border-blue-100 bg-blue-50/60 dark:border-blue-900/60 dark:bg-blue-950/20"
                  : display.tone === "error"
                    ? "border-red-200 bg-red-50/70 dark:border-red-900/70 dark:bg-red-950/20"
                    : "border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950"
              }`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <StatusBadge value={display.tone === "error" ? "analysis_failed" : display.tone === "assistant" ? "assistant" : "message"} />
                <span className="text-xs text-slate-400">{formatDate(message.created_at)}</span>
              </div>
              <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-700 dark:text-slate-300">{display.content}</p>
            </div>
          );
        })}

        {visibleMessages.length === 0 ? (
          <EmptyState>Analyze another question to continue this investigation.</EmptyState>
        ) : null}
      </div>

      <div className="mt-4 space-y-2">
        <textarea
          ref={inputRef}
          value={content}
          onChange={(event) => setContent(event.target.value)}
          placeholder="Analyze another question..."
          className="min-h-28 w-full rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-900 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-500/10 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-100"
        />
        <Button
          onClick={analyzeFollowUp}
          disabled={isSubmitting}
          className="w-full"
        >
          {isSubmitting ? "Analyzing..." : "Analyze"}
        </Button>
        {error ? <p className="text-sm text-red-600">{error}</p> : null}
      </div>
    </Card>
  );
}
