"use client";

import { useRouter } from "next/navigation";
import { KeyboardEvent, useRef, useState } from "react";
import { createInvestigationMessage, runInvestigation } from "@/lib/api";
import { Button } from "@/components/ui";
import { useToast } from "@/components/ToastProvider";

export function FollowUpComposer({
  investigationId,
  linkedDataSourceIds,
  canRunCurrentQuestion = false,
  suggestions = [],
  onCurrentQuestionStarted,
  onCurrentQuestionSettled,
  onStarted,
  onSettled
}: {
  investigationId: string;
  linkedDataSourceIds: string[];
  canRunCurrentQuestion?: boolean;
  suggestions?: string[];
  onCurrentQuestionStarted?: () => void;
  onCurrentQuestionSettled?: () => void;
  onStarted?: (question: string, messageId: string) => void;
  onSettled?: (question: string, messageId: string) => void;
}) {
  const router = useRouter();
  const { showToast } = useToast();
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const [content, setContent] = useState("");
  const [busy, setBusy] = useState(false);
  const visibleSuggestions = Array.from(new Set(suggestions.map((item) => item.trim()).filter(Boolean))).slice(0, 5);

  async function submit(question = content) {
    const trimmed = question.trim();
    if (busy) return;
    if (!trimmed) {
      if (canRunCurrentQuestion) {
        await runCurrentQuestion();
        return;
      }
      inputRef.current?.focus();
      showToast("Enter a question to analyze.", "info");
      return;
    }
    setBusy(true);
    let messageId = "";
    try {
      const message = await createInvestigationMessage(investigationId, {
        content: trimmed,
        type: "follow_up",
        role: "user"
      });
      messageId = message.message_id;
      onStarted?.(trimmed, message.message_id);
      setContent("");
      showToast("Analysis started", "info");
      await runInvestigation(investigationId, {
        message_id: message.message_id,
        data_source_ids: linkedDataSourceIds
      });
      showToast("Analytica answered", "success");
      router.refresh();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Could not run follow-up", "error");
      if (messageId) onSettled?.(trimmed, messageId);
    } finally {
      setBusy(false);
    }
  }

  async function runCurrentQuestion() {
    setBusy(true);
    onCurrentQuestionStarted?.();
    try {
      showToast("Analysis started", "info");
      await runInvestigation(investigationId, {
        data_source_ids: linkedDataSourceIds
      });
      showToast("Analytica answered", "success");
      router.refresh();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Could not run analysis", "error");
    } finally {
      onCurrentQuestionSettled?.();
      setBusy(false);
    }
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      submit();
    }
  }

  return (
    <div className="sticky bottom-0 z-20 border-t border-slate-200 bg-slate-50/95 px-4 py-4 backdrop-blur dark:border-slate-800 dark:bg-slate-950/95 sm:px-6">
      {visibleSuggestions.length ? (
        <div className="mb-3 flex gap-2 overflow-x-auto pb-1">
          {visibleSuggestions.map((question) => (
            <button
              key={question}
              type="button"
              onClick={() => submit(question)}
              disabled={busy}
              className="shrink-0 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-blue-900 dark:hover:bg-blue-950/30"
            >
              {question.length > 72 ? `${question.slice(0, 71)}…` : question}
            </button>
          ))}
        </div>
      ) : null}
      <div className="rounded-2xl border border-slate-200 bg-white p-2 shadow-lg shadow-slate-200/70 dark:border-slate-800 dark:bg-slate-950 dark:shadow-black/20">
        <textarea
          ref={inputRef}
          value={content}
          onChange={(event) => setContent(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Analyze another question about this investigation..."
          className="min-h-20 w-full resize-none rounded-xl border-0 bg-transparent px-3 py-2 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400 dark:text-slate-100"
        />
        <div className="flex items-center justify-between gap-3 px-1 pb-1">
          <span className="text-xs text-slate-400">Press Cmd/Ctrl + Enter to run</span>
          <Button type="button" onClick={() => submit()} disabled={busy}>
            {busy ? "Analyzing..." : "Analyze"}
          </Button>
        </div>
      </div>
    </div>
  );
}
