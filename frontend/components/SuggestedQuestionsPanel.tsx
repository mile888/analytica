"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { createInvestigationMessage, runInvestigation } from "@/lib/api";
import { Button, Card, EmptyState, SectionHeader } from "@/components/ui";
import { truncateText } from "@/lib/display";

export function SuggestedQuestionsPanel({
  investigationId,
  suggestions,
  linkedDataSourceIds,
  latestFindings
}: {
  investigationId: string;
  suggestions: string[];
  linkedDataSourceIds: string[];
  latestFindings: string[];
}) {
  const router = useRouter();
  const [busyQuestion, setBusyQuestion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const visibleSuggestions = Array.from(new Set(suggestions.map((item) => item.trim()).filter(Boolean))).slice(0, 6);
  const visibleFindings = Array.from(new Set(latestFindings.map((item) => truncateText(item, 120)).filter(Boolean))).slice(0, 3);

  async function useQuestion(question: string) {
    setBusyQuestion(question);
    setError(null);
    try {
      const message = await createInvestigationMessage(investigationId, {
        content: question,
        type: "follow_up",
        role: "user"
      });
      await runInvestigation(investigationId, {
        message_id: message.message_id,
        data_source_ids: linkedDataSourceIds
      });
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not run suggested question");
    } finally {
      setBusyQuestion(null);
    }
  }

  return (
    <Card>
      <SectionHeader
        title="Suggested questions"
        description="Data-aware prompts based on profile, semantic notes, and column roles."
      />

      {visibleFindings.length ? (
        <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50/80 p-3 dark:border-slate-800 dark:bg-slate-900/50">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Latest findings</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-xs leading-5 text-slate-600 dark:text-slate-400">
            {visibleFindings.map((finding) => <li key={finding}>{finding}</li>)}
          </ul>
        </div>
      ) : null}

      <div className="mt-4 space-y-2">
        {visibleSuggestions.length ? (
          visibleSuggestions.map((question) => (
            <div key={question} className="group rounded-xl border border-slate-200 bg-white p-3 transition hover:border-blue-200 hover:bg-blue-50/40 dark:border-slate-800 dark:bg-slate-950 dark:hover:border-blue-900/70 dark:hover:bg-blue-950/20">
              <p className="text-sm leading-6 text-slate-700 dark:text-slate-300">{truncateText(question, 150)}</p>
              <Button
                type="button"
                onClick={() => useQuestion(question)}
                disabled={busyQuestion === question}
                variant="secondary"
                size="sm"
                className="mt-2"
              >
                {busyQuestion === question ? "Starting..." : "Use as investigation question"}
              </Button>
            </div>
          ))
        ) : (
          <EmptyState>Link a profiled data source to generate suggestions.</EmptyState>
        )}
      </div>
      {error ? <p className="mt-3 text-xs text-red-600">{error}</p> : null}
    </Card>
  );
}
