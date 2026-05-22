"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { EventStreamResponse, RunEvent, getRunEvents } from "@/lib/api";
import { EmptyState, StatusBadge, formatDate, stageLabel } from "@/components/ui";

export function mergeRunEvents(current: RunEvent[], incoming: RunEvent[]) {
  const seen = new Set(current.map((event) => event.event_id));
  return [...current, ...incoming.filter((event) => !seen.has(event.event_id))].sort(
    (a, b) => a.created_at.localeCompare(b.created_at) || a.event_id.localeCompare(b.event_id)
  );
}

export function RunTimeline({ runId, status }: { runId: string; status?: string }) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const cursorRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    cursorRef.current = undefined;
    setEvents([]);

    async function poll() {
      try {
        const response: EventStreamResponse = await getRunEvents(runId, cursorRef.current, 100);
        if (cancelled) return;
        if (response.events.length > 0) {
          setEvents((current) => mergeRunEvents(current, response.events));
        }
        if (response.next_cursor) {
          cursorRef.current = response.next_cursor;
        }
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Could not load timeline");
      }
    }

    poll();
    const shouldContinue = status === "queued" || status === "running" || status === undefined;
    const id = shouldContinue ? window.setInterval(poll, 4000) : undefined;
    return () => {
      cancelled = true;
      if (id) window.clearInterval(id);
    };
  }, [runId, status]);

  const ordered = useMemo(
    () => mergeRunEvents([], events),
    [events]
  );

  if (error) return <p className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900/70 dark:bg-red-950/40 dark:text-red-300">{error}</p>;
  if (ordered.length === 0) return <EmptyState>No timeline events yet.</EmptyState>;

  return (
    <div className="relative space-y-4 before:absolute before:bottom-2 before:left-[11px] before:top-2 before:w-px before:bg-slate-200 dark:before:bg-slate-800">
      {ordered.map((event) => (
        <div key={event.event_id} className="relative pl-8">
          <div className={`absolute left-0 top-1.5 h-5 w-5 rounded-full border-4 border-white dark:border-slate-950 ${
            event.severity === "error" ? "bg-red-500" : event.severity === "warning" ? "bg-amber-500" : "bg-blue-500"
          }`} />
          <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-950">
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge value={event.severity} />
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">{stageLabel(event.stage)}</span>
              <span className="text-xs text-slate-400">{formatDate(event.created_at)}</span>
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-300">{eventMessage(event.message)}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

function eventMessage(message: string) {
  if (/Built usage context/i.test(message)) return "Analytica prepared the dataset context for this investigation.";
  if (/Prepared data sources/i.test(message)) return "Dataset context is ready.";
  if (/Loaded CSV/i.test(message)) return "Dataset loaded for analysis.";
  if (/Created an investigation artifact/i.test(message)) return "Saved a chart, table, or analytical output.";
  if (/Generated a DecisionReport/i.test(message)) return "Prepared a report-ready summary.";
  if (/Validation check/i.test(message)) return "Checked the analysis output.";
  if (/Run completed|Analysis pass updated|Latest analytical pass is available/i.test(message)) return "Latest analytical pass is available.";
  if (/semantic column notes/i.test(message)) return "No human notes are available yet; Analytica used inferred column roles.";
  if (/sample rows/i.test(message)) return "Sample rows were unavailable, so context may be less detailed.";
  return message.replace(/^.+?:\s*/, "");
}
