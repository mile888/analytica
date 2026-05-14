"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { EventStreamResponse, RunEvent, getRunEvents } from "@/lib/api";
import { StatusBadge, formatDate, stageLabel } from "@/components/ui";

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

  if (error) return <p className="text-sm text-red-600">{error}</p>;
  if (ordered.length === 0) return <p className="text-sm text-slate-500">No timeline events yet.</p>;

  return (
    <div className="space-y-3">
      {ordered.map((event) => (
        <div key={event.event_id} className="border-l-2 border-slate-200 pl-3">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge value={event.severity} />
            <span className="text-xs font-medium text-slate-500">{stageLabel(event.stage)}</span>
            <span className="text-xs text-slate-400">{formatDate(event.created_at)}</span>
          </div>
          <p className="mt-1 text-sm text-slate-700">{event.message}</p>
        </div>
      ))}
    </div>
  );
}
