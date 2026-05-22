"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import {
  InvestigationMemoryItem,
  addInvestigationMemory,
  updateInvestigationMemory
} from "@/lib/api";
import { Button, Card, EmptyState, SectionHeader, StatusBadge, formatDate } from "@/components/ui";

const memoryTypes = [
  { value: "assumption", label: "Assumptions" },
  { value: "open_question", label: "Unresolved questions" },
  { value: "decision", label: "Key decisions" },
  { value: "risk", label: "Investigation risks" },
  { value: "milestone", label: "Milestones" }
] as const;

export function InvestigationMemoryPanel({
  investigationId,
  initialItems
}: {
  investigationId: string;
  initialItems: InvestigationMemoryItem[];
}) {
  const router = useRouter();
  const [items, setItems] = useState(initialItems);
  const [type, setType] = useState<InvestigationMemoryItem["memory_type"]>("assumption");
  const [content, setContent] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function addItem() {
    const trimmed = content.trim();
    if (!trimmed) return;
    setBusy("add");
    setError(null);
    try {
      const created = await addInvestigationMemory(investigationId, { type, content: trimmed });
      setItems((current) => [created, ...current]);
      setContent("");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save memory item");
    } finally {
      setBusy(null);
    }
  }

  async function setStatus(item: InvestigationMemoryItem, status: InvestigationMemoryItem["status"]) {
    setBusy(`${item.memory_id}-${status}`);
    setError(null);
    try {
      const updated = await updateInvestigationMemory(investigationId, item.memory_id, { status });
      setItems((current) => current.map((candidate) => (candidate.memory_id === item.memory_id ? updated : candidate)));
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update memory item");
    } finally {
      setBusy(null);
    }
  }

  return (
    <Card>
      <SectionHeader
        title="Investigation Memory"
        description="Persistent context that carries across follow-up runs."
      />

      <div className="mt-4 space-y-4">
        {memoryTypes.slice(0, 4).map((entry) => {
          const group = items.filter((item) => item.memory_type === entry.value && item.status !== "archived");
          return (
            <section key={entry.value}>
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">{entry.label}</h3>
              {group.length ? (
                <div className="mt-2 space-y-2">
                  {group.slice(0, 4).map((item) => (
                    <article key={item.memory_id} className="rounded-xl border border-slate-200 bg-slate-50/80 p-3 dark:border-slate-800 dark:bg-slate-900/50">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <StatusBadge value={item.status} />
                        <span className="text-xs text-slate-500">{formatDate(item.updated_at)}</span>
                      </div>
                      <p className="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-300">{item.content}</p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {item.status === "active" ? (
                          <button
                            type="button"
                            onClick={() => setStatus(item, "resolved")}
                            disabled={busy === `${item.memory_id}-resolved`}
                            className="text-xs font-semibold text-slate-600 hover:text-slate-950 dark:text-slate-400 dark:hover:text-slate-100"
                          >
                            Mark resolved
                          </button>
                        ) : null}
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="mt-2 rounded-lg border border-dashed border-slate-200 p-3 text-xs text-slate-500 dark:border-slate-800">No items yet.</p>
              )}
            </section>
          );
        })}
      </div>

      <div className="mt-4 border-t border-slate-200 pt-4 dark:border-slate-800">
        <label className="text-xs font-medium text-slate-500">Add memory</label>
        <select
          value={type}
          onChange={(event) => setType(event.target.value as InvestigationMemoryItem["memory_type"])}
          className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-500/10 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
        >
          {memoryTypes.map((entry) => (
            <option key={entry.value} value={entry.value}>{entry.label}</option>
          ))}
        </select>
        <textarea
          value={content}
          onChange={(event) => setContent(event.target.value)}
          placeholder="Add an assumption, open question, decision, or risk..."
          className="mt-2 min-h-20 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-500/10 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
        />
        <Button
          type="button"
          onClick={addItem}
          disabled={!content.trim() || busy === "add"}
          className="mt-2 w-full"
        >
          {busy === "add" ? "Saving..." : "Add to memory"}
        </Button>
        {error ? <p className="mt-2 text-xs text-red-600">{error}</p> : null}
      </div>
    </Card>
  );
}
