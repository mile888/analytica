"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";
import { createInvestigation, DataSource } from "@/lib/api";
import { Card } from "@/components/ui";

export function CreateInvestigationForm({ dataSources }: { dataSources: DataSource[] }) {
  const router = useRouter();
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function createFromForm(formData: FormData) {
    setIsSubmitting(true);
    setError(null);
    try {
      const created = await createInvestigation({
        title: String(formData.get("title") || "") || null,
        question: String(formData.get("question") || ""),
        data_source_ids: selectedSourceIds
      });
      router.push(`/investigations/${created.investigation_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create investigation");
    } finally {
      setIsSubmitting(false);
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void createFromForm(new FormData(event.currentTarget));
  }

  return (
    <Card>
      <h2 className="text-sm font-semibold text-slate-950">Start new analysis</h2>
      <p className="mt-1 text-xs text-slate-500">Choose data, ask a question, then run from the detail page.</p>
      <form onSubmit={onSubmit} className="mt-4 grid gap-3">
        <input name="title" placeholder="Title, optional" className="rounded-md border border-slate-300 px-3 py-2 text-sm" />
        <textarea name="question" required placeholder="What should we investigate?" rows={3} className="rounded-md border border-slate-300 px-3 py-2 text-sm" />
        {dataSources.length ? (
          <div className="space-y-2 rounded-md border border-slate-200 p-3">
            <div className="text-xs font-medium text-slate-500">Linked data sources</div>
            {dataSources.map((source) => (
              <label key={source.data_source_id} className="flex items-center gap-2 text-xs text-slate-600">
                <input
                  type="checkbox"
                  checked={selectedSourceIds.includes(source.data_source_id)}
                  onChange={(event) => {
                    setSelectedSourceIds((current) =>
                      event.target.checked
                        ? [...current, source.data_source_id]
                        : current.filter((id) => id !== source.data_source_id)
                    );
                  }}
                />
                <span>{source.name}</span>
              </label>
            ))}
          </div>
        ) : (
          <p className="rounded-md border border-dashed border-slate-300 p-3 text-xs text-slate-500">
            No data sources yet. You can still create an investigation and link data later.
          </p>
        )}
        <button disabled={isSubmitting} className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:bg-slate-400">
          {isSubmitting ? "Creating..." : "Create investigation"}
        </button>
      </form>
      {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
    </Card>
  );
}
