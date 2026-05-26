"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";
import { createDataSource } from "@/lib/api";

const sourceTypes = ["csv", "sqlite", "postgres", "duckdb", "unknown"] as const;

export function CreateDataSourceForm() {
  const router = useRouter();
  const [createdId, setCreatedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function createFromForm(formData: FormData) {
    setIsSubmitting(true);
    setError(null);
    setCreatedId(null);
    try {
      const created = await createDataSource({
        name: String(formData.get("name") || ""),
        type: String(formData.get("type") || "unknown") as (typeof sourceTypes)[number],
        location: String(formData.get("location") || "") || null,
        description: String(formData.get("description") || "") || null
      });
      setCreatedId(created.data_source_id);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create data source");
    } finally {
      setIsSubmitting(false);
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void createFromForm(new FormData(event.currentTarget));
  }

  return (
    <details className="rounded-2xl border border-slate-200/80 bg-white/90 p-5 shadow-sm shadow-slate-200/60 dark:border-slate-800 dark:bg-slate-950/85 dark:shadow-black/20">
      <summary className="cursor-pointer text-sm font-semibold text-slate-950 dark:text-slate-50">
        Advanced: create a metadata-only dataset
      </summary>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
        Use this only when the data lives elsewhere. The primary flow is CSV upload above.
      </p>
      <form onSubmit={onSubmit} className="mt-4 grid gap-3">
        <input name="name" required placeholder="Name" className="rounded-md border border-slate-300 px-3 py-2 text-sm" />
        <select name="type" defaultValue="csv" className="rounded-md border border-slate-300 px-3 py-2 text-sm">
          {sourceTypes.map((type) => <option key={type} value={type}>{type}</option>)}
        </select>
        <input name="location" placeholder="Location or connection reference" className="rounded-md border border-slate-300 px-3 py-2 text-sm" />
        <textarea name="description" placeholder="Description" rows={3} className="rounded-md border border-slate-300 px-3 py-2 text-sm" />
        <button disabled={isSubmitting} className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:bg-slate-400">
          {isSubmitting ? "Creating..." : "Create source"}
        </button>
      </form>
      {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
      {createdId ? (
        <p className="mt-3 text-sm text-emerald-700">
          Created. <Link className="underline" href={`/data-sources/${createdId}`}>Open dataset</Link>
        </p>
      ) : null}
    </details>
  );
}
