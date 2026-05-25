"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";
import { uploadCsvDataSource, UploadCsvDataSourceResponse } from "@/lib/api";
import { Card } from "@/components/ui";

export function UploadCsvDataSourceForm() {
  const router = useRouter();
  const [created, setCreated] = useState<UploadCsvDataSourceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function uploadFromForm(formData: FormData) {
    const file = formData.get("file");
    if (!(file instanceof File) || !file.name) {
      setError("Choose a CSV file first.");
      return;
    }
    setIsSubmitting(true);
    setError(null);
    setCreated(null);
    try {
      const response = await uploadCsvDataSource({
        file,
        name: String(formData.get("name") || "") || null,
        description: String(formData.get("description") || "") || null
      });
      setCreated(response);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not upload CSV");
    } finally {
      setIsSubmitting(false);
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void uploadFromForm(new FormData(event.currentTarget));
  }

  return (
    <Card>
      <h2 className="text-sm font-semibold text-slate-950 dark:text-slate-50">Upload dataset</h2>
      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
        Upload a CSV and Analytica will create the dataset, profile it, and make it available for investigation.
      </p>
      <form onSubmit={onSubmit} className="mt-4 grid gap-3">
        <input
          name="file"
          required
          type="file"
          accept=".csv,text/csv"
          className="rounded-md border border-slate-300 px-3 py-2 text-sm"
        />
        <input name="name" placeholder="Name, optional" className="rounded-md border border-slate-300 px-3 py-2 text-sm" />
        <textarea name="description" placeholder="Description" rows={3} className="rounded-md border border-slate-300 px-3 py-2 text-sm" />
        <button disabled={isSubmitting} className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:bg-slate-400">
          {isSubmitting ? "Uploading..." : "Upload CSV"}
        </button>
      </form>
      {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
      {created ? (
        <div className="mt-3 rounded-md border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800 dark:border-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
          <p>
            Uploaded. <Link className="underline" href={`/data-sources/${created.data_source.data_source_id}`}>Open dataset</Link>
          </p>
          {created.profile ? (
            <p className="mt-1 text-xs">
              Profile: {created.profile.row_count} rows, {created.profile.column_count} columns.
            </p>
          ) : (
            <p className="mt-1 text-xs text-amber-700 dark:text-amber-400">
              Dataset saved, but profiling could not complete. The dataset is still usable.
            </p>
          )}
          {created.warnings && created.warnings.length > 0 ? (
            <div className="mt-2 rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800 dark:border-amber-700 dark:bg-amber-900/30 dark:text-amber-300">
              <p className="font-medium">Warnings:</p>
              <ul className="ml-3 mt-1 list-disc">
                {created.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}

    </Card>
  );
}
