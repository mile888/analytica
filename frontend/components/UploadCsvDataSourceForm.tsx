"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";
import { uploadCsvDataSource, UploadCsvDataSourceResponse } from "@/lib/api";
import { parseTags } from "@/lib/forms";
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
        description: String(formData.get("description") || "") || null,
        tags: parseTags(String(formData.get("tags") || ""))
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
      <h2 className="text-sm font-semibold text-slate-950">Upload CSV</h2>
      <p className="mt-1 text-xs text-slate-500">Creates a DataSource, stores the file locally and profiles it.</p>
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
        <input name="tags" placeholder="Tags, comma-separated" className="rounded-md border border-slate-300 px-3 py-2 text-sm" />
        <button disabled={isSubmitting} className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:bg-slate-400">
          {isSubmitting ? "Uploading..." : "Upload CSV"}
        </button>
      </form>
      {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
      {created ? (
        <div className="mt-3 rounded-md border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">
          <p>
            Uploaded. <Link className="underline" href={`/data-sources/${created.data_source.data_source_id}`}>Open data source</Link>
          </p>
          <p className="mt-1 text-xs">
            Profile: {created.profile.row_count} rows, {created.profile.column_count} columns.
          </p>
        </div>
      ) : null}
    </Card>
  );
}
