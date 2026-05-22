"use client";

import { useState } from "react";
import { ReportSectionEditor } from "@/components/ReportSectionEditor";
import { ReportComment, ShareableReport } from "@/lib/api";

export function ReportViewModeToggle({
  report,
  comments = []
}: {
  report: ShareableReport;
  comments?: ReportComment[];
}) {
  const [mode, setMode] = useState<"preview" | "edit">("preview");
  const orderedSections = [...report.sections].sort((a, b) => a.order - b.order);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-950">Sections</h2>
        <div className="inline-flex rounded-md border border-slate-200 bg-slate-50 p-1">
          <button
            type="button"
            onClick={() => setMode("preview")}
            className={`rounded px-3 py-1.5 text-xs font-medium ${mode === "preview" ? "bg-white text-slate-950 shadow-sm" : "text-slate-500"}`}
          >
            Preview
          </button>
          <button
            type="button"
            onClick={() => setMode("edit")}
            className={`rounded px-3 py-1.5 text-xs font-medium ${mode === "edit" ? "bg-white text-slate-950 shadow-sm" : "text-slate-500"}`}
          >
            Edit
          </button>
        </div>
      </div>

      {mode === "edit" ? (
        <div className="mt-4">
          <ReportSectionEditor report={report} comments={comments} />
        </div>
      ) : orderedSections.length ? (
        <div className="mt-4 space-y-4">
          {orderedSections.map((section) => (
            <section id={`section-${section.section_id}`} key={section.section_id} className="rounded-md border border-slate-200 p-4">
              <h3 className="font-medium text-slate-950">{section.title}</h3>
              <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-600">
                {section.content || "No content yet."}
              </p>
            </section>
          ))}
        </div>
      ) : (
        <p className="mt-3 text-sm text-slate-500">No sections yet.</p>
      )}
    </div>
  );
}
