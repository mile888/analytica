"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import {
  ReportComment,
  ReportSection,
  ShareableReport,
  addReportComment,
  addReportSection,
  approveReportSection,
  deleteReportSection,
  duplicateReportSection,
  reorderReportSections,
  requestReportSectionChanges,
  updateReportSection
} from "@/lib/api";
import { ReportComments } from "@/components/ReportComments";
import { StatusBadge } from "@/components/ui";

export function ReportSectionEditor({
  report,
  comments = []
}: {
  report: ShareableReport;
  comments?: ReportComment[];
}) {
  const router = useRouter();
  const [currentReportId, setCurrentReportId] = useState(report.report_id);
  const [sections, setSections] = useState([...report.sections].sort((a, b) => a.order - b.order));
  const [drafts, setDrafts] = useState(() => Object.fromEntries(sections.map((section) => [section.section_id, section])));
  const [isAdding, setIsAdding] = useState(false);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  function applyReport(updated: ShareableReport) {
    const ordered = [...updated.sections].sort((a, b) => a.order - b.order);
    setCurrentReportId(updated.report_id);
    setSections(ordered);
    setDrafts(Object.fromEntries(ordered.map((section) => [section.section_id, section])));
    router.refresh();
  }

  async function saveSection(sectionId: string) {
    const draft = drafts[sectionId];
    if (!draft) return;
    await mutate(`save-${sectionId}`, () =>
      updateReportSection(currentReportId, sectionId, {
        title: draft.title,
        content: draft.content
      })
    );
  }

  async function addSection() {
    await mutate("add-section", () =>
      addReportSection(currentReportId, {
        title: "New section",
        content: ""
      })
    );
    setIsAdding(false);
  }

  async function removeSection(sectionId: string) {
    await mutate(`delete-${sectionId}`, () => deleteReportSection(currentReportId, sectionId));
  }

  async function duplicateSection(sectionId: string) {
    await mutate(`duplicate-${sectionId}`, () => duplicateReportSection(currentReportId, sectionId));
  }

  async function moveSection(sectionId: string, direction: -1 | 1) {
    const index = sections.findIndex((section) => section.section_id === sectionId);
    const target = index + direction;
    if (index < 0 || target < 0 || target >= sections.length) return;
    const next = [...sections];
    [next[index], next[target]] = [next[target], next[index]];
    await mutate(`move-${sectionId}`, () => reorderReportSections(currentReportId, next.map((section) => section.section_id)));
  }

  async function approveSection(sectionId: string) {
    await mutate(`approve-${sectionId}`, () => approveReportSection(currentReportId, sectionId));
  }

  async function requestChanges(sectionId: string, reason?: string) {
    await mutate(`changes-${sectionId}`, async () => {
      const updated = await requestReportSectionChanges(currentReportId, sectionId);
      if (reason?.trim()) {
        await addReportComment(updated.report_id, {
          section_id: sectionId,
          text: reason.trim(),
          author: "reviewer"
        });
      }
      return updated;
    });
  }

  async function mutate(key: string, action: () => Promise<ShareableReport>) {
    setBusyKey(key);
    setError(null);
    try {
      applyReport(await action());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Report edit failed");
    } finally {
      setBusyKey(null);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-950">Edit sections</h2>
          <p className="mt-1 text-xs text-slate-500">
            Plain text edits create a new report version through the backend.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setIsAdding(true)}
          className="rounded-md border border-slate-300 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50"
        >
          Add section
        </button>
      </div>

      {isAdding ? (
        <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 p-4">
          <p className="text-sm text-slate-600">A new empty section will be added at the end.</p>
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={addSection}
              disabled={busyKey === "add-section"}
              className="rounded-md bg-slate-900 px-3 py-2 text-xs font-medium text-white disabled:bg-slate-400"
            >
              {busyKey === "add-section" ? "Adding..." : "Add"}
            </button>
            <button
              type="button"
              onClick={() => setIsAdding(false)}
              className="rounded-md border border-slate-300 px-3 py-2 text-xs font-medium text-slate-700"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}

      {sections.map((section, index) => (
        <SectionEditor
          key={section.section_id}
          section={drafts[section.section_id] || section}
          index={index}
          total={sections.length}
          busyKey={busyKey}
          onChange={(updated) => setDrafts((current) => ({ ...current, [section.section_id]: updated }))}
          onSave={() => saveSection(section.section_id)}
          onDuplicate={() => duplicateSection(section.section_id)}
          onDelete={() => removeSection(section.section_id)}
          onMoveUp={() => moveSection(section.section_id, -1)}
          onMoveDown={() => moveSection(section.section_id, 1)}
          onApprove={() => approveSection(section.section_id)}
          onRequestChanges={(reason) => requestChanges(section.section_id, reason)}
          comments={comments.filter((comment) => comment.section_id === section.section_id)}
          reportId={currentReportId}
        />
      ))}

      {sections.length === 0 ? (
        <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-500">
          No sections yet. Add a section to start editing the report.
        </div>
      ) : null}
      {error ? <p className="text-sm text-red-600">{error}</p> : null}
    </div>
  );
}

function SectionEditor({
  section,
  index,
  total,
  busyKey,
  onChange,
  onSave,
  onDuplicate,
  onDelete,
  onMoveUp,
  onMoveDown,
  onApprove,
  onRequestChanges,
  comments,
  reportId
}: {
  section: ReportSection;
  index: number;
  total: number;
  busyKey: string | null;
  onChange: (section: ReportSection) => void;
  onSave: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
  onApprove: () => void;
  onRequestChanges: (reason?: string) => void;
  comments: ReportComment[];
  reportId: string;
}) {
  const [changeReason, setChangeReason] = useState("");
  const openCommentCount = comments.filter((comment) => comment.status === "open").length;

  return (
    <section id={`section-${section.section_id}`} className="rounded-md border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge value={section.review_status} />
          {section.edited_by_user ? <StatusBadge value="edited" /> : null}
          {openCommentCount ? <StatusBadge value={`${openCommentCount} open comments`} /> : null}
          <span className="text-xs text-slate-500">Section v{section.version}</span>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <button type="button" onClick={onApprove} disabled={busyKey === `approve-${section.section_id}`} className="editor-button">Approve section</button>
          <button type="button" onClick={onMoveUp} disabled={index === 0} className="editor-button">Up</button>
          <button type="button" onClick={onMoveDown} disabled={index === total - 1} className="editor-button">Down</button>
          <button type="button" onClick={onDuplicate} disabled={busyKey === `duplicate-${section.section_id}`} className="editor-button">Duplicate</button>
          <button type="button" onClick={onDelete} disabled={busyKey === `delete-${section.section_id}`} className="editor-button-danger">Delete</button>
        </div>
      </div>

      <label className="mt-4 block text-xs font-medium text-slate-500">Title</label>
      <input
        value={section.title}
        onChange={(event) => onChange({ ...section, title: event.target.value })}
        className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500"
      />

      <label className="mt-4 block text-xs font-medium text-slate-500">Content</label>
      <textarea
        value={section.content}
        onChange={(event) => onChange({ ...section, content: event.target.value })}
        className="mt-1 min-h-44 w-full rounded-md border border-slate-300 px-3 py-2 text-sm leading-6 outline-none focus:border-slate-500"
      />

      <label className="mt-4 block text-xs font-medium text-slate-500">Reason for requested changes</label>
      <textarea
        value={changeReason}
        onChange={(event) => setChangeReason(event.target.value)}
        placeholder="Optional: explain what should change before approving this section."
        className="mt-1 min-h-20 w-full rounded-md border border-slate-300 px-3 py-2 text-sm leading-6 outline-none focus:border-slate-500"
      />

      <div className="mt-3 flex flex-wrap justify-end gap-2">
        <button
          type="button"
          onClick={() => {
            onRequestChanges(changeReason);
            setChangeReason("");
          }}
          disabled={busyKey === `changes-${section.section_id}`}
          className="rounded-md border border-amber-300 px-3 py-2 text-xs font-medium text-amber-700 hover:bg-amber-50 disabled:text-amber-300"
        >
          {busyKey === `changes-${section.section_id}` ? "Requesting..." : "Request changes"}
        </button>
        <button
          type="button"
          onClick={onSave}
          disabled={busyKey === `save-${section.section_id}`}
          className="rounded-md bg-slate-900 px-3 py-2 text-xs font-medium text-white disabled:bg-slate-400"
        >
          {busyKey === `save-${section.section_id}` ? "Saving..." : "Save section"}
        </button>
      </div>

      <ReportComments reportId={reportId} sectionId={section.section_id} initialComments={comments} />
    </section>
  );
}
