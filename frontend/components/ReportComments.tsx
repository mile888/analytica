"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import {
  ReportComment,
  addReportComment,
  deleteReportComment,
  promoteReportCommentToMemory,
  resolveReportComment
} from "@/lib/api";
import { StatusBadge, formatDate } from "@/components/ui";

export function ReportComments({
  reportId,
  sectionId,
  initialComments
}: {
  reportId: string;
  sectionId: string;
  initialComments: ReportComment[];
}) {
  const router = useRouter();
  const [comments, setComments] = useState(initialComments);
  const [text, setText] = useState("");
  const [showResolved, setShowResolved] = useState(false);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setComments(initialComments);
  }, [initialComments]);

  const visibleComments = useMemo(
    () => comments.filter((comment) => showResolved || comment.status !== "resolved"),
    [comments, showResolved]
  );
  const openCount = comments.filter((comment) => comment.status === "open").length;
  const resolvedCount = comments.length - openCount;

  async function addComment() {
    const trimmed = text.trim();
    if (!trimmed) return;
    setBusyKey("add-comment");
    setError(null);
    try {
      const created = await addReportComment(reportId, {
        section_id: sectionId,
        text: trimmed,
        author: "reviewer"
      });
      setComments((current) => [...current, created]);
      setText("");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not add comment");
    } finally {
      setBusyKey(null);
    }
  }

  async function resolveComment(commentId: string) {
    setBusyKey(`resolve-${commentId}`);
    setError(null);
    try {
      const resolved = await resolveReportComment(reportId, commentId);
      setComments((current) => current.map((comment) => (comment.comment_id === commentId ? resolved : comment)));
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not resolve comment");
    } finally {
      setBusyKey(null);
    }
  }

  async function removeComment(commentId: string) {
    setBusyKey(`delete-${commentId}`);
    setError(null);
    try {
      await deleteReportComment(reportId, commentId);
      setComments((current) => current.filter((comment) => comment.comment_id !== commentId));
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete comment");
    } finally {
      setBusyKey(null);
    }
  }

  async function promoteComment(commentId: string) {
    setBusyKey(`promote-${commentId}`);
    setError(null);
    try {
      await promoteReportCommentToMemory(reportId, commentId);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not promote comment");
    } finally {
      setBusyKey(null);
    }
  }

  return (
    <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Section comments</h4>
          <p className="mt-1 text-xs text-slate-500">
            {openCount} open · {resolvedCount} resolved
          </p>
        </div>
        {resolvedCount ? (
          <button
            type="button"
            onClick={() => setShowResolved((current) => !current)}
            className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-white"
          >
            {showResolved ? "Hide resolved" : "Show resolved"}
          </button>
        ) : null}
      </div>

      <div className="mt-3 space-y-2">
        {visibleComments.length ? (
          visibleComments.map((comment) => (
            <article key={comment.comment_id} className="rounded-md border border-slate-200 bg-white p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge value={comment.status} />
                  <span className="text-xs text-slate-500">
                    {comment.author || "reviewer"} · {formatDate(comment.created_at)}
                  </span>
                </div>
                <div className="flex gap-2">
                  {comment.status === "open" ? (
                    <button
                      type="button"
                      onClick={() => resolveComment(comment.comment_id)}
                      disabled={busyKey === `resolve-${comment.comment_id}`}
                      className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:text-slate-400"
                    >
                      Resolve
                    </button>
                  ) : null}
                  <button
                    type="button"
                    onClick={() => promoteComment(comment.comment_id)}
                    disabled={busyKey === `promote-${comment.comment_id}`}
                    className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:text-slate-400"
                  >
                    Promote
                  </button>
                  <button
                    type="button"
                    onClick={() => removeComment(comment.comment_id)}
                    disabled={busyKey === `delete-${comment.comment_id}`}
                    className="rounded-md border border-red-200 px-2 py-1 text-xs font-medium text-red-600 hover:bg-red-50 disabled:text-red-300"
                  >
                    Delete
                  </button>
                </div>
              </div>
              <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-700">{comment.text}</p>
            </article>
          ))
        ) : (
          <p className="rounded-md border border-dashed border-slate-200 bg-white p-3 text-sm text-slate-500">
            No section comments yet.
          </p>
        )}
      </div>

      <label className="mt-3 block text-xs font-medium text-slate-500">Add review comment</label>
      <textarea
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="Explain what should be clarified or changed..."
        className="mt-1 min-h-20 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-slate-500"
      />
      <div className="mt-2 flex justify-end">
        <button
          type="button"
          onClick={addComment}
          disabled={!text.trim() || busyKey === "add-comment"}
          className="rounded-md bg-slate-900 px-3 py-2 text-xs font-medium text-white disabled:bg-slate-400"
        >
          {busyKey === "add-comment" ? "Adding..." : "Add comment"}
        </button>
      </div>
      {error ? <p className="mt-2 text-xs text-red-600">{error}</p> : null}
    </div>
  );
}
