"use client";

import { useMemo, useState } from "react";

type PreviewMode = "html" | "markdown" | "txt";

const modes: Array<{ id: PreviewMode; label: string }> = [
  { id: "html", label: "HTML" },
  { id: "markdown", label: "Markdown" },
  { id: "txt", label: "TXT" }
];

export function FinalReportPreview({
  html,
  markdown,
  txt
}: {
  html: string;
  markdown: string;
  txt: string;
}) {
  const [mode, setMode] = useState<PreviewMode>("html");
  const hasContent = Boolean(html || markdown || txt);
  const sections = useMemo(() => extractMarkdownSections(markdown), [markdown]);

  function scrollToSection(id: string) {
    setMode("markdown");
    window.setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
  }

  return (
    <section className="print-report rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="print-hidden flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-5 py-4">
        <div>
          <h2 className="font-semibold text-slate-950">Document reader</h2>
          <p className="text-xs text-slate-500">Switch between immutable export formats.</p>
        </div>
        <div className="inline-flex rounded-md border border-slate-200 bg-slate-50 p-1">
          {modes.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setMode(item.id)}
              className={`rounded px-3 py-1.5 text-xs font-medium ${
                mode === item.id
                  ? "bg-white text-slate-950 shadow-sm"
                  : "text-slate-500 hover:text-slate-800"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {sections.length ? (
        <div className="print-hidden border-b border-slate-200 px-5 py-3">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">Sections</div>
          <div className="flex flex-wrap gap-2">
            {sections.slice(0, 8).map((section) => (
              <button
                key={section.id}
                type="button"
                onClick={() => scrollToSection(section.id)}
                className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 hover:bg-slate-50"
              >
                {section.title}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="bg-slate-100/70 p-4 sm:p-6">
        {!hasContent ? (
          <div className="mx-auto max-w-3xl rounded-lg border border-dashed border-slate-300 bg-white p-8 text-sm text-slate-500">
            No final report content is available yet.
          </div>
        ) : mode === "html" ? (
          <iframe
            title="Final report HTML preview"
            className="final-report-html-preview mx-auto h-[780px] w-full max-w-[920px] rounded-lg border border-slate-200 bg-white shadow-sm"
            sandbox=""
            srcDoc={html || "<p>No HTML content available.</p>"}
          />
        ) : mode === "markdown" ? (
          <div className="mx-auto max-h-[780px] max-w-[920px] overflow-auto rounded-lg border border-slate-200 bg-white p-6 text-sm leading-7 text-slate-800 shadow-sm">
            {markdown ? renderMarkdownPreview(markdown) : "No Markdown content available."}
          </div>
        ) : (
          <pre className="mx-auto max-h-[780px] max-w-[920px] overflow-auto whitespace-pre-wrap rounded-lg border border-slate-200 bg-white p-6 text-sm leading-7 text-slate-800 shadow-sm">
            {txt || "No TXT content available."}
          </pre>
        )}
      </div>
    </section>
  );
}

function extractMarkdownSections(markdown: string) {
  return markdown
    .split("\n")
    .map((line) => line.match(/^#{1,3}\s+(.+)$/)?.[1]?.trim())
    .filter((title): title is string => Boolean(title))
    .slice(1)
    .map((title, index) => ({ title, id: `report-section-${slugify(title)}-${index}` }));
}

function renderMarkdownPreview(markdown: string) {
  let headingIndex = 0;
  return markdown.split("\n").map((line, index) => {
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const title = heading[2].trim();
      const id = heading[1].length <= 3 && headingIndex > 0
        ? `report-section-${slugify(title)}-${headingIndex - 1}`
        : undefined;
      headingIndex += 1;
      const className =
        heading[1].length === 1
          ? "mt-0 text-2xl font-semibold text-slate-950"
          : "mt-6 text-lg font-semibold text-slate-950";
      return (
        <h3 key={index} id={id} className={className}>
          {title}
        </h3>
      );
    }
    if (!line.trim()) {
      return <div key={index} className="h-3" />;
    }
    return <p key={index} className="whitespace-pre-wrap font-mono text-[13px] leading-7">{line}</p>;
  });
}

function slugify(value: string) {
  return (
    value
      .toLowerCase()
      .replace(/[^a-z0-9а-яё]+/gi, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 48) || "section"
  );
}
