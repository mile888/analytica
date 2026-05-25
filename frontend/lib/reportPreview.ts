import type { ShareableReport } from "./api";

export type ReportArtifactSnapshot = Record<string, unknown>;
export type ReportTableRow = Record<string, string>;
export type ReportTranscriptItem = {
  index: number;
  question: string;
  answer: string;
};

export function reportVisualArtifactSnapshots(section: ShareableReport["sections"][number]): ReportArtifactSnapshot[] {
  return Array.isArray(section.metadata?.artifact_snapshots) ? (section.metadata.artifact_snapshots as ReportArtifactSnapshot[]) : [];
}

export function reportPreviewRows(item: ReportArtifactSnapshot) {
  const content = item.content && typeof item.content === "object" ? (item.content as Record<string, unknown>) : {};
  const rows = Array.isArray(content.rows) ? content.rows.slice(0, 8) : [];
  const xKey = typeof content.x === "string" ? content.x : typeof content.dimension === "string" ? content.dimension : "label";
  const yKey = typeof content.y === "string" ? content.y : typeof content.metric === "string" ? content.metric : "value";

  return rows.map((row) => {
    const record = row && typeof row === "object" ? (row as Record<string, unknown>) : {};
    const rawValue = Number(record[yKey] ?? record.value ?? 0);
    return {
      label: String(record[xKey] ?? record.label ?? record.name ?? "Item"),
      value: Number.isFinite(rawValue) ? rawValue : 0
    };
  });
}

export function reportTableRows(item: ReportArtifactSnapshot, limit = 12): ReportTableRow[] {
  const content = item.content;
  const rows = Array.isArray(content)
    ? content
    : content && typeof content === "object"
      ? Array.isArray((content as Record<string, unknown>).rows)
        ? ((content as Record<string, unknown>).rows as unknown[])
        : Array.isArray((content as Record<string, unknown>).data)
          ? ((content as Record<string, unknown>).data as unknown[])
          : []
      : [];

  return rows.slice(0, limit).map((row) => {
    const record = row && typeof row === "object" ? (row as Record<string, unknown>) : {};
    return Object.fromEntries(
      Object.entries(record).map(([key, value]) => [
        key,
        value == null ? "" : typeof value === "object" ? JSON.stringify(value) : String(value)
      ])
    );
  });
}

export function reportTableColumns(rows: ReportTableRow[], limit = 8): string[] {
  const columns: string[] = [];
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!columns.includes(key)) columns.push(key);
      if (columns.length >= limit) return columns;
    }
  }
  return columns;
}

export function reportTranscriptItems(content: string): ReportTranscriptItem[] {
  const blocks = content.split(/\n\s*\n/).map((block) => block.trim()).filter(Boolean);
  const items: ReportTranscriptItem[] = [];

  for (const block of blocks) {
    const paired = block.match(/^(\d+)\.\s*Question:\s*([\s\S]*?)(?:\n\s*Answer:\s*([\s\S]*))?$/i);
    if (paired) {
      items.push({ index: Number(paired[1]), question: paired[2].trim(), answer: (paired[3] || "").trim() });
      continue;
    }
    const question = block.match(/^Question\s+(\d+):\s*([\s\S]*)$/i);
    if (question) {
      items.push({ index: Number(question[1]), question: question[2].trim(), answer: "" });
      continue;
    }
    const answer = block.match(/^Answer\s+(\d+):\s*([\s\S]*)$/i);
    if (answer) {
      const index = Number(answer[1]);
      const existing = items.find((item) => item.index === index);
      if (existing) existing.answer = answer[2].trim();
      else items.push({ index, question: "", answer: answer[2].trim() });
    }
  }

  return items.sort((a, b) => a.index - b.index);
}
