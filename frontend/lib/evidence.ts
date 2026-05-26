export type EvidenceType = "artifact" | "data_source" | "memory_item" | "report_section" | "report" | "unknown";

export interface EvidenceReference {
  type: EvidenceType;
  id: string;
  label: string;
  href?: string | null;
  metadata: Record<string, unknown>;
  key: string;
}

function normalizeEvidenceType(value: unknown): EvidenceType {
  if (
    value === "artifact" ||
    value === "data_source" ||
    value === "memory_item" ||
    value === "report_section" ||
    value === "report"
  ) {
    return value;
  }
  return "unknown";
}

export function normalizeEvidenceReference(value: unknown): EvidenceReference | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const id = typeof raw.id === "string" ? raw.id : "";
  if (!id) return null;
  const type = normalizeEvidenceType(raw.type);
  const label = typeof raw.label === "string" && raw.label.trim() ? raw.label : id;
  const href = typeof raw.href === "string" ? raw.href : null;
  const metadata = raw.metadata && typeof raw.metadata === "object" ? raw.metadata as Record<string, unknown> : {};
  return {
    type,
    id,
    label,
    href,
    metadata,
    key: `${type}:${id}`
  };
}

export function normalizeEvidenceReferences(values: unknown): EvidenceReference[] {
  if (!Array.isArray(values)) return [];
  const seen = new Set<string>();
  const references: EvidenceReference[] = [];
  for (const value of values) {
    const reference = normalizeEvidenceReference(value);
    if (reference && !seen.has(reference.key)) {
      seen.add(reference.key);
      references.push(reference);
    }
  }
  return references;
}
