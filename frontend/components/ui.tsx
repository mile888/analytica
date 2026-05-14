export function StatusBadge({ value }: { value: string }) {
  const tone =
    value === "failed" || value === "error" || value === "revoked"
      ? "border-red-200 bg-red-50 text-red-700"
      : value === "completed" || value === "verified" || value === "approved" || value === "final"
        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
        : "border-slate-200 bg-slate-50 text-slate-700";
  return (
    <span className={`inline-flex rounded-md border px-2 py-1 text-xs font-medium ${tone}`}>
      {value}
    </span>
  );
}

export function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`rounded-lg border border-slate-200 bg-white p-4 shadow-sm ${className}`}>{children}</section>;
}

export function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-500">{children}</div>;
}

export function formatDate(value?: string | null) {
  if (!value) return "Not set";
  return new Intl.DateTimeFormat("en", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}

export function stageLabel(value?: string | null) {
  if (!value) return "Unknown";
  return value
    .split("_")
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(" ");
}
