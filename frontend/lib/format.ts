const PARTS_FORMATTER = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  hour12: true,
  timeZone: "UTC"
});

function parseDate(value?: string | null) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatDateTime(value?: string | null): string {
  const date = parseDate(value);
  if (!date) return "Not set";
  const parts = dateParts(date);
  return `${parts.month} ${parts.day}, ${parts.year}, ${parts.hour}:${parts.minute} ${parts.dayPeriod}`;
}

export function formatDate(value?: string | null): string {
  const date = parseDate(value);
  if (!date) return "Not set";
  const parts = dateParts(date);
  return `${parts.month} ${parts.day}, ${parts.year}`;
}

export function formatTime(value?: string | null): string {
  const date = parseDate(value);
  if (!date) return "Not set";
  const parts = dateParts(date);
  return `${parts.hour}:${parts.minute} ${parts.dayPeriod}`;
}

function dateParts(date: Date) {
  const parts = Object.fromEntries(
    PARTS_FORMATTER.formatToParts(date)
      .filter((part) => part.type !== "literal")
      .map((part) => [part.type, part.value])
  );
  return {
    month: parts.month || "Jan",
    day: parts.day || "1",
    year: parts.year || "1970",
    hour: parts.hour || "12",
    minute: parts.minute || "00",
    dayPeriod: parts.dayPeriod || "AM"
  };
}
