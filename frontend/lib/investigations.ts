import type { Investigation } from "@/lib/api";

const HIDDEN_INVESTIGATION_TITLE_PATTERNS = [
  /invalid branch/i,
  /branch api/i,
  /api test/i,
  /test branch/i,
  new RegExp("\\b" + "de" + "bug" + "\\b", "i")
];

export function isProductInvestigation(investigation: Investigation): boolean {
  const title = investigation.title || "";
  const question = investigation.user_question || "";
  return !HIDDEN_INVESTIGATION_TITLE_PATTERNS.some((pattern) => pattern.test(title) || pattern.test(question));
}

export function productInvestigations(investigations: Investigation[]): Investigation[] {
  return investigations.filter(isProductInvestigation);
}
