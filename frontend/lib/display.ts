import type { Artifact, DataSource, Finding, InvestigationMessage, InvestigationRun } from "@/lib/api";

const RAW_PATTERNS = [
  /value_preview=/i,
  /shape=\(/i,
  /columns=\[/i,
  /preview=/i,
  /row_count=/i,
  /truncated=/i,
  /table=/i,
  /type=dict/i,
  /^\{[\s\S]*\}$/,
  /^\[[\s\S]*\]$/
];

const TECHNICAL_ARTIFACT_TYPES = new Set(["python_code", "sql", "validation", "trace", "unknown"]);

export interface UserFacingFinding {
  title: string;
  summary: string;
  isTechnicalSummary: boolean;
  confidenceLabel: string;
  evidenceStrength: string;
  businessImpact: string;
  confidenceReason: string;
  evidenceReason: string;
  limitation: string;
  recommendedValidation: string;
  businessImplication: string;
  hypothesis: string;
  uncertaintyNote: string;
  possibleDrivers: string[];
  validationQuestions: string[];
  analysisType: string;
  recommendedNextStep: string;
  supportingOutputCount: number;
}

export interface UserFacingArtifact {
  title: string;
  typeLabel: string;
  description: string;
  isTechnical: boolean;
}

export interface UserFacingMessage {
  label: string;
  content: string;
  tone: "user" | "assistant" | "system" | "error";
  status?: string;
}

export interface UserFacingRun {
  label: string;
  summary: string;
  technicalId: string;
}

export type DatasetExecutionStatus = "Ready for analysis" | "Loading" | "Execution unavailable" | "Reattach required" | "Parsing failed";

export function datasetExecutionStatus(source: DataSource): DatasetExecutionStatus {
  const metadata = source.metadata || {};
  const context = isPlainObject(metadata.execution_context) ? metadata.execution_context : metadata;
  if (source.status === "error") return "Parsing failed";
  if (context.executable_available === true) return "Ready for analysis";
  if (context.unavailable_reason === "raw_rows_unavailable" || context.executable_available === false) return "Reattach required";
  if (context.storage_reference || context.dataset_runtime_reference) return "Loading";
  return "Execution unavailable";
}

export function formatFindingForUser(finding: Finding): UserFacingFinding {
  const metadata = finding.metadata || {};
  const conclusion = metadataText(metadata, "conclusion");
  const title = cleanText(finding.title) || truncateText(conclusion, 80) || "Finding";
  const text = cleanText(finding.text);
  const summary = summarizeTechnicalText(conclusion || text) || conclusion || text || "This finding needs a human-readable summary.";
  const recommendedValidation = metadataText(metadata, "recommended_validation");
  return {
    title: summarizeTechnicalText(title) || title,
    summary: truncateText(summary, 280),
    isTechnicalSummary: looksTechnical(text),
    confidenceLabel: titleCase(metadataText(metadata, "confidence_level")) || confidenceLabel(finding.confidence),
    evidenceStrength: evidenceStrengthLabel(metadataText(metadata, "evidence_strength")) || evidenceStrength(finding),
    businessImpact: titleCase(metadataText(metadata, "business_impact")) || "Medium",
    confidenceReason: metadataText(metadata, "confidence_reason"),
    evidenceReason: metadataText(metadata, "evidence_reason"),
    limitation: metadataText(metadata, "limitation"),
    recommendedValidation,
    businessImplication: metadataText(metadata, "business_implication"),
    hypothesis: metadataList(metadata, "hypotheses")[0] || "",
    uncertaintyNote: metadataList(metadata, "uncertainty_notes")[0] || "",
    possibleDrivers: metadataList(metadata, "possible_drivers").slice(0, 4),
    validationQuestions: metadataList(metadata, "validation_questions").slice(0, 3),
    analysisType: analysisTypeLabel(metadataText(metadata, "analysis_type")),
    recommendedNextStep: recommendedValidation || metadataText(metadata, "recommended_next_step"),
    supportingOutputCount: supportingOutputCount(finding)
  };
}

export function isRealAnalyticalFinding(finding: Finding): boolean {
  const metadata = finding.metadata || {};
  const analysisType = metadataText(metadata, "analysis_type").toLowerCase();
  if (["overview", "profile", "suggestion"].includes(analysisType)) return false;
  const text = `${cleanText(finding.title)} ${cleanText(finding.text)} ${metadataText(metadata, "conclusion")}`.toLowerCase();
  const profileMarkers = [
    "dataset has enough structure",
    "good candidates for quantitative analysis",
    "can anchor quantitative analysis",
    "can explain differences between groups",
    "useful for segmentation",
    "supports trend analysis",
    "should not be treated as metrics",
    "candidate metric"
  ];
  return !profileMarkers.some((marker) => text.includes(marker));
}

export function isHighConfidenceKeyFinding(finding: Finding): boolean {
  if (!isRealAnalyticalFinding(finding)) return false;
  const metadata = finding.metadata || {};
  const confidence = (
    metadataText(metadata, "confidence_level") ||
    metadataText(metadata, "confidence") ||
    confidenceLabel(finding.confidence)
  ).toLowerCase();
  if (confidence !== "high") return false;
  const analysisType = metadataText(metadata, "analysis_type").toLowerCase();
  const text = `${cleanText(finding.title)} ${cleanText(finding.text)} ${metadataText(metadata, "conclusion")}`.toLowerCase();
  const forbiddenTypes = new Set([
    "clarification_needed",
    "execution_context_unavailable",
    "fallback",
    "error",
    "non_analytical",
    "validation",
    "limitation",
    "profile",
    "overview",
    "suggestion"
  ]);
  if (forbiddenTypes.has(analysisType)) return false;
  const forbiddenMarkers = [
    "i cannot explain this chart",
    "exact artifact",
    "not available in the current artifact payload",
    "no active metric/dimension context",
    "execution context is unavailable",
    "raw rows are not attached",
    "this answer uses the latest saved analytical result",
    "needs validation confidence",
    "limitation evidence is stronger",
    "limitation `",
    "limitation no ",
    "could not complete",
    "could not compute",
    "no written analytical answer"
  ];
  if (forbiddenMarkers.some((marker) => text.includes(marker))) return false;
  const substantiveMarkers = [
    " led by ",
    " ranks ",
    " distribution ",
    " compares ",
    " comparison ",
    " median ",
    " mean ",
    " range ",
    " highest ",
    " lowest ",
    " contributes ",
    " created ",
    " tested against ",
    " correlation ",
    " total ",
    " average ",
    " varies ",
    " outlier",
    " missing ",
    " duplicates"
  ];
  return substantiveMarkers.some((marker) => text.includes(marker));
}

export function formatArtifactForUser(artifact: Artifact): UserFacingArtifact {
  const type = artifact.type || artifact.artifact_type || "unknown";
  const typeLabel = artifactTypeLabel(type);
  const title = cleanText(artifact.title) || typeLabel;
  return {
    title: truncateText(title, 80),
    typeLabel,
    description: artifactDescription(artifact),
    isTechnical: artifact.visibility !== "user" || TECHNICAL_ARTIFACT_TYPES.has(type)
  };
}

export function formatMessageForUser(message: InvestigationMessage): UserFacingMessage {
  const messageType = getMessageType(message);
  const content = assistantContentFallback(message);
  if (messageType === "question") {
    return { label: "You", content, tone: "user" };
  }
  if (messageType === "follow_up") {
    return { label: "You", content, tone: "user" };
  }
  if (messageType === "run_summary" || messageType === "answer") {
    return { label: "Analytica", content, tone: "assistant" };
  }
  if (messageType === "error" || message.role === "system") {
    return {
      label: messageType === "error" ? "Analytica" : "Workspace note",
      content: messageType === "error" ? friendlyErrorMessage(message.content) : content,
      tone: messageType === "error" ? "error" : "system",
      status: messageType === "error" ? "Could not complete" : undefined
    };
  }
  return { label: "Note", content, tone: message.role === "assistant" ? "assistant" : "user" };
}

export function assistantAnswerForRun(
  run: InvestigationRun | undefined,
  findingCount: number,
  artifactCount: number
): UserFacingMessage | null {
  if (!run) return null;
  if (run.status === "running" || run.status === "queued") {
    return {
      label: "Analytica",
      tone: "assistant",
      content: "I’m analyzing this question now and looking for concrete patterns, comparisons, and evidence.",
      status: "Thinking"
    };
  }
  if (run.status === "failed") {
    return {
      label: "Analytica",
      tone: "error",
      content: friendlyErrorMessage(run.error_message || "The analysis could not be completed."),
      status: "Could not complete"
    };
  }
  if (run.status === "completed") {
    return null;
  }
  return null;
}

export function formatRunForUser(run: InvestigationRun): UserFacingRun {
  const artifactCount = run.artifact_ids?.length || 0;
  const reportCount = run.report_ids?.length || 0;
  const sourceCount = run.data_source_ids?.length || 0;
  const pieces = [
    `${artifactCount} output${artifactCount === 1 ? "" : "s"}`,
    `${reportCount} report${reportCount === 1 ? "" : "s"}`,
    sourceCount ? `${sourceCount} dataset${sourceCount === 1 ? "" : "s"}` : null
  ].filter(Boolean);
  return {
    label: "Analytical output",
    summary: pieces.join(" · ") || "No outputs yet.",
    technicalId: run.run_id
  };
}

export function truncateText(value: string | null | undefined, maxLength = 240): string {
  const text = cleanText(value || "");
  if (text.length <= maxLength) return text;
  return `${text.slice(0, Math.max(0, maxLength - 1)).trim()}…`;
}

export function cleanText(value: string): string {
  return String(value || "").replace(/\s+/g, " ").trim();
}

export function looksTechnical(value: string | null | undefined): boolean {
  const text = cleanText(value || "");
  return RAW_PATTERNS.some((pattern) => pattern.test(text));
}

export function dedupeConsecutiveMessages(messages: InvestigationMessage[]): InvestigationMessage[] {
  const deduped: InvestigationMessage[] = [];
  const assistantResponses = new Set<string>();
  for (const message of messages) {
    const previous = deduped[deduped.length - 1];
    const messageType = getMessageType(message);
    const responseTo = typeof message.metadata?.response_to_message_id === "string" ? message.metadata.response_to_message_id : "";
    const responseKey = responseTo && (message.role === "assistant" || messageType === "run_summary" || messageType === "answer")
      ? `${responseTo}:${messageType}`
      : "";
    if (responseKey && assistantResponses.has(responseKey)) {
      continue;
    }
    if (
      previous &&
      previous.role === message.role &&
      getMessageType(previous) === messageType &&
      semanticallySameMessage(previous.content, message.content)
    ) {
      continue;
    }
    if (responseKey) assistantResponses.add(responseKey);
    deduped.push(message);
  }
  return deduped;
}

function assistantContentFallback(message: InvestigationMessage): string {
  const messageType = getMessageType(message);
  const summarized = summarizeTechnicalText(message.content);
  const cleaned = openEndedAssistantText(summarized || cleanText(message.content));
  if (!cleaned && (messageType === "run_summary" || messageType === "answer")) {
    return "No written analytical answer was returned for this run.";
  }
  if (messageType === "run_summary" && looksTooGenericOrTechnical(cleaned)) {
    return "No clean written analytical answer was returned. Ask a focused question about a metric, segment, chart, anomaly, trend, or evidence gap.";
  }
  return cleaned || "No answer generated yet.";
}

function openEndedAssistantText(value: string): string {
  const text = cleanText(value);
  const normalized = text.toLowerCase();
  if (
    normalized === "done" ||
    /finished the analysis|analysis completed|the analysis is complete|main takeaway is available above|available above|i found new analytical material|keep exploring|investigation has been updated|continue with another follow-up/i.test(text)
  ) {
    return "No written analytical answer was returned for this run.";
  }
  return text;
}

function getMessageType(message: InvestigationMessage): InvestigationMessage["message_type"] {
  return message.message_type || message.type || "note";
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function friendlyErrorMessage(value: string | null | undefined): string {
  const text = cleanText(value || "");
  if (!text) return "I could not complete this analysis. Try again or check your model/API settings.";
  if (/APIConnectionError|connection error/i.test(text)) {
    return "I could not complete this analysis because the AI service connection failed. Try again or check your model/API settings.";
  }
  if (/OperationalError|syntax error/i.test(text)) {
    return "I could not complete this analysis because one generated query was invalid. Try again with a simpler question.";
  }
  if (/Unsupported table type|to_pandas/i.test(text)) {
    return "I could not complete this analysis because the dataset could not be converted into a table. Try re-uploading the CSV or selecting another data source.";
  }
  return `I could not complete this analysis. Reason: ${truncateText(text.replace(/^RuntimeError:\s*/i, ""), 180)}`;
}

function semanticallySameMessage(first: string, second: string): boolean {
  const left = semanticSignature(first);
  const right = semanticSignature(second);
  if (!left || !right) return false;
  if (left === right) return true;
  const leftTokens = new Set(left.split(" "));
  const rightTokens = new Set(right.split(" "));
  const union = new Set([...leftTokens, ...rightTokens]);
  let intersection = 0;
  leftTokens.forEach((token) => {
    if (rightTokens.has(token)) intersection += 1;
  });
  return intersection / Math.max(union.size, 1) >= 0.9 && intersection >= 10;
}

function semanticSignature(value: string): string {
  return cleanText(value)
    .toLowerCase()
    .replace(/[^\p{L}\p{N}_\s]/gu, " ")
    .split(/\s+/)
    .filter((token) => !["the", "and", "or", "to", "of", "in", "by", "for", "with", "это", "данных"].includes(token))
    .join(" ");
}

function looksTooGenericOrTechnical(value: string): boolean {
  const text = cleanText(value);
  return !text || looksTechnical(text) || /^run completed/i.test(text) || /^created \d+/i.test(text);
}

function summarizeTechnicalText(value: string): string | null {
  const text = cleanText(value);
  if (!looksTechnical(text)) return null;

  const outlierCount = text.match(/['"]?outlier_count['"]?\s*[:=]\s*(\d+)/i)?.[1];
  const outlierColumn = text.match(/['"]?(?:column|metric|field)['"]?\s*[:=]\s*['"]?([A-Za-z0-9_ .-]+)/i)?.[1];
  if (outlierCount) {
    return `Detected ${outlierCount} possible outliers${outlierColumn ? ` in ${outlierColumn.trim()}` : ""}.`;
  }

  const dataframeShape = text.match(/dataframe\s+shape=\((\d+),\s*(\d+)\)/i);
  if (dataframeShape) {
    return `Generated a table preview with ${dataframeShape[1]} rows and ${dataframeShape[2]} columns.`;
  }

  const sqlRows = text.match(/SQL result.*?row_count=(\d+)/i)?.[1] || text.match(/row_count['"]?\s*[:=]\s*(\d+)/i)?.[1];
  if (sqlRows) {
    return `Generated a table with ${sqlRows} rows. Preview is available in the Outputs area.`;
  }

  if (/scalar\s+type=dict/i.test(text) || /type=dict/i.test(text)) {
    return "Generated a structured analytical result. Details are available in advanced details.";
  }

  if (/columns=\[/i.test(text) || /preview=/i.test(text)) {
    return "Generated a data preview. The linked artifact contains the details.";
  }

  return "Generated a technical result. Details are available in advanced details.";
}

function artifactTypeLabel(type: string): string {
  const normalized = type.replaceAll("_", " ");
  const labels: Record<string, string> = {
    table: "Table",
    chart: "Chart",
    report: "Report",
    text: "Summary",
    python_code: "Code",
    sql: "SQL",
    validation: "Check",
    trace: "Trace",
    unknown: "Artifact"
  };
  return labels[type] || normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function artifactDescription(artifact: Artifact): string {
  const type = artifact.type || artifact.artifact_type || "unknown";
  if (type === "table") return "A generated table is available for review.";
  if (type === "chart") return "A generated visualization is available for review.";
  if (type === "report") return "A report-style output created from this investigation.";
  if (type === "text") return "A short textual summary from the analysis.";
  if (type === "python_code") return "Generated Python code, hidden from the main report view.";
  if (type === "sql") return "Generated SQL query, available as technical detail.";
  if (type === "validation") return "Validation or execution check from the run.";
  return "Output created during the investigation.";
}

function metadataText(metadata: Record<string, unknown> | undefined, key: string): string {
  const value = metadata?.[key];
  return typeof value === "string" ? value : "";
}

function metadataList(metadata: Record<string, unknown> | undefined, key: string): string[] {
  const value = metadata?.[key];
  if (!Array.isArray(value)) return [];
  return value.map((item) => cleanText(String(item))).filter(Boolean);
}

function confidenceLabel(confidence: number | null | undefined): string {
  if (typeof confidence !== "number") return "Medium";
  if (confidence >= 0.75) return "High";
  if (confidence >= 0.5) return "Medium";
  return "Low";
}

function evidenceStrength(finding: Finding): string {
  const evidenceCount = (finding.evidence_artifact_ids || []).length;
  if (evidenceCount >= 2) return "Strong";
  if (evidenceCount === 1) return "Moderate";
  return "Needs evidence";
}

function evidenceStrengthLabel(value: string): string {
  const normalized = value.toLowerCase();
  if (normalized === "high") return "High";
  if (normalized === "medium") return "Medium";
  if (normalized === "low") return "Low";
  return titleCase(value);
}

function analysisTypeLabel(value: string): string {
  const labels: Record<string, string> = {
    grouped_metric: "Grouped analysis",
    outlier: "Outlier analysis",
    correlation: "Correlation",
    trend: "Trend",
    data_quality: "Data quality",
    overview: "Overview"
  };
  return labels[value] || titleCase(value.replaceAll("_", " "));
}

function titleCase(value: string): string {
  if (!value) return "";
  return value
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

function supportingOutputCount(finding: Finding): number {
  const raw = finding.metadata?.supporting_evidence_count ?? finding.metadata?.linked_output_count;
  if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  return (finding.evidence_artifact_ids || []).length;
}
