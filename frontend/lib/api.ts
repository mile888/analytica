const DEFAULT_INTERNAL_API_URL = "http://backend:8000";
const SAME_ORIGIN_API_BASE = "/api";

function getPublicApiBaseUrl() {
  return process.env.NEXT_PUBLIC_API_URL || "";
}

function getApiBaseUrl() {
  const publicUrl = getPublicApiBaseUrl();
  if (typeof window === "undefined") {
    return process.env.INTERNAL_API_URL || publicUrl || DEFAULT_INTERNAL_API_URL;
  }
  return publicUrl || SAME_ORIGIN_API_BASE;
}

type FetchOptions = RequestInit & { query?: Record<string, string | number | boolean | undefined | null> };

export class ApiError extends Error {
  status: number;
  body: string;

  constructor(status: number, body: string) {
    super(`API ${status}: ${body || "Request failed"}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

export function buildApiUrl(
  path: string,
  query: Record<string, string | number | boolean | undefined | null> = {},
  baseUrl = getApiBaseUrl()
) {
  if (baseUrl.startsWith("/")) {
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;
    const url = new URL(`${baseUrl}${normalizedPath}`, "http://same-origin.local");
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }
    return `${url.pathname}${url.search}`;
  }
  const url = new URL(path, baseUrl);
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

export function reportPdfDownloadUrl(reportId: string) {
  return buildApiUrl(`/reports/${reportId}/download/pdf`, {}, getPublicApiBaseUrl() || SAME_ORIGIN_API_BASE);
}

export async function apiFetch<T>(path: string, options: FetchOptions = {}): Promise<T> {
  const url = buildApiUrl(path, options.query);
  const response = await fetch(url.toString(), {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    cache: "no-store"
  });
  if (!response.ok) {
    throw new ApiError(response.status, await response.text());
  }
  return response.json() as Promise<T>;
}

export interface Artifact {
  artifact_id: string;
  title: string;
  artifact_type?: string;
  type?: string;
  visibility: string;
  pinned?: boolean;
  run_id?: string | null;
  content?: unknown;
  metadata?: Record<string, unknown>;
}

export interface Finding {
  finding_id: string;
  title: string;
  text: string;
  status: string;
  confidence?: number | null;
  evidence_artifact_ids?: string[];
  metadata?: Record<string, unknown>;
}

export interface Investigation {
  investigation_id: string;
  title: string;
  user_question: string;
  status: string;
  created_at: string;
  updated_at: string;
  linked_data_source_ids: string[];
  data_sources?: string[];
  findings: Finding[];
  artifacts: Artifact[];
  runs?: unknown[];
}

export interface InvestigationBranch {
  branch_id: string;
  title: string;
  branch_type: string;
  intent_type?: string;
  dataset_scope?: string | null;
  dataset_id?: string | null;
  dataset_ids?: string[];
  active_artifact_id?: string | null;
  metric?: string | null;
  dimension?: string | null;
  filters?: Array<{ column: string; operator: string; value: string }>;
  chart_type?: string | null;
  artifact_count: number;
  finding_count: number;
  updated_at: string;
  is_active: boolean;
  subtitle?: string;
}

export interface InvestigationRun {
  run_id: string;
  investigation_id: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  status: string;
  current_stage: string;
  data_source_ids: string[];
  run_context_summary: Record<string, unknown>;
  error_message?: string | null;
  artifact_ids: string[];
  report_ids: string[];
}

export interface RunEvent {
  event_id: string;
  run_id: string;
  investigation_id: string;
  event_type: string;
  stage?: string | null;
  message: string;
  created_at: string;
  severity: "info" | "warning" | "error";
  metadata: Record<string, unknown>;
}

export interface EventStreamResponse {
  events: RunEvent[];
  next_cursor?: string | null;
  has_more: boolean;
}

export interface RunInvestigationPayload {
  data_source_ids?: string[];
  force_refresh_context?: boolean;
  message_id?: string | null;
  analysis_mode?: "exploration" | "validation" | "executive" | "data_quality";
}

export interface InvestigationMessage {
  message_id: string;
  investigation_id: string;
  run_id?: string | null;
  role: "user" | "assistant" | "system";
  message_type: "question" | "follow_up" | "answer" | "note" | "run_summary" | "error";
  type?: "question" | "follow_up" | "answer" | "note" | "run_summary" | "error";
  content: string;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface InvestigationMemoryItem {
  memory_id: string;
  investigation_id: string;
  memory_type: "assumption" | "open_question" | "decision" | "risk" | "milestone";
  title: string;
  content: string;
  status: "active" | "resolved" | "archived";
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface WorkflowGuidance {
  workflow: {
    workflow_id: string;
    title: string;
    description: string;
    stages: Array<{ stage_id: string; title: string; description: string }>;
  };
  progress: Array<{ stage_id: string; status: string; message: string; next_step?: string }>;
  review: {
    status: string;
    evidence_quality: string;
    summary: string;
    suggested_next_step?: string;
    checkpoints: Array<{ checkpoint_id: string; title: string; status: string; message: string; severity: string }>;
    feedback: Array<{ feedback_id: string; title: string; message: string; severity: string; related_finding_id?: string | null }>;
  };
  validation_expectations: Array<{ expectation_id: string; title: string; description: string; severity: string }>;
  playbooks: Array<{
    playbook_id: string;
    title: string;
    workflow_id: string;
    analytical_sequence: string[];
    chart_priorities: string[];
    validation_questions: string[];
    report_guidance: string[];
  }>;
  report_standard: {
    standard_id: string;
    title: string;
    required_sections: string[];
    evidence_expectations: string[];
    tone_guidance: string;
    validation_guidance: string[];
  };
}

export interface CreateInvestigationMessagePayload {
  content: string;
  type?: "question" | "follow_up" | "answer" | "note" | "run_summary" | "error";
  role?: "user" | "assistant" | "system";
  metadata?: Record<string, unknown>;
}

export interface CreateDataSourcePayload {
  name: string;
  type: "csv" | "sqlite" | "postgres" | "duckdb" | "unknown";
  location?: string | null;
  description?: string | null;
}

export interface CreateInvestigationPayload {
  title?: string | null;
  question: string;
  data_source_ids?: string[];
  linked_data_source_ids?: string[];
}

export interface UploadCsvDataSourcePayload {
  file: File;
  name?: string | null;
  description?: string | null;
}

export interface DataSource {
  data_source_id: string;
  name: string;
  data_source_type: string;
  status: string;
  created_at: string;
  updated_at: string;
  location?: string | null;
  description?: string | null;
  tags: string[];
  linked_investigation_ids: string[];
  metadata?: Record<string, unknown>;
}

export interface DataSourceProfile {
  row_count: number;
  column_count: number;
  columns: Array<{
    name: string;
    dtype: string;
    nullable: boolean;
    unique_count?: number | null;
    sample_values: unknown[];
  }>;
  missing_summary: Record<string, number>;
  numeric_summary: Record<string, Record<string, unknown>>;
  categorical_summary: Record<string, Record<string, unknown>>;
  sampled_rows: Record<string, unknown>[];
}

export interface UploadCsvDataSourceResponse {
  data_source: DataSource;
  profile: DataSourceProfile | null;
  profile_status?: "complete" | "partial";
  warnings?: string[];
}

export interface DataSourceUsageContext {
  data_source_id: string;
  name: string;
  status: string;
  description?: string | null;
  schema_summary: {
    row_count?: number | null;
    column_count?: number | null;
    roles?: Record<string, string[]>;
  };
  column_summaries: Array<{
    name: string;
    dtype: string;
    inferred_role: string;
    notes: string[];
  }>;
  caveats: string[];
  linked_investigation_ids: string[];
  previous_questions: string[];
}

export interface DecisionMetadata {
  tags: string[];
  owner?: string | null;
  audience?: string | null;
  business_area?: string | null;
  decision_date?: string | null;
  decision_status: string;
  short_description?: string | null;
}

export interface FinalReportSnapshot {
  snapshot_id: string;
  report_id: string;
  investigation_id: string;
  report_version: number;
  title: string;
  created_at: string;
  created_by: string;
  status: string;
  markdown_content?: string;
  html_content?: string;
  txt_content?: string;
  readiness_snapshot: {
    is_ready?: boolean;
    blocking_count?: number;
    warning_count?: number;
    checks?: Array<Record<string, unknown>>;
  };
  approval_status: string;
  approved_at?: string | null;
  approved_by?: string | null;
  decision_metadata: DecisionMetadata;
  metadata: Record<string, unknown>;
}

export interface ReportSection {
  section_id: string;
  title: string;
  content: string;
  order: number;
  section_type?: string;
  source_branch_id?: string | null;
  source_artifact_ids?: string[];
  source_question_ids?: string[];
  artifact_ids: string[];
  edited_by_user: boolean;
  created_by: string;
  version: number;
  review_status: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface ShareableReport {
  report_id: string;
  investigation_id: string;
  title: string;
  template: "executive_summary" | "product_decision_memo" | "technical_appendix";
  status: string;
  version: number;
  previous_version_id?: string | null;
  is_latest: boolean;
  approval_status: string;
  reviewer_notes?: string;
  approved_at?: string | null;
  approved_by?: string | null;
  sections: ReportSection[];
  dataset_ids?: string[];
  branch_ids?: string[];
  included_question_ids?: string[];
  summary?: string;
  limitations?: string[];
  source_finding_ids: string[];
  source_artifact_ids: string[];
  include_technical: boolean;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface ReportReadinessResult {
  report_id: string;
  is_ready: boolean;
  blocking_count: number;
  warning_count: number;
  checks: Array<Record<string, unknown>>;
  generated_at: string;
}

export interface ReportComment {
  comment_id: string;
  report_id: string;
  section_id: string;
  text: string;
  status: "open" | "resolved";
  created_at: string;
  updated_at: string;
  resolved_at?: string | null;
  author: string;
  metadata: Record<string, unknown>;
}

export interface CreateShareableReportPayload {
  template: "executive_summary" | "product_decision_memo" | "technical_appendix";
  include_technical?: boolean;
}

export interface ReportSectionUpdatePayload {
  title?: string;
  content?: string;
}

export interface ReportSectionCreatePayload {
  title: string;
  content?: string;
  order?: number;
}

export function listInvestigations() {
  return apiFetch<Investigation[]>("/investigations");
}

export function createInvestigation(payload: CreateInvestigationPayload) {
  return apiFetch<Investigation>("/investigations", {
    method: "POST",
    body: JSON.stringify({
      title: payload.title || undefined,
      question: payload.question,
      data_source_ids: payload.data_source_ids || payload.linked_data_source_ids || []
    })
  });
}

export function getInvestigation(id: string) {
  return apiFetch<Investigation>(`/investigations/${id}`);
}

export function getInvestigationBranches(id: string) {
  return apiFetch<InvestigationBranch[]>(`/investigations/${id}/branches`);
}

export function activateInvestigationBranch(id: string, branchId: string) {
  return apiFetch<{ active_branch_id: string; branches: InvestigationBranch[] }>(`/investigations/${id}/branches/${encodeURIComponent(branchId)}/activate`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function deleteInvestigation(id: string) {
  return apiFetch<{ deleted: boolean; id: string }>(`/investigations/${id}`, {
    method: "DELETE"
  });
}

export function runInvestigation(id: string, payload: RunInvestigationPayload = {}) {
  return apiFetch<InvestigationRun>(`/investigations/${id}/run`, {
    method: "POST",
    body: JSON.stringify({
      data_source_ids: payload.data_source_ids || [],
      force_refresh_context: payload.force_refresh_context || false,
      message_id: payload.message_id || undefined,
      analysis_mode: payload.analysis_mode || "exploration"
    })
  });
}

export function listInvestigationMessages(id: string) {
  return apiFetch<InvestigationMessage[]>(`/investigations/${id}/messages`);
}

export function listInvestigationMemory(id: string) {
  return apiFetch<InvestigationMemoryItem[]>(`/investigations/${id}/memory`);
}

export function addInvestigationMemory(
  id: string,
  payload: { type: InvestigationMemoryItem["memory_type"]; content: string; title?: string; status?: InvestigationMemoryItem["status"] }
) {
  return apiFetch<InvestigationMemoryItem>(`/investigations/${id}/memory`, {
    method: "POST",
    body: JSON.stringify({
      type: payload.type,
      content: payload.content,
      title: payload.title || "",
      status: payload.status || "active"
    })
  });
}

export function updateInvestigationMemory(
  investigationId: string,
  memoryId: string,
  payload: { content?: string; title?: string; status?: InvestigationMemoryItem["status"] }
) {
  return apiFetch<InvestigationMemoryItem>(`/investigations/${investigationId}/memory/${memoryId}`, {
    method: "PATCH",
    body: JSON.stringify(payload)
  });
}

export function promoteFindingToMemory(
  investigationId: string,
  findingId: string,
  type: "assumption" | "risk" | "decision"
) {
  return apiFetch<InvestigationMemoryItem>(`/investigations/${investigationId}/findings/${findingId}/promote-memory`, {
    method: "POST",
    body: JSON.stringify({ type })
  });
}

export interface EvidenceLinkPayload {
  type: "report_section" | "artifact" | "data_source" | "memory_item";
  id: string;
  label?: string | null;
  href?: string | null;
}

export function linkFindingEvidence(investigationId: string, findingId: string, payload: EvidenceLinkPayload) {
  return apiFetch<Finding>(`/investigations/${investigationId}/findings/${findingId}/evidence`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function getInvestigationSuggestedQuestions(id: string, limit = 8) {
  return apiFetch<{ suggestions: string[] }>(`/investigations/${id}/suggested-questions`, {
    query: { limit }
  });
}

export function getInvestigationWorkflowGuidance(id: string) {
  return apiFetch<WorkflowGuidance>(`/investigations/${id}/workflow-guidance`);
}

export function createInvestigationMessage(id: string, payload: CreateInvestigationMessagePayload) {
  return apiFetch<InvestigationMessage>(`/investigations/${id}/messages`, {
    method: "POST",
    body: JSON.stringify({
      content: payload.content,
      type: payload.type || "follow_up",
      role: payload.role || "user",
      metadata: payload.metadata || {}
    })
  });
}

export function getInvestigationRun(runId: string) {
  return apiFetch<InvestigationRun>(`/investigation-runs/${runId}`);
}

export function getInvestigationRuns(id: string) {
  return apiFetch<InvestigationRun[]>(`/investigations/${id}/runs`);
}

export function getInvestigationEvents(id: string, after?: string, limit = 100) {
  return apiFetch<EventStreamResponse>(`/investigations/${id}/events`, {
    query: { after, limit }
  });
}

export function getRunEvents(runId: string, after?: string, limit = 100) {
  return apiFetch<EventStreamResponse>(`/investigation-runs/${runId}/events`, {
    query: { after, limit }
  });
}

export function listReportsForInvestigation(investigationId: string) {
  return apiFetch<ShareableReport[]>(`/investigations/${investigationId}/reports`);
}

export function createShareableReport(investigationId: string, payload: CreateShareableReportPayload) {
  return apiFetch<ShareableReport>(`/investigations/${investigationId}/reports`, {
    method: "POST",
    body: JSON.stringify({
      template: payload.template,
      include_technical: payload.include_technical || false
    })
  });
}

export function selectArtifactForReport(investigationId: string, artifactId: string, selected = true) {
  return apiFetch<Artifact>(`/investigations/${investigationId}/artifacts/${artifactId}/report-selection`, {
    method: "POST",
    body: JSON.stringify({ selected })
  });
}

export function explainArtifact(investigationId: string, artifactId: string) {
  return apiFetch<{
    request_type: string;
    artifact_id: string;
    title: string;
    chart_type: string;
    metric: string;
    dimension: string;
    dataset_id: string;
    dataset_ids: string[];
    filters: unknown[];
    row_count: number | null;
    branch_id: string;
    run_id: string;
    rows?: number;
    bins?: number;
    comparison_groups?: number;
    summary?: Record<string, number>;
  }>(`/investigations/${investigationId}/artifacts/${artifactId}/explain`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function getReport(reportId: string) {
  return apiFetch<ShareableReport>(`/reports/${reportId}`);
}

export function getReportReadiness(reportId: string) {
  return apiFetch<ReportReadinessResult>(`/reports/${reportId}/readiness`);
}

export function listReportComments(reportId: string) {
  return apiFetch<ReportComment[]>(`/reports/${reportId}/comments`);
}

export function addReportComment(
  reportId: string,
  payload: { section_id: string; text: string; author?: string }
) {
  return apiFetch<ReportComment>(`/reports/${reportId}/comments`, {
    method: "POST",
    body: JSON.stringify({
      section_id: payload.section_id,
      text: payload.text,
      author: payload.author || "reviewer"
    })
  });
}

export function resolveReportComment(reportId: string, commentId: string) {
  return apiFetch<ReportComment>(`/reports/${reportId}/comments/${commentId}/resolve`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function deleteReportComment(reportId: string, commentId: string) {
  return apiFetch<{ deleted: boolean; comment_id: string }>(`/reports/${reportId}/comments/${commentId}`, {
    method: "DELETE"
  });
}

export function promoteReportCommentToMemory(reportId: string, commentId: string, investigationId?: string) {
  return apiFetch<InvestigationMemoryItem>(`/reports/${reportId}/comments/${commentId}/promote-memory`, {
    method: "POST",
    body: JSON.stringify({ investigation_id: investigationId || undefined })
  });
}

export function promoteReportSectionToMemory(reportId: string, sectionId: string, investigationId?: string) {
  return apiFetch<InvestigationMemoryItem>(`/reports/${reportId}/sections/${sectionId}/promote-memory`, {
    method: "POST",
    body: JSON.stringify({ investigation_id: investigationId || undefined })
  });
}

export function updateReportSection(reportId: string, sectionId: string, payload: ReportSectionUpdatePayload) {
  return apiFetch<ShareableReport>(`/reports/${reportId}/sections/${sectionId}`, {
    method: "PATCH",
    body: JSON.stringify(payload)
  });
}

export function addReportSection(reportId: string, payload: ReportSectionCreatePayload) {
  return apiFetch<ShareableReport>(`/reports/${reportId}/sections`, {
    method: "POST",
    body: JSON.stringify({
      title: payload.title,
      content: payload.content || "",
      order: payload.order
    })
  });
}

export function deleteReportSection(reportId: string, sectionId: string) {
  return apiFetch<ShareableReport>(`/reports/${reportId}/sections/${sectionId}`, {
    method: "DELETE"
  });
}

export function duplicateReportSection(reportId: string, sectionId: string) {
  return apiFetch<ShareableReport>(`/reports/${reportId}/sections/${sectionId}/duplicate`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function reorderReportSections(reportId: string, sectionIds: string[]) {
  return apiFetch<ShareableReport>(`/reports/${reportId}/sections/reorder`, {
    method: "POST",
    body: JSON.stringify({ section_ids: sectionIds })
  });
}

export function approveReportSection(reportId: string, sectionId: string) {
  return apiFetch<ShareableReport>(`/reports/${reportId}/sections/${sectionId}/approve`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function requestReportSectionChanges(reportId: string, sectionId: string) {
  return apiFetch<ShareableReport>(`/reports/${reportId}/sections/${sectionId}/request-changes`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function approveReport(reportId: string) {
  return apiFetch<ShareableReport>(`/reports/${reportId}/approve`, {
    method: "POST",
    body: JSON.stringify({ approved_by: "reviewer", force: false })
  });
}

export function finalizeReport(reportId: string, force = false) {
  return apiFetch<FinalReportSnapshot>(`/reports/${reportId}/finalize`, {
    method: "POST",
    body: JSON.stringify({ created_by: "user", force })
  });
}

export function listFinalSnapshotsForReport(reportId: string) {
  return apiFetch<FinalReportSnapshot[]>(`/reports/${reportId}/final-snapshots`);
}

export function listDataSources() {
  return apiFetch<DataSource[]>("/data-sources");
}

export function createDataSource(payload: CreateDataSourcePayload) {
  return apiFetch<DataSource>("/data-sources", {
    method: "POST",
    body: JSON.stringify({
      name: payload.name,
      type: payload.type,
      location: payload.location || undefined,
      description: payload.description || undefined
    })
  });
}

export async function uploadCsvDataSource(payload: UploadCsvDataSourcePayload) {
  const formData = new FormData();
  formData.set("file", payload.file);
  if (payload.name) formData.set("name", payload.name);
  if (payload.description) formData.set("description", payload.description);
  const response = await fetch(buildApiUrl("/data-sources/upload-csv"), {
    method: "POST",
    body: formData,
    cache: "no-store"
  });
  if (!response.ok) {
    const body = await response.text();
    let detail = body;
    try {
      const parsed = JSON.parse(body);
      if (typeof parsed.detail === "string") detail = parsed.detail;
    } catch {
      // body is not JSON, use as-is
    }
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<UploadCsvDataSourceResponse>;

}

export function getDataSource(id: string) {
  return apiFetch<DataSource>(`/data-sources/${id}`);
}

export function deleteDataSource(id: string, deleteFile = true) {
  return apiFetch<{ deleted: boolean; id: string; file_deleted: boolean }>(`/data-sources/${id}`, {
    method: "DELETE",
    query: { delete_file: deleteFile }
  });
}

export function getDataSourceProfile(id: string) {
  return apiFetch<DataSourceProfile>(`/data-sources/${id}/profile`);
}

export function getUsageContext(id: string) {
  return apiFetch<DataSourceUsageContext>(`/data-sources/${id}/usage-context`);
}

export function getDataSourceSuggestedQuestions(id: string, limit = 8) {
  return apiFetch<{ suggestions: string[] }>(`/data-sources/${id}/suggested-questions`, {
    query: { limit }
  });
}
