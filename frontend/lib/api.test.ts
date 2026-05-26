import { afterEach, describe, expect, it, vi } from "vitest";
import {
  apiFetch,
  activateInvestigationBranch,
  addReportSection,
  addReportComment,
  addInvestigationMemory,
  approveReportSection,
  buildApiUrl,
  createDataSource,
  createInvestigation,
  createInvestigationMessage,
  createShareableReport,
  deleteReportSection,
  duplicateReportSection,
  finalizeReport,
  getInvestigationBranches,
  getInvestigationSuggestedQuestions,
  getInvestigationWorkflowGuidance,
  listReportComments,
  listInvestigationMemory,
  linkFindingEvidence,
  promoteFindingToMemory,
  promoteReportCommentToMemory,
  promoteReportSectionToMemory,
  listReportsForInvestigation,
  reorderReportSections,
  resolveReportComment,
  requestReportSectionChanges,
  runInvestigation,
  selectArtifactForReport,
  explainArtifact,
  updateInvestigationMemory,
  deleteReportComment,
  deleteDataSource,
  deleteInvestigation,
  updateReportSection,
  uploadCsvDataSource
} from "./api";
import { buildReportReviewSummary } from "@/components/ReportReviewSummary";
import { formatDate, formatDateTime, formatTime } from "./format";
import { normalizeEvidenceReferences } from "./evidence";
import { chartSummary, normalizeChartArtifact } from "./charts";
import { latestChartOrReport, visualAnalysisArtifacts } from "./visualArtifacts";
import {
  assistantAnswerForRun,
  datasetExecutionStatus,
  dedupeConsecutiveMessages,
  formatArtifactForUser,
  formatFindingForUser,
  formatMessageForUser,
  isHighConfidenceKeyFinding,
  isRealAnalyticalFinding,
  truncateText
} from "./display";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("api client", () => {


  it("renders comparison histogram payloads from backend row counts", () => {
    const chart = normalizeChartArtifact({
      artifact_id: "hist_compare_1",
      title: "Amount distribution comparison",
      type: "chart",
      visibility: "user",
      content: {
        chart_type: "histogram",
        visualization_type: "histogram",
        metric: "Amount",
        series: "Location",
        row_count: 42,
        comparison_groups: [
          {
            group: "East",
            label: "East",
            record_count: 25,
            median: 10,
            mean: 12,
            min: 1,
            max: 50,
            bins: [
              { left: 0, right: 10, label: "0 to 10", count: 10 },
              { left: 10, right: 20, label: "10 to 20", count: 15 }
            ]
          },
          {
            group: "West",
            label: "West",
            record_count: 17,
            median: 8,
            mean: 9,
            min: 2,
            max: 30,
            bins: [
              { left: 0, right: 10, label: "0 to 10", count: 7 },
              { left: 10, right: 20, label: "10 to 20", count: 10 }
            ]
          }
        ]
      }
    });

    expect(chart?.type).toBe("histogram");
    expect(chart?.points).toEqual([
      { label: "East: 0 to 10", value: 10, left: 0, right: 10 },
      { label: "East: 10 to 20", value: 15, left: 10, right: 20 },
      { label: "West: 0 to 10", value: 7, left: 0, right: 10 },
      { label: "West: 10 to 20", value: 10, left: 10, right: 20 }
    ]);
    expect(chart ? chartSummary(chart) : "").toContain("2 groups and 42 records");
  });


  it("uploads CSV data sources with FormData and no manual content type", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        data_source: {
          data_source_id: "ds_1",
          name: "Uploaded dataset",
          data_source_type: "csv",
          status: "active",
          created_at: "2026-05-12T00:00:00",
          updated_at: "2026-05-12T00:00:00",
          tags: ["analysis"],
          linked_investigation_ids: []
        },
        profile: {
          row_count: 2,
          column_count: 1,
          columns: [],
          missing_summary: {},
          numeric_summary: {},
          categorical_summary: {},
          sampled_rows: []
        }
      })
    }));
    vi.stubGlobal("fetch", fetchMock);

    await uploadCsvDataSource({
      file: new File(["metric_value\n10\n"], "dataset.csv", { type: "text/csv" }),
      name: "Uploaded dataset",
      description: "Uploaded CSV"
    });

    const calls = fetchMock.mock.calls as unknown as Array<[string, RequestInit]>;
    const [, init] = calls[0];
    expect(calls[0][0]).toBe("http://backend:8000/data-sources/upload-csv");
    expect(init).toMatchObject({ method: "POST", cache: "no-store" });
    expect(init.headers).toBeUndefined();
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("name")).toBe("Uploaded dataset");
    expect((init.body as FormData).get("tags")).toBeNull();
  });


});
