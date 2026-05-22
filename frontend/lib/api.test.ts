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
  it("maps execution-context metadata to dataset status", () => {
    const base = {
      data_source_id: "ds_1",
      name: "Dataset",
      data_source_type: "csv",
      status: "active",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      tags: [],
      linked_investigation_ids: []
    };

    expect(datasetExecutionStatus({ ...base, metadata: { execution_context: { executable_available: false, unavailable_reason: "raw_rows_unavailable" } } })).toBe("Reattach required");
    expect(datasetExecutionStatus({ ...base, metadata: { execution_context: { executable_available: true, storage_reference: ".analytica/runtime/ds_1.sqlite" } } })).toBe("Ready for analysis");
    expect(datasetExecutionStatus({ ...base, status: "error" })).toBe("Parsing failed");
  });

  it("formats dates deterministically in UTC", () => {
    expect(formatDateTime("2026-05-15T20:10:00.000Z")).toBe("May 15, 2026, 8:10 PM");
    expect(formatDate("2026-05-15T20:10:00.000Z")).toBe("May 15, 2026");
    expect(formatTime("2026-05-15T20:10:00.000Z")).toBe("8:10 PM");
    expect(formatDateTime("not-a-date")).toBe("Not set");
  });

  it("normalizes evidence references safely", () => {
    const references = normalizeEvidenceReferences([
      { type: "data_source", id: "ds_1", label: "Dataset" },
      { type: "data_source", id: "ds_1", label: "Dataset duplicate" },
      { type: "mystery", id: "x" },
      { type: "artifact" }
    ]);

    expect(references).toEqual([
      {
        type: "data_source",
        id: "ds_1",
        label: "Dataset",
        href: null,
        metadata: {},
        key: "data_source:ds_1"
      },
      {
        type: "unknown",
        id: "x",
        label: "x",
        href: null,
        metadata: {},
        key: "unknown:x"
      }
    ]);
  });

  it("normalizes chart artifacts for lightweight previews", () => {
    const chart = normalizeChartArtifact({
      artifact_id: "chart_1",
      title: "Average metric by group",
      type: "chart",
      visibility: "user",
      content: {
        chart_type: "bar",
        x: "group",
        y: "mean",
        rows: [
          { group: "A", mean: 12 },
          { group: "B", mean: 4 }
        ]
      }
    });

    expect(chart).toMatchObject({
      type: "bar",
      title: "Average metric by group",
      points: [
        { label: "A", value: 12 },
        { label: "B", value: 4 }
      ]
    });
    expect(chart ? chartSummary(chart) : "").toContain("A is highest");
  });

  it("normalizes seasonality heatmap artifacts without turning them into bars", () => {
    const chart = normalizeChartArtifact({
      artifact_id: "heatmap_1",
      title: "Sales seasonality heatmap",
      type: "chart",
      visibility: "user",
      content: {
        chart_type: "seasonality_heatmap",
        x: "month",
        y: "year",
        value: "mean",
        rows: [
          { year: 2024, month: 1, mean: 10 },
          { year: 2024, month: 2, mean: 20 }
        ]
      }
    });

    expect(chart?.type).toBe("heatmap");
    expect(chart ? chartSummary(chart) : "").toContain("Heatmap");
  });

  it("summarizes histograms as ordered bins rather than categorical rankings", () => {
    const chart = normalizeChartArtifact({
      artifact_id: "hist_1",
      title: "Sales distribution in Los Angeles",
      type: "chart",
      visibility: "user",
      content: {
        chart_type: "histogram",
        visualization_type: "histogram",
        bins: [
          { left: 0, right: 10, label: "0 to 10", count: 3 },
          { left: 10, right: 20, label: "10 to 20", count: 1 }
        ],
        is_ordered_distribution: true
      }
    });

    const summary = chart ? chartSummary(chart) : "";
    expect(chart?.type).toBe("histogram");
    expect(summary).toContain("ordered bins");
    expect(summary).not.toContain("highest");
  });

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

  it("rejects comparison histogram groups without backend bins", () => {
    const chart = normalizeChartArtifact({
      artifact_id: "hist_compare_bad",
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
          { group: "East", label: "East", record_count: 25 },
          { group: "West", label: "West", record_count: 17 }
        ]
      }
    });

    expect(chart).toBeNull();
  });

  it("rejects comparison histogram row-count mismatches", () => {
    const chart = normalizeChartArtifact({
      artifact_id: "hist_compare_mismatch",
      title: "Amount distribution comparison",
      type: "chart",
      visibility: "user",
      content: {
        chart_type: "histogram",
        visualization_type: "histogram",
        metric: "Amount",
        row_count: 99,
        comparison_groups: [
          { group: "East", label: "East", record_count: 2, bins: [{ left: 0, right: 1, label: "0 to 1", count: 2 }] },
          { group: "West", label: "West", record_count: 1, bins: [{ left: 0, right: 1, label: "0 to 1", count: 1 }] }
        ]
      }
    });

    expect(chart).toBeNull();
  });

  it("keeps resolved histogram artifacts visible in visual analysis", () => {
    const artifacts = [
      {
        artifact_id: "hist_1",
        title: "Sales distribution in Los Angeles",
        type: "chart",
        visibility: "user",
        content: {
          chart_type: "histogram",
          metric: "Sales",
          x_axis: "Sales bins",
          y_axis: "Record count",
          filters: [{ column: "City", operator: "equals", value: "Los Angeles", source: "observed_value" }],
          bins: [
            { left: 0, right: 10, label: "0 to 10", count: 3 },
            { left: 10, right: 20, label: "10 to 20", count: 1 }
          ]
        }
      }
    ];

    const visual = visualAnalysisArtifacts(artifacts);
    const chart = normalizeChartArtifact(visual.charts[0]);

    expect(visual.charts.map((artifact) => artifact.artifact_id)).toEqual(["hist_1"]);
    expect(chart?.type).toBe("histogram");
    expect(chart?.title).toBe("Sales distribution in Los Angeles");
  });

  it("keeps visual analysis focused on charts and meaningful tables", () => {
    const artifacts = [
      { artifact_id: "chart_1", title: "Sales by City", type: "chart", visibility: "user" },
      { artifact_id: "table_1", title: "Sales by City", type: "table", visibility: "user" },
      { artifact_id: "table_2", title: "Result preview", type: "table", visibility: "user" },
      { artifact_id: "trace_1", title: "Run trace", type: "validation", visibility: "technical" },
      { artifact_id: "report_1", title: "Decision report", type: "report", visibility: "user" }
    ];

    const visual = visualAnalysisArtifacts(artifacts);

    expect(visual.charts.map((artifact) => artifact.artifact_id)).toEqual(["chart_1"]);
    expect(visual.tables.map((artifact) => artifact.artifact_id)).toEqual(["table_1"]);
    expect(visual.technical.map((artifact) => artifact.artifact_id)).toEqual(["trace_1"]);
    expect([...visual.charts, ...visual.tables].map((artifact) => artifact.artifact_id)).not.toContain("report_1");
    expect([...visual.charts, ...visual.tables].map((artifact) => artifact.artifact_id)).not.toContain("table_2");
  });

  it("deduplicates repeated chart artifacts by chart content while keeping the latest id", () => {
    const artifacts = [
      {
        artifact_id: "chart_1",
        title: "Average Sales by Ship Mode",
        type: "chart",
        visibility: "user",
        content: {
          chart_type: "bar",
          x: "Ship Mode",
          y: "mean",
          rows: [
            { "Ship Mode": "Second Class", mean: 235.36 },
            { "Ship Mode": "First Class", mean: 229.08 }
          ]
        }
      },
      {
        artifact_id: "chart_2",
        title: "Average Sales by Ship Mode",
        type: "chart",
        visibility: "user",
        content: {
          chart_type: "bar",
          x: "Ship Mode",
          y: "mean",
          rows: [
            { "Ship Mode": "Second Class", mean: 235.36 },
            { "Ship Mode": "First Class", mean: 229.08 }
          ]
        }
      }
    ];

    const visual = visualAnalysisArtifacts(artifacts);

    expect(visual.charts.map((artifact) => artifact.artifact_id)).toEqual(["chart_2"]);
  });

  it("uses the latest real chart before reports and ignores traces", () => {
    const result = latestChartOrReport(
      [
        { artifact_id: "trace_1", title: "Run trace", type: "validation", visibility: "technical" },
        { artifact_id: "chart_1", title: "Sales by City", type: "chart", visibility: "user" }
      ],
      [{ report_id: "report_1", title: "Draft report" } as never]
    );

    expect(result.chart?.artifact_id).toBe("chart_1");
    expect(result.report).toBeUndefined();
  });

  it("formats technical finding text into user-facing summaries", () => {
    const finding = formatFindingForUser({
      finding_id: "finding_1",
      title: "Finding",
      text: "scalar type=dict; value_preview={'outlier_count': 354, 'column': 'Salary_LPA'}",
      status: "proposed",
      metadata: {}
    });

    expect(finding.summary).toBe("Detected 354 possible outliers in Salary_LPA.");
    expect(finding.summary).not.toContain("value_preview");
  });

  it("formats structured insight metadata for report-ready cards", () => {
    const finding = formatFindingForUser({
      finding_id: "finding_structured",
      title: "Salary variance",
      text: "Raw text should not be primary.",
      status: "proposed",
      metadata: {
        conclusion: "Salary_LPA varies strongly across Job_Title.",
        confidence_level: "High",
        confidence_reason: "Supported by grouped rows and a chart.",
        evidence_strength: "high",
        evidence_reason: "Supported by grouped table and bar chart.",
        limitation: "Small groups may distort ranking.",
        recommended_validation: "Inspect low-sample high-variance groups.",
        business_implication: "Compensation bands may be inconsistent across roles.",
        analysis_type: "grouped_metric",
        supporting_evidence_count: 2
      }
    });

    expect(finding.summary).toBe("Salary_LPA varies strongly across Job_Title.");
    expect(finding.confidenceLabel).toBe("High");
    expect(finding.evidenceStrength).toBe("High");
    expect(finding.businessImplication).toContain("Compensation bands");
    expect(finding.limitation).toContain("Small groups");
    expect(finding.recommendedValidation).toContain("Inspect");
    expect(finding.analysisType).toBe("Grouped analysis");
    expect(JSON.stringify(finding)).not.toContain("supporting_evidence_count");
  });

  it("does not treat profile observations as key findings", () => {
    expect(isRealAnalyticalFinding({
      finding_id: "finding_profile",
      title: "Dataset structure",
      text: "The dataset has enough structure for analytical questions.",
      status: "proposed",
      metadata: { analysis_type: "overview" }
    })).toBe(false);
    expect(isRealAnalyticalFinding({
      finding_id: "finding_real",
      title: "Metric differs",
      text: "Metric values differ strongly across category labels.",
      status: "proposed",
      metadata: { analysis_type: "grouped_metric" }
    })).toBe(true);
  });

  it("keeps only high-confidence substantive findings for key finding surfaces", () => {
    expect(isHighConfidenceKeyFinding({
      finding_id: "finding_high",
      title: "Salary distribution",
      text: "Salary distribution has a high mean across the selected records.",
      status: "proposed",
      metadata: { confidence: "High", analysis_type: "distribution" }
    })).toBe(true);
    expect(isHighConfidenceKeyFinding({
      finding_id: "finding_medium",
      title: "Salary distribution",
      text: "Salary distribution has a high mean across the selected records.",
      status: "proposed",
      metadata: { confidence: "Medium", analysis_type: "distribution" }
    })).toBe(false);
    expect(isHighConfidenceKeyFinding({
      finding_id: "finding_error",
      title: "Chart explanation failed",
      text: "I cannot explain this chart because the exact artifact is missing.",
      status: "proposed",
      metadata: { confidence: "High", analysis_type: "chart_explanation" }
    })).toBe(false);
    expect(isHighConfidenceKeyFinding({
      finding_id: "finding_limitation",
      title: "Limitation",
      text: "Evidence is stronger for the larger groups.",
      status: "proposed",
      metadata: { confidence: "High", analysis_type: "limitation" }
    })).toBe(false);
  });

  it("formats dataframe and SQL payloads without exposing raw previews", () => {
    expect(formatFindingForUser({
      finding_id: "finding_2",
      title: "Finding",
      text: "dataframe shape=(6, 2); columns=['category_label', 'count']; preview=[...]",
      status: "proposed",
      metadata: {}
    }).summary).toBe("Generated a table preview with 6 rows and 2 columns.");
    expect(formatFindingForUser({
      finding_id: "finding_3",
      title: "Finding",
      text: "SQL result shape=(100, 1); row_count=125; truncated=True; table=records",
      status: "proposed",
      metadata: {}
    }).summary).toBe("Generated a table with 125 rows. Preview is available in the Outputs area.");
  });

  it("formats artifacts and messages for non-technical display", () => {
    expect(formatArtifactForUser({
      artifact_id: "artifact_1",
      title: "Result preview",
      type: "table",
      visibility: "user"
    })).toMatchObject({
      typeLabel: "Table",
      isTechnical: false
    });
    expect(formatArtifactForUser({
      artifact_id: "artifact_2",
      title: "Generated Python",
      type: "python_code",
      visibility: "technical"
    }).isTechnical).toBe(true);
    expect(formatMessageForUser({
      message_id: "msg_1",
      investigation_id: "inv_1",
      run_id: "run_technical",
      role: "assistant",
      message_type: "run_summary",
      content: "scalar type=dict; value_preview={'outlier_count': 354}",
      created_at: "2026-01-01T00:00:00Z",
      metadata: {}
    })).toMatchObject({
      label: "Analytica",
      content: "Detected 354 possible outliers."
    });
    expect(formatMessageForUser({
      message_id: "msg_terminal",
      investigation_id: "inv_1",
      run_id: "run_1",
      role: "assistant",
      message_type: "run_summary",
      content: "Done",
      created_at: "2026-01-01T00:00:00Z",
      metadata: {}
    }).content).toBe("No written analytical answer was returned for this run.");
    expect(truncateText("x".repeat(20), 8)).toBe("xxxxxxx…");
  });

  it("renders run states as user-facing assistant answers", () => {
    expect(assistantAnswerForRun({
      run_id: "run_1",
      investigation_id: "inv_1",
      created_at: "2026-01-01T00:00:00Z",
      status: "failed",
      current_stage: "running_analysis",
      data_source_ids: [],
      run_context_summary: {},
      error_message: "RuntimeError: OperationalError: near \")\": syntax error",
      artifact_ids: [],
      report_ids: []
    }, 0, 0)).toMatchObject({
      label: "Analytica",
      tone: "error",
      content: "I could not complete this analysis because one generated query was invalid. Try again with a simpler question."
    });

    const completedAnswer = assistantAnswerForRun({
      run_id: "run_2",
      investigation_id: "inv_1",
      created_at: "2026-01-01T00:00:00Z",
      status: "completed",
      current_stage: "completed",
      data_source_ids: [],
      run_context_summary: {},
      artifact_ids: ["artifact_1", "artifact_2"],
      report_ids: []
    }, 3, 2);
    expect(completedAnswer).toBeNull();
  });

  it("accepts API messages that use type instead of message_type", () => {
    const longAnswer = "Я бы продолжил так: " + "проверить различия между группами. ".repeat(30);
    const display = formatMessageForUser({
      message_id: "msg_2",
      investigation_id: "inv_1",
      run_id: "run_1",
      role: "assistant",
      // Older API serialization may expose the dataclass property as `type`.
      message_type: undefined as never,
      type: "run_summary",
      content: longAnswer,
      created_at: "2026-01-01T00:00:00Z",
      metadata: {}
    });

    expect(display).toMatchObject({
      label: "Analytica",
      tone: "assistant"
    });
    expect(display.content).toBe(longAnswer.trim());
    expect(display.content).not.toContain("…");
    expect(display.status).toBeUndefined();
  });

  it("keeps repeated assistant summaries when user messages are between them", () => {
    const messages = dedupeConsecutiveMessages([
      {
        message_id: "msg_1",
        investigation_id: "inv_1",
        role: "assistant",
        message_type: "run_summary",
        content: "The dataset has 9,898 rows and 18 columns.",
        created_at: "2026-01-01T00:00:00Z",
        metadata: {}
      },
      {
        message_id: "msg_2",
        investigation_id: "inv_1",
        role: "user",
        message_type: "follow_up",
        content: "Build a chart.",
        created_at: "2026-01-01T00:01:00Z",
        metadata: {}
      },
      {
        message_id: "msg_3",
        investigation_id: "inv_1",
        role: "assistant",
        message_type: "run_summary",
        content: "The dataset has 9,898 rows and 18 columns.",
        created_at: "2026-01-01T00:02:00Z",
        metadata: {}
      }
    ]);

    expect(messages.map((message) => message.message_id)).toEqual(["msg_1", "msg_2", "msg_3"]);
  });

  it("deduplicates duplicate assistant responses for the same user message", () => {
    const messages = dedupeConsecutiveMessages([
      {
        message_id: "msg_user",
        investigation_id: "inv_1",
        role: "user",
        message_type: "follow_up",
        content: "есть ли аномалии?",
        created_at: "2026-01-01T00:00:00Z",
        metadata: {}
      },
      {
        message_id: "msg_a",
        investigation_id: "inv_1",
        role: "assistant",
        message_type: "run_summary",
        content: "Sales by City has unusual values in sparse groups and needs sample-size validation.",
        created_at: "2026-01-01T00:01:00Z",
        metadata: { response_to_message_id: "msg_user" }
      },
      {
        message_id: "msg_b",
        investigation_id: "inv_1",
        role: "assistant",
        message_type: "run_summary",
        content: "Sales by City has unusual values in sparse groups; it needs sample size validation.",
        created_at: "2026-01-01T00:02:00Z",
        metadata: { response_to_message_id: "msg_user" }
      }
    ]);

    expect(messages.map((message) => message.message_id)).toEqual(["msg_user", "msg_a"]);
  });

  it("builds URLs with query params and skips empty values", () => {
    const url = buildApiUrl(
      "/investigation-runs/run_1/events",
      { after: "2026-01-01|event_1", limit: 50, empty: "" },
      "http://localhost:8000"
    );

    expect(url).toBe(
      "http://localhost:8000/investigation-runs/run_1/events?after=2026-01-01%7Cevent_1&limit=50"
    );
  });

  it("throws ApiError with status and body on non-2xx", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 500,
        text: async () => "boom"
      }))
    );

    await expect(apiFetch("/broken")).rejects.toMatchObject({
      status: 500,
      body: "boom"
    });
  });

  it("creates data sources with a metadata-only request body", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        data_source_id: "ds_1",
        name: "Uploaded dataset",
        data_source_type: "csv",
        status: "active",
        created_at: "2026-05-12T00:00:00",
        updated_at: "2026-05-12T00:00:00",
        tags: ["analysis"],
        linked_investigation_ids: []
      })
    }));
    vi.stubGlobal("fetch", fetchMock);

    await createDataSource({
      name: "Orders",
      type: "csv",
      location: "local/path/example.csv",
      description: "Uploaded dataset"
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/data-sources",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          name: "Orders",
          type: "csv",
          location: "local/path/example.csv",
          description: "Uploaded dataset"
        })
      })
    );
  });

  it("creates investigations with linked data source ids", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        investigation_id: "inv_1",
        title: "Investigate patterns",
        user_question: "What changed in this dataset?",
        status: "draft",
        created_at: "2026-05-12T00:00:00",
        updated_at: "2026-05-12T00:00:00",
        linked_data_source_ids: ["ds_1"],
        findings: [],
        artifacts: []
      })
    }));
    vi.stubGlobal("fetch", fetchMock);

    await createInvestigation({
      title: "Investigate patterns",
      question: "What changed in this dataset?",
      data_source_ids: ["ds_1"]
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/investigations",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          title: "Investigate patterns",
          question: "What changed in this dataset?",
          data_source_ids: ["ds_1"]
        })
      })
    );
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
    expect(calls[0][0]).toBe("http://localhost:8000/data-sources/upload-csv");
    expect(init).toMatchObject({ method: "POST", cache: "no-store" });
    expect(init.headers).toBeUndefined();
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("name")).toBe("Uploaded dataset");
    expect((init.body as FormData).get("tags")).toBeNull();
  });

  it("creates follow-up messages and runs investigations with a message id", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({})
    }));
    vi.stubGlobal("fetch", fetchMock);

    await createInvestigationMessage("inv_1", { content: "What should I inspect next?" });
    await runInvestigation("inv_1", { data_source_ids: ["ds_1"], message_id: "msg_1" });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/investigations/inv_1/messages",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          content: "What should I inspect next?",
          type: "follow_up",
          role: "user",
          metadata: {}
        })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/investigations/inv_1/run",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          data_source_ids: ["ds_1"],
          force_refresh_context: false,
          message_id: "msg_1",
          analysis_mode: "exploration"
        })
      })
    );
  });

  it("fetches and activates investigation branches", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ branches: [] })
    }));
    vi.stubGlobal("fetch", fetchMock);

    await getInvestigationBranches("inv_1");
    await activateInvestigationBranch("inv_1", "Sales by City");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/investigations/inv_1/branches",
      expect.objectContaining({ cache: "no-store" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/investigations/inv_1/branches/Sales%20by%20City/activate",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("calls report action endpoints", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({})
    }));
    vi.stubGlobal("fetch", fetchMock);

    await listReportsForInvestigation("inv_1");
    await createShareableReport("inv_1", {
      template: "executive_summary",
      include_technical: true
    });
    await selectArtifactForReport("inv_1", "artifact_1", true);
    await finalizeReport("share_1", true);

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/investigations/inv_1/reports",
      expect.objectContaining({ cache: "no-store" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/investigations/inv_1/reports",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          template: "executive_summary",
          include_technical: true
        })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://localhost:8000/investigations/inv_1/artifacts/artifact_1/report-selection",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ selected: true })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://localhost:8000/reports/share_1/finalize",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ created_by: "user", force: true })
      })
    );
  });

  it("calls report section editing endpoints", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({})
    }));
    vi.stubGlobal("fetch", fetchMock);

    await updateReportSection("share_1", "section_1", { title: "Edited", content: "Body" });
    await addReportSection("share_1", { title: "New", content: "Content" });
    await deleteReportSection("share_1", "section_1");
    await duplicateReportSection("share_1", "section_2");
    await reorderReportSections("share_1", ["section_2", "section_1"]);
    await approveReportSection("share_1", "section_2");
    await requestReportSectionChanges("share_1", "section_2");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/reports/share_1/sections/section_1",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ title: "Edited", content: "Body" })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/reports/share_1/sections",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ title: "New", content: "Content" })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://localhost:8000/reports/share_1/sections/section_1",
      expect.objectContaining({ method: "DELETE" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://localhost:8000/reports/share_1/sections/section_2/duplicate",
      expect.objectContaining({ method: "POST" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      "http://localhost:8000/reports/share_1/sections/reorder",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ section_ids: ["section_2", "section_1"] })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      6,
      "http://localhost:8000/reports/share_1/sections/section_2/approve",
      expect.objectContaining({ method: "POST" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      7,
      "http://localhost:8000/reports/share_1/sections/section_2/request-changes",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("calls report comment endpoints", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({})
    }));
    vi.stubGlobal("fetch", fetchMock);

    await listReportComments("share_1");
    await addReportComment("share_1", { section_id: "section_1", text: "Clarify this", author: "reviewer" });
    await resolveReportComment("share_1", "comment_1");
    await deleteReportComment("share_1", "comment_1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/reports/share_1/comments",
      expect.objectContaining({ cache: "no-store" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/reports/share_1/comments",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ section_id: "section_1", text: "Clarify this", author: "reviewer" })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://localhost:8000/reports/share_1/comments/comment_1/resolve",
      expect.objectContaining({ method: "POST" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://localhost:8000/reports/share_1/comments/comment_1",
      expect.objectContaining({ method: "DELETE" })
    );
  });

  it("calls permanent delete endpoints", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ deleted: true })
    }));
    vi.stubGlobal("fetch", fetchMock);

    await deleteInvestigation("inv_1");
    await deleteDataSource("ds_1", true);

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/investigations/inv_1",
      expect.objectContaining({ method: "DELETE" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/data-sources/ds_1?delete_file=true",
      expect.objectContaining({ method: "DELETE" })
    );
  });

  it("calls investigation memory and suggestion endpoints", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ suggestions: [] })
    }));
    vi.stubGlobal("fetch", fetchMock);

    await listInvestigationMemory("inv_1");
    await addInvestigationMemory("inv_1", { type: "risk", content: "Data may be stale" });
    await updateInvestigationMemory("inv_1", "mem_1", { status: "resolved" });
    await getInvestigationSuggestedQuestions("inv_1", 6);
    await getInvestigationWorkflowGuidance("inv_1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/investigations/inv_1/memory",
      expect.objectContaining({ cache: "no-store" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/investigations/inv_1/memory",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ type: "risk", content: "Data may be stale", title: "", status: "active" })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://localhost:8000/investigations/inv_1/memory/mem_1",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ status: "resolved" })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://localhost:8000/investigations/inv_1/suggested-questions?limit=6",
      expect.objectContaining({ cache: "no-store" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      "http://localhost:8000/investigations/inv_1/workflow-guidance",
      expect.objectContaining({ cache: "no-store" })
    );
  });

  it("calls promote-to-memory and evidence-linking endpoints", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({})
    }));
    vi.stubGlobal("fetch", fetchMock);

    await promoteFindingToMemory("inv_1", "finding_1", "risk");
    await linkFindingEvidence("inv_1", "finding_1", {
      type: "data_source",
      id: "ds_1",
      label: "Dataset",
      href: "/data-sources/ds_1"
    });
    await promoteReportCommentToMemory("share_1", "comment_1", "inv_1");
    await promoteReportSectionToMemory("share_1", "section_1", "inv_1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/investigations/inv_1/findings/finding_1/promote-memory",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ type: "risk" })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/investigations/inv_1/findings/finding_1/evidence",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          type: "data_source",
          id: "ds_1",
          label: "Dataset",
          href: "/data-sources/ds_1"
        })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://localhost:8000/reports/share_1/comments/comment_1/promote-memory",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ investigation_id: "inv_1" })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://localhost:8000/reports/share_1/sections/section_1/promote-memory",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ investigation_id: "inv_1" })
      })
    );
  });

  it("computes report review summary counts", () => {
    const summary = buildReportReviewSummary(
      {
        report_id: "share_1",
        investigation_id: "inv_1",
        title: "Report",
        template: "executive_summary",
        status: "draft",
        version: 1,
        is_latest: true,
        approval_status: "draft",
        sections: [
          {
            section_id: "section_1",
            title: "A",
            content: "Body",
            order: 10,
            artifact_ids: [],
            edited_by_user: false,
            created_by: "ai",
            version: 1,
            review_status: "approved",
            updated_at: "2026-01-01T00:00:00",
            metadata: {}
          },
          {
            section_id: "section_2",
            title: "B",
            content: "Body",
            order: 20,
            artifact_ids: [],
            edited_by_user: true,
            created_by: "user",
            version: 2,
            review_status: "changes_requested",
            updated_at: "2026-01-01T00:00:00",
            metadata: {}
          }
        ],
        source_finding_ids: [],
        source_artifact_ids: [],
        include_technical: false,
        created_at: "2026-01-01T00:00:00",
        updated_at: "2026-01-01T00:00:00",
        metadata: {}
      },
      [
        {
          comment_id: "comment_1",
          report_id: "share_1",
          section_id: "section_2",
          text: "Needs detail",
          status: "open",
          created_at: "2026-01-01T00:00:00",
          updated_at: "2026-01-01T00:00:00",
          author: "reviewer",
          metadata: {}
        }
      ],
      {
        report_id: "share_1",
        is_ready: false,
        blocking_count: 1,
        warning_count: 2,
        checks: [],
        generated_at: "2026-01-01T00:00:00"
      }
    );

    expect(summary.totalSections).toBe(2);
    expect(summary.approvedSections).toBe(1);
    expect(summary.changesRequestedSections).toBe(1);
    expect(summary.openComments).toBe(1);
    expect(summary.blockingChecks).toBe(1);
    expect(summary.overallStatus).toBe("Changes requested");
    expect(summary.firstOpenCommentSectionId).toBe("section_2");
  });
});
