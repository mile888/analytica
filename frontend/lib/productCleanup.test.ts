import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

function source(path: string) {
  return readFileSync(path, "utf8");
}

describe("investigation workspace product cleanup", () => {
  it("removes the finalized report reader from navigation and routes", () => {
    const appShell = source("components/AppShell.tsx");
    const dashboard = source("app/page.tsx");
    const reportActions = source("components/InvestigationReportActions.tsx");

    expect(existsSync("app/" + "published-" + "reports")).toBe(false);
    expect(appShell).not.toContain("/published-" + "reports");
    expect(appShell).not.toContain("Published " + "Reports");
    expect(dashboard).not.toContain("listFinal" + "Reports");
    expect(reportActions).not.toContain("/published-" + "reports");
  });

  it("keeps one clean theme control and filters developer-like recent investigations", () => {
    const appShell = source("components/AppShell.tsx");
    const investigationFilter = source("lib/investigations.ts");

    expect((appShell.match(/<ThemeToggle/g) || []).length).toBe(1);
    expect(appShell).toContain("productInvestigations");
    expect(investigationFilter).toContain("HIDDEN_INVESTIGATION_TITLE_PATTERNS");
    expect(appShell).toContain(".slice(0, 3)");
    expect(investigationFilter).toContain("/api test/i");
    expect(investigationFilter).toContain("/invalid branch/i");
    expect(investigationFilter).toContain('"de" + "bug"');
  });

  it("hides developer-like investigations from the main list and dashboard", () => {
    const investigationsPage = source("app/investigations/page.tsx");
    const dashboardPage = source("app/page.tsx");

    expect(investigationsPage).toContain("visibleInvestigations");
    expect(investigationsPage).toContain("productInvestigations(investigations)");
    expect(dashboardPage).toContain("visibleInvestigations");
    expect(dashboardPage).toContain("productInvestigations(investigations)");
  });

  it("keeps the regular Reports page with a polished empty state", () => {
    const reportsPage = source("app/reports/page.tsx");

    expect(reportsPage).toContain("<h1 className=\"text-2xl font-semibold tracking-tight text-ink\">Reports</h1>");
    expect(reportsPage).toContain("No reports yet.");
    expect(reportsPage).toContain("Create a report from an investigation after reviewing findings and charts.");
    expect(reportsPage).toContain('href="/investigations"');
    expect(reportsPage).not.toContain("Published " + "Reports");
  });

  it("keeps dataset preview, profile summary, and parse warning visible", () => {
    const datasetPage = source("app/data-sources/[id]/page.tsx");

    expect(datasetPage).toContain('Metric label="Rows"');
    expect(datasetPage).toContain('Metric label="Columns"');
    expect(datasetPage).toContain('Metric label="Numeric"');
    expect(datasetPage).toContain('Metric label="Categorical"');
    expect(datasetPage).toContain("Preview");
    expect(datasetPage).toContain("This dataset may be parsed as a single column.");
    expect(datasetPage).toContain("Created {formatDate(source.created_at)}");
    expect(datasetPage).toContain("linked investigations");
    expect(datasetPage).not.toContain("No description yet.");
    expect(datasetPage).not.toContain("DatasetState");
  });

  it("keeps investigation workspace surfaces after UI cleanup", () => {
    const sidebar = source("components/InvestigationKnowledgeSidebar.tsx");
    const rail = source("components/InvestigationRunRail.tsx");
    const reportActions = source("components/InvestigationReportActions.tsx");

    expect(sidebar).toContain('title="Key findings"');
    expect(sidebar).toContain('title="Visual analysis"');
    expect(sidebar).toContain('title="Turn into report"');
    expect(rail).toContain('title="Suggested next analyses"');
    expect(rail).toContain("<BranchSwitcher");
    expect(reportActions).toContain('title="Report actions"');
    expect(reportActions).toContain("Create shareable report");
    expect(reportActions).toContain("templateDescriptions");
    expect(reportActions).toContain("Include technical details");
    expect(reportActions).not.toContain("technical_appendix");
  });

  it("renders concise finding cards without reasoning boilerplate", () => {
    const sidebar = source("components/InvestigationKnowledgeSidebar.tsx");

    expect(sidebar).toContain('title="Key findings"');
    expect(sidebar).toContain("recordCountChip");
    expect(sidebar).toContain("fullFindingSummary");
    expect(sidebar).toContain("Show all");
    expect(sidebar).toContain("warningText");
    expect(sidebar).not.toContain("Advanced outputs");
    expect(sidebar).not.toContain("WHY IT MATTERS");
    expect(sidebar).not.toContain("WORKING HYPOTHESIS");
    expect(sidebar).not.toContain("POSSIBLE DRIVERS");
    expect(sidebar).not.toContain("WATCHOUT");
    expect(sidebar).not.toContain("VALIDATE NEXT");
    expect(sidebar).not.toContain("INVESTIGATION BRANCHES");
    expect(sidebar).not.toContain("High evidence");
    expect(sidebar).not.toContain("FindingEvidenceActions");
  });

  it("removes suggested workflow and separate Ask Analytica panels", () => {
    const rail = source("components/InvestigationRunRail.tsx");
    const runPanel = source("components/RunInvestigationPanel.tsx");
    const conversation = source("components/InvestigationConversationPanel.tsx");

    expect(rail).not.toContain("Suggested workflow");
    expect(rail).not.toContain("WorkflowGuidance");
    expect(runPanel).not.toContain("Ask Analytica");
    expect(conversation).not.toContain("Ask follow-up");
    expect(conversation).toContain("Analyze");
  });

  it("uses one Analyze action model for follow-ups", () => {
    const composer = source("components/FollowUpComposer.tsx");
    const conversation = source("components/InvestigationConversationPanel.tsx");

    expect(composer).toContain('"Analyze"');
    expect(conversation).toContain('placeholder="Analyze another question..."');
    expect(composer).not.toContain("Ask follow-up");
    expect(composer).not.toContain("disabled={busy || !content.trim()}");
    expect(conversation).not.toContain("disabled={isSubmitting || !content.trim()}");
    expect(composer).toContain("inputRef.current?.focus()");
    expect(composer).toContain("canRunCurrentQuestion");
    expect(composer).toContain("runCurrentQuestion");
    expect(composer).toContain("onCurrentQuestionStarted");
    expect(source("components/ChatWorkspace.tsx")).toContain("Analytica is analyzing this question...");
  });

  it("keeps branch switching in the right rail only", () => {
    const chat = source("components/ChatWorkspace.tsx");
    const rail = source("components/InvestigationRunRail.tsx");

    expect(chat).not.toContain("BranchSwitcher");
    expect((rail.match(/<BranchSwitcher/g) || []).length).toBe(1);
  });

  it("keeps Danger Zone in the scrolling main insight stack", () => {
    const page = source("app/investigations/[id]/page.tsx");
    const sidebar = source("components/InvestigationKnowledgeSidebar.tsx");

    expect(page).toContain("Danger zone");
    expect(page.indexOf("<InvestigationKnowledgeSidebar")).toBeLessThan(page.indexOf("Danger zone"));
    expect(sidebar).not.toContain("sticky");
    expect(sidebar).not.toContain("max-h");
  });

  it("removes advanced source setup and tags inputs", () => {
    const dataSourcesPage = source("app/data-sources/page.tsx");
    const uploadForm = source("components/UploadCsvDataSourceForm.tsx");
    const createForm = source("components/CreateDataSourceForm.tsx");

    expect(dataSourcesPage).not.toContain("Advanced source setup");
    expect(uploadForm).not.toContain("Tags, comma-separated");
    expect(createForm).not.toContain("Tags, comma-separated");
  });

  it("removes archive actions from dataset, investigation, and report pages", () => {
    const files = [
      "app/data-sources/page.tsx",
      "app/data-sources/[id]/page.tsx",
      "app/investigations/page.tsx",
      "app/investigations/[id]/page.tsx",
      "app/reports/[id]/page.tsx"
    ].map(source);

    for (const file of files) {
      expect(file).not.toContain("Archive");
      expect(file).not.toContain("/archive");
      expect(file).not.toContain("archiveInvestigation");
      expect(file).not.toContain("archiveDataSource");
      expect(file).not.toContain("archiveReport");
    }
  });

  it("sends artifact-grounded chart explanation actions", () => {
    const card = source("components/ChartArtifactCard.tsx");

    expect(card).toContain("activateInvestigationBranch");
    expect(card).toContain('action: "explain_artifact"');
    expect(card).toContain("artifact_id: artifact.artifact_id");
    expect(card).toContain("branch_id: branchId");
    expect(card).toContain("artifactBranchId");
  });
});
