import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

function source(path: string) {
  return readFileSync(path, "utf8");
}

describe("investigation workspace product cleanup", () => {


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


  it("sends artifact-grounded chart explanation actions", () => {
    const card = source("components/ChartArtifactCard.tsx");

    expect(card).toContain("activateInvestigationBranch");
    expect(card).toContain('action: "explain_artifact"');
    expect(card).toContain("artifact_id: artifact.artifact_id");
    expect(card).toContain("branchId || undefined");
    expect(card).toContain("artifactBranchId");

    expect(card).toContain("chart_type:");
    expect(card).toContain("chart_title:");
    expect(card).toContain("metric:");

    expect(card).toContain("// Branch not found or stale");
  });

});
