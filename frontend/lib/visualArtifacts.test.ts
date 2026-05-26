import { describe, expect, it } from "vitest";
import type { Artifact } from "./api";
import { normalizeChartArtifact } from "./charts";
import { artifactsForBranch, prioritizeArtifactsForBranch } from "./visualArtifacts";

function chart(id: string, branchId: string): Artifact {
  return {
    artifact_id: id,
    title: id,
    artifact_type: "chart",
    visibility: "user",
    metadata: { branch_id: branchId },
    content: { chart_type: "line", rows: [] }
  };
}

describe("visual branch artifact ordering", () => {
  it("prioritizes artifacts from the active branch", () => {
    const ordered = prioritizeArtifactsForBranch([chart("global", "b1"), chart("selected", "b2")], "b2");

    expect(ordered.map((item) => item.artifact_id)).toEqual(["selected", "global"]);
  });


  it("renders comparison histograms from the exact artifact payload", () => {
    const artifact: Artifact = {
      artifact_id: "hist_compare",
      title: "Amount distribution comparison",
      artifact_type: "chart",
      visibility: "user",
      metadata: { branch_id: "distribution::Amount::Location", chart_type: "histogram" },
      content: {
        chart_type: "histogram",
        row_count: 6,
        series: "Location",
        comparison_groups: [
          {
            group: "East",
            row_count: 3,
            bins: [{ label: "0-10", count: 1 }, { label: "10-20", count: 2 }]
          },
          {
            group: "West",
            row_count: 3,
            bins: [{ label: "0-10", count: 2 }, { label: "10-20", count: 1 }]
          }
        ]
      }
    };

    const normalized = normalizeChartArtifact(artifact);

    expect(normalized?.type).toBe("histogram");
    expect(normalized?.comparisonGroups).toBe(2);
    expect(normalized?.rowCount).toBe(6);
    expect(normalized?.points.map((point) => point.label)).toContain("East: 0-10");
    expect(normalized?.points.map((point) => point.label)).toContain("West: 10-20");
  });


});
