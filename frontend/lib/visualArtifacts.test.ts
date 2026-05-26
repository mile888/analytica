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

  it("finds empty selected branch outputs", () => {
    expect(artifactsForBranch([chart("global", "b1")], "b2")).toEqual([]);
  });

  it("preserves histogram artifact identity in card-compatible payloads", () => {
    const artifact: Artifact = {
      artifact_id: "hist_sales_la",
      title: "Sales distribution in Los Angeles",
      artifact_type: "chart",
      visibility: "user",
      metadata: { branch_id: "distribution::Sales::City", chart_type: "histogram" },
      content: {
        chart_type: "histogram",
        row_count: 3,
        bin_edges: [0, 10, 20],
        bin_counts: [1, 2],
        filters: [{ column: "City", operator: "equals", value: "Los Angeles" }]
      }
    };

    expect(artifact.artifact_id).toBe("hist_sales_la");
    expect(artifact.content).toMatchObject({ chart_type: "histogram", row_count: 3 });
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

  it("normalizes scatter relationship artifacts without requiring histogram bins", () => {
    const artifact: Artifact = {
      artifact_id: "sales_profit_relationship",
      title: "Total Sales versus Total Profit by Category",
      artifact_type: "chart",
      visibility: "user",
      metadata: { chart_type: "scatter", x_metric: "Sales", y_metric: "Profit" },
      content: {
        chart_type: "scatter",
        x: "x",
        y: "y",
        x_metric: "Sales",
        y_metric: "Profit",
        grouping: "Category",
        points: [
          { label: "Furniture", x: 4110874.19, y: 285204.72 },
          { label: "Technology", x: 4744557.5, y: 663778.73 }
        ]
      }
    };

    const normalized = normalizeChartArtifact(artifact);

    expect(normalized?.type).toBe("scatter");
    expect(normalized?.xLabel).toBe("Sales");
    expect(normalized?.yLabel).toBe("Profit");
    expect(normalized?.points).toEqual([
      { label: "Furniture", value: 285204.72, x: 4110874.19, y: 285204.72 },
      { label: "Technology", value: 663778.73, x: 4744557.5, y: 663778.73 }
    ]);
  });

  it("rejects malformed scatter artifacts safely", () => {
    const artifact: Artifact = {
      artifact_id: "bad_scatter",
      title: "Broken relationship",
      artifact_type: "chart",
      visibility: "user",
      metadata: { chart_type: "scatter" },
      content: {
        chart_type: "scatter",
        x: "x",
        y: "y",
        points: [
          { label: "Furniture", x: 4110874.19 },
          { label: "Technology", y: 663778.73 }
        ]
      }
    };

    expect(normalizeChartArtifact(artifact)).toBeNull();
  });
});
