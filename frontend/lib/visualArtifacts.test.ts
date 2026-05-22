import { describe, expect, it } from "vitest";
import type { Artifact } from "./api";
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
});
