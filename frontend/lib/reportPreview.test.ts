import { describe, expect, it } from "vitest";
import type { ShareableReport } from "./api";
import { reportPreviewRows, reportTableColumns, reportTableRows, reportTranscriptItems, reportVisualArtifactSnapshots } from "./reportPreview";

describe("report preview artifact snapshots", () => {
  it("restores persisted visual artifacts for the visual analysis section", () => {
    const section: ShareableReport["sections"][number] = {
      section_id: "section_visual",
      title: "Visual analysis",
      content: "Selected charts",
      order: 40,
      section_type: "visual_analysis",
      artifact_ids: ["chart_1"],
      edited_by_user: false,
      created_by: "system",
      version: 1,
      review_status: "draft",
      updated_at: "2026-01-01T00:00:00Z",
      metadata: {
        artifact_snapshots: [
          {
            artifact_id: "chart_1",
            dataset_ids: ["sales"],
            image_path: "/tmp/chart_1.png",
            content: { x: "City", y: "Sales", rows: [{ City: "LA", Sales: 12 }] }
          }
        ]
      }
    };

    const snapshots = reportVisualArtifactSnapshots(section);

    expect(snapshots).toHaveLength(1);
    expect(snapshots[0]).toMatchObject({ artifact_id: "chart_1", dataset_ids: ["sales"], image_path: "/tmp/chart_1.png" });
    expect(reportPreviewRows(snapshots[0])).toEqual([{ label: "LA", value: 12 }]);
  });


  it("parses numbered question and answer transcript blocks", () => {
    expect(reportTranscriptItems("1. Question: First?\nAnswer: One.\n\n2. Question: Second?\nAnswer: Two.")).toEqual([
      { index: 1, question: "First?", answer: "One." },
      { index: 2, question: "Second?", answer: "Two." }
    ]);
  });
});
