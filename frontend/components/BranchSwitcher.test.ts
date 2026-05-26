import { describe, expect, it, vi } from "vitest";
import type { InvestigationBranch } from "@/lib/api";
import { activateBranchSelection, applyActivatedBranch, branchTypeLabel } from "./BranchSwitcher";

vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh: vi.fn() }) }));

function branch(id: string, title: string, active = false, metadata: Partial<InvestigationBranch> = {}): InvestigationBranch {
  return {
    branch_id: id,
    title,
    branch_type: "distribution",
    artifact_count: 0,
    finding_count: 0,
    updated_at: "2026-01-01T00:00:00Z",
    is_active: active,
    subtitle: "distribution",
    ...metadata
  };
}

describe("BranchSwitcher state", () => {
  it("renders branch list data without exposing ids in titles", () => {
    const branches = [branch("intent::Sales::::histogram::City=Los Angeles", "Sales Distribution in Los Angeles")];

    expect(branches.map((item) => item.title)).toEqual(["Sales Distribution in Los Angeles"]);
    expect(branches[0].title).not.toContain("intent::");
  });

  it("clicking an inactive branch calls activate API", async () => {
    const activateApi = vi.fn().mockResolvedValue({ active_branch_id: "b2", branches: [] });

    await activateBranchSelection({
      investigationId: "inv_1",
      branch: branch("b2", "Sales Distribution in Los Angeles"),
      busy: null,
      activateApi
    });

    expect(activateApi).toHaveBeenCalledWith("inv_1", "b2");
  });


});
