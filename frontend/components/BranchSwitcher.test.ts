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

  it("does not call activate API for active branch or while busy", async () => {
    const activateApi = vi.fn();

    await activateBranchSelection({ investigationId: "inv_1", branch: branch("b1", "Sales by City", true), busy: null, activateApi });
    await activateBranchSelection({ investigationId: "inv_1", branch: branch("b2", "Sales by Customer Name"), busy: "b1", activateApi });

    expect(activateApi).not.toHaveBeenCalled();
  });

  it("active branch style state changes locally after selection", () => {
    const updated = applyActivatedBranch([
      branch("b1", "Sales by City", true),
      branch("b2", "Sales Distribution in Los Angeles", false)
    ], "b2");

    expect(updated.map((item) => [item.branch_id, item.is_active])).toEqual([
      ["b1", false],
      ["b2", true]
    ]);
  });

  it("next follow-up can use selected branch metadata when available", () => {
    const selected = applyActivatedBranch([
      branch("b1", "Sales by Customer Name", true, { metric: "Sales", dimension: "Customer Name" }),
      branch("b2", "Sales Distribution in Los Angeles", false, {
        metric: "Sales",
        filters: [{ column: "City", operator: "equals", value: "Los Angeles" }],
        chart_type: "histogram"
      })
    ], "b2").find((item) => item.is_active);

    expect(selected?.metric).toBe("Sales");
    expect(selected?.filters?.[0]).toMatchObject({ column: "City", value: "Los Angeles" });
    expect(selected?.chart_type).toBe("histogram");
  });

  it("preserves dataset scope on active branch selection", () => {
    const selected = applyActivatedBranch([
      branch("b1", "Sales by City", false, { dataset_id: "ds_sales", dataset_ids: ["ds_sales"], dataset_scope: "single_dataset" }),
      branch("b2", "Salary by City", true, { dataset_id: "ds_salary", dataset_ids: ["ds_salary"], dataset_scope: "single_dataset" })
    ], "b1").find((item) => item.is_active);

    expect(selected?.dataset_id).toBe("ds_sales");
    expect(selected?.dataset_ids).toEqual(["ds_sales"]);
    expect(selected?.dataset_scope).toBe("single_dataset");
  });

  it("displays meaningful labels for business and multi-dataset branch types", () => {
    expect(branchTypeLabel("business")).toBe("Biz");
    expect(branchTypeLabel("customer_behavior")).toBe("Cust");
    expect(branchTypeLabel("joinability")).toBe("Join");
    expect(branchTypeLabel("warehouse_design")).toBe("WH");
    expect(branchTypeLabel("multi_dataset")).toBe("Multi");
  });
});
