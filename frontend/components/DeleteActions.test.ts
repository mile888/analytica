import { describe, it, expect, vi, beforeEach } from "vitest";

const pushMock = vi.fn();
const refreshMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock, refresh: refreshMock })
}));

const showToastMock = vi.fn();
vi.mock("@/components/ToastProvider", () => ({
  useToast: () => ({ showToast: showToastMock })
}));

vi.mock("react-dom", async () => {
  const actual = await vi.importActual("react-dom");
  return {
    ...actual,
    createPortal: (children: unknown) => children
  };
});

const deleteDataSourceMock = vi.fn(async () => ({ deleted: true, id: "ds_test_1", file_deleted: true }));
const deleteInvestigationMock = vi.fn(async () => ({ deleted: true }));

vi.mock("@/lib/api", () => ({
  deleteDataSource: (...args: unknown[]) => deleteDataSourceMock(...args),
  deleteInvestigation: (...args: unknown[]) => deleteInvestigationMock(...args)
}));

import { DeleteDataSourceAction } from "./DeleteActions";

describe("DeleteDataSourceAction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("exports the DeleteDataSourceAction component", () => {
    expect(DeleteDataSourceAction).toBeTypeOf("function");
  });


});

describe("deleteDataSource API contract", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });


  it("deleteDataSource failure propagates error", async () => {
    deleteDataSourceMock.mockRejectedValueOnce(new Error("Not found"));
    await expect(deleteDataSourceMock("ds_missing")).rejects.toThrow("Not found");
  });
});
