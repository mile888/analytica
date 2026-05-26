import { describe, expect, it } from "vitest";
import { parseTags } from "./forms";

describe("form helpers", () => {
  it("parses, normalizes and deduplicates comma-separated tags", () => {
    expect(parseTags(" Sales, revenue, sales,  , North America ")).toEqual([
      "sales",
      "revenue",
      "north america"
    ]);
  });
});
