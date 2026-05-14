import { describe, expect, it } from "vitest";
import { RunEvent } from "@/lib/api";
import { mergeRunEvents } from "./RunTimeline";

function event(event_id: string, created_at: string): RunEvent {
  return {
    event_id,
    created_at,
    run_id: "run_1",
    investigation_id: "inv_1",
    event_type: "info",
    stage: "running_analysis",
    message: event_id,
    severity: "info",
    metadata: {}
  };
}

describe("mergeRunEvents", () => {
  it("deduplicates and keeps chronological order", () => {
    const merged = mergeRunEvents(
      [event("event_2", "2026-01-01T00:00:02Z")],
      [event("event_1", "2026-01-01T00:00:01Z"), event("event_2", "2026-01-01T00:00:02Z")]
    );

    expect(merged.map((item) => item.event_id)).toEqual(["event_1", "event_2"]);
  });
});
