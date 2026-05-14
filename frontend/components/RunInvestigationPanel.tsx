"use client";

import { useMemo, useState } from "react";
import { InvestigationRun, runInvestigation } from "@/lib/api";
import { Card, StatusBadge, formatDate } from "@/components/ui";
import { RunTimeline } from "@/components/RunTimeline";

export function RunInvestigationPanel({
  investigationId,
  linkedDataSourceIds,
  initialRuns
}: {
  investigationId: string;
  linkedDataSourceIds: string[];
  initialRuns: InvestigationRun[];
}) {
  const [runs, setRuns] = useState<InvestigationRun[]>(initialRuns);
  const [selectedDataSourceIds, setSelectedDataSourceIds] = useState<string[]>(linkedDataSourceIds);
  const [forceRefreshContext, setForceRefreshContext] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const latestRun = runs[0];

  async function startRun() {
    setIsRunning(true);
    setError(null);
    try {
      const run = await runInvestigation(investigationId, {
        data_source_ids: selectedDataSourceIds,
        force_refresh_context: forceRefreshContext
      });
      setRuns((current) => [run, ...current.filter((item) => item.run_id !== run.run_id)]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start investigation run");
    } finally {
      setIsRunning(false);
    }
  }

  const dataSourceOptions = useMemo(() => linkedDataSourceIds, [linkedDataSourceIds]);

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-950">Run investigation</h2>
            <p className="mt-1 text-xs text-slate-500">
              Starts a backend run, then shows the event timeline through polling.
            </p>
          </div>
          <button
            onClick={startRun}
            disabled={isRunning}
            className="rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {isRunning ? "Running..." : "Run investigation"}
          </button>
        </div>

        {dataSourceOptions.length ? (
          <div className="mt-4 space-y-2">
            <div className="text-xs font-medium text-slate-500">Linked data sources</div>
            {dataSourceOptions.map((id) => (
              <label key={id} className="flex items-center gap-2 text-xs text-slate-600">
                <input
                  type="checkbox"
                  checked={selectedDataSourceIds.includes(id)}
                  onChange={(event) => {
                    setSelectedDataSourceIds((current) =>
                      event.target.checked ? [...current, id] : current.filter((item) => item !== id)
                    );
                  }}
                />
                <span>{id}</span>
              </label>
            ))}
          </div>
        ) : null}

        <label className="mt-4 flex items-center gap-2 text-xs text-slate-600">
          <input
            type="checkbox"
            checked={forceRefreshContext}
            onChange={(event) => setForceRefreshContext(event.target.checked)}
          />
          <span>Force refresh context</span>
        </label>

        {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
      </Card>

      <Card>
        <h2 className="text-sm font-semibold text-slate-950">Runs</h2>
        {runs.length ? (
          <div className="mt-3 space-y-3">
            {runs.map((run) => (
              <div key={run.run_id} className="rounded-md border border-slate-200 p-3">
                <div className="flex items-center justify-between gap-2">
                  <StatusBadge value={run.status} />
                  <span className="text-xs text-slate-500">{run.current_stage}</span>
                </div>
                <div className="mt-2 text-xs text-slate-500">{formatDate(run.created_at)}</div>
                <div className="mt-2 text-xs text-slate-500">
                  {run.artifact_ids.length} artifacts · {run.report_ids.length} reports
                </div>
                {run.error_message ? <p className="mt-2 text-xs text-red-600">{run.error_message}</p> : null}
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-3 text-sm text-slate-500">No backend runs yet.</p>
        )}
      </Card>

      <Card>
        <h2 className="text-sm font-semibold text-slate-950">Timeline</h2>
        <div className="mt-3">
          {latestRun ? (
            <RunTimeline runId={latestRun.run_id} status={latestRun.status} />
          ) : (
            <p className="text-sm text-slate-500">No run selected.</p>
          )}
        </div>
      </Card>
    </div>
  );
}
