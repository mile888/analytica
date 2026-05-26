"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { runInvestigation } from "@/lib/api";
import { InvestigationDataContextItem } from "@/components/InvestigationDataContext";
import { Button, Card, SectionHeader } from "@/components/ui";

export function RunInvestigationPanel({
  investigationId,
  linkedDataSourceIds,
  dataSourceContexts = [],
}: {
  investigationId: string;
  linkedDataSourceIds: string[];
  dataSourceContexts?: InvestigationDataContextItem[];
}) {
  const router = useRouter();
  const [selectedDataSourceIds, setSelectedDataSourceIds] = useState<string[]>(linkedDataSourceIds);
  const [forceRefreshContext, setForceRefreshContext] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function startAnalysis() {
    setIsRunning(true);
    setError(null);
    try {
      await runInvestigation(investigationId, {
        data_source_ids: selectedDataSourceIds,
        force_refresh_context: forceRefreshContext
      });
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analytica could not start the analysis.");
    } finally {
      setIsRunning(false);
    }
  }

  const dataSourceOptions = useMemo(
    () => linkedDataSourceIds.map((id) => ({
      id,
      name: dataSourceContexts.find((item) => item.source.data_source_id === id)?.source.name || "Dataset"
    })),
    [linkedDataSourceIds, dataSourceContexts]
  );

  return (
    <Card>
      <SectionHeader
        title="Analyze"
        description="Analyze the current question against the selected dataset."
        action={
          <Button onClick={startAnalysis} disabled={isRunning || !selectedDataSourceIds.length}>
            {isRunning ? "Analyzing..." : "Analyze"}
          </Button>
        }
      />

      {error ? (
        <p className="mt-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/70 dark:bg-red-950/30 dark:text-red-200">
          {error}
        </p>
      ) : null}

      <details className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-900/60">
        <summary className="cursor-pointer text-xs font-semibold text-slate-500 dark:text-slate-400">
          Advanced settings
        </summary>
        <div className="mt-3 space-y-3">
          {dataSourceOptions.length ? (
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Datasets</div>
              {dataSourceOptions.map((source) => (
                <label
                  key={source.id}
                  className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-600 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-300"
                >
                  <input
                    type="checkbox"
                    checked={selectedDataSourceIds.includes(source.id)}
                    onChange={(event) => {
                      setSelectedDataSourceIds((current) =>
                        event.target.checked ? [...current, source.id] : current.filter((item) => item !== source.id)
                      );
                    }}
                  />
                  <span>{source.name}</span>
                </label>
              ))}
            </div>
          ) : null}

          <label className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400">
            <input
              type="checkbox"
              checked={forceRefreshContext}
              onChange={(event) => setForceRefreshContext(event.target.checked)}
            />
            <span>Refresh dataset profile before analyzing</span>
          </label>
        </div>
      </details>
    </Card>
  );
}
