import { Investigation, InvestigationRun } from "@/lib/api";
import { Card, Chip, StatusBadge, formatDate, stageLabel } from "@/components/ui";

export function InvestigationHeader({
  investigation,
  latestRun
}: {
  investigation: Investigation;
  latestRun?: InvestigationRun;
}) {
  return (
    <Card className="overflow-hidden border-slate-200 bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.10),transparent_34%),linear-gradient(135deg,rgba(255,255,255,0.96),rgba(248,250,252,0.92))] p-0 dark:border-slate-800 dark:bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.16),transparent_34%),linear-gradient(135deg,rgba(2,6,23,0.96),rgba(15,23,42,0.92))]">
      <header className="p-6 lg:p-7">
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div className="min-w-0 space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <Chip>Investigation</Chip>
              <StatusBadge value={investigation.status} />
              {latestRun ? <StatusBadge value={latestRun.status} /> : <StatusBadge value="not_started" />}
            </div>
            <div>
              <h1 className="max-w-5xl text-2xl font-semibold tracking-tight text-slate-950 dark:text-white lg:text-3xl">
                {investigation.title}
              </h1>
              <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-600 dark:text-slate-300">
                {investigation.user_question}
              </p>
            </div>
          </div>
          <div className="grid min-w-[260px] gap-3 rounded-2xl border border-white/70 bg-white/70 p-4 text-xs text-slate-600 shadow-sm dark:border-slate-800 dark:bg-slate-950/70 dark:text-slate-300">
            <HeaderFact label="Updated" value={formatDate(investigation.updated_at)} />
            <HeaderFact label="Data sources" value={`${investigation.linked_data_source_ids?.length || 0} linked`} />
            <HeaderFact
              label="Latest run"
              value={latestRun ? `${stageLabel(latestRun.current_stage)} · ${formatDate(latestRun.created_at)}` : "Not run yet"}
            />
          </div>
        </div>
      </header>
    </Card>
  );
}

function HeaderFact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 font-medium text-slate-800 dark:text-slate-100">{value}</div>
    </div>
  );
}
