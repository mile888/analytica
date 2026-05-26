"use client";

import { deleteDataSource, deleteInvestigation } from "@/lib/api";
import { ConfirmAction } from "@/components/ConfirmAction";

export function DeleteInvestigationAction({ investigationId }: { investigationId: string }) {
  return (
    <ConfirmAction
      label="Delete"
      title="Delete investigation permanently?"
      description="This will permanently delete this investigation, runs, messages, findings, reports and related history. This cannot be undone."
      confirmLabel="Delete permanently"
      action={() => deleteInvestigation(investigationId)}
      redirectTo="/investigations"
    />
  );
}

export function DeleteDataSourceAction({ dataSourceId }: { dataSourceId: string }) {
  return (
    <ConfirmAction
      label="Delete"
      title="Delete dataset permanently?"
      description="This will permanently delete this data source, its profile, semantic notes and uploaded CSV file if it was stored locally. This cannot be undone."
      confirmLabel="Delete permanently"
      action={() => deleteDataSource(dataSourceId, true)}
      redirectTo="/data-sources"
    />
  );
}
