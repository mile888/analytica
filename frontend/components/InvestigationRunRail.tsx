import type { Artifact, InvestigationBranch, ShareableReport } from "@/lib/api";
import { BranchSwitcher } from "@/components/BranchSwitcher";
import { InvestigationDataContext, type InvestigationDataContextItem } from "@/components/InvestigationDataContext";
import { Card, Chip, EmptyState, SectionHeader } from "@/components/ui";
import { truncateText } from "@/lib/display";

export function InvestigationRunRail({
  investigationId,
  linkedDataSourceIds,
  dataContextItems,
  suggestions,
  branches = []
}: {
  investigationId: string;
  linkedDataSourceIds: string[];
  dataContextItems: InvestigationDataContextItem[];
  suggestions: string[];
  artifacts: Artifact[];
  reports: ShareableReport[];
  branches?: InvestigationBranch[];
}) {
  return (
    <aside className="space-y-3 xl:sticky xl:top-20 xl:max-h-[calc(100vh-6rem)] xl:overflow-y-auto xl:pl-1">
      <BranchSwitcher investigationId={investigationId} branches={branches} />
      <InvestigationDataContext items={dataContextItems} />
      <SuggestedNextAnalyses suggestions={suggestions} />
    </aside>
  );
}

function SuggestedNextAnalyses({ suggestions }: { suggestions: string[] }) {
  const visible = Array.from(new Set(suggestions.map((item) => item.trim()).filter(Boolean))).slice(0, 3);
  return (
    <Card>
      <SectionHeader title="Suggested next analyses" description="Useful follow-ups for this dataset." />
      {visible.length ? (
        <div className="mt-4 flex flex-wrap gap-2">
          {visible.map((suggestion) => (
            <Chip key={suggestion}>{truncateText(suggestion, 58)}</Chip>
          ))}
        </div>
      ) : (
        <div className="mt-4">
          <EmptyState>Suggestions will appear after the dataset profile is available.</EmptyState>
        </div>
      )}
    </Card>
  );
}
