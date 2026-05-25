from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MultiDatasetSynthesisPlan:
    comparison_type: str
    operation: str
    subquestions: list[dict[str, Any]] = field(default_factory=list)
    artifact_required: bool = False
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_multi_dataset_synthesis(question: str, dataset_names: list[str] | None = None) -> MultiDatasetSynthesisPlan:
    text = _norm(question)
    artifact_required = any(marker in text for marker in ("visual", "visualization", "chart", "plot", "graph", "side by side", "side-by-side"))
    if any(marker in text for marker in ("separate visual", "separate visuals", "separate visual analyses", "separate visual analysis", "parallel visual", "visual analyses for")):
        comparison_type = "VISUAL_COMPARISON"
        operation = "PARALLEL_VISUAL_ANALYSIS"
        artifact_required = True
    elif any(marker in text for marker in ("support", "supports", "suited for", "best suited", "most suitable")) and any(marker in text for marker in ("time series", "time-series", "trend", "forecast")):
        comparison_type = "CAPABILITY_COMPARISON"
        operation = "DATASET_CAPABILITY_REASONING"
    elif any(marker in text for marker in ("trend", "growth", "over time", "yearly", "annual", "year over year", "time series", "time-series")):
        comparison_type = "TREND_COMPARISON"
        operation = "CROSS_DATASET_TEMPORAL_ANALYSIS"
        artifact_required = True
    elif "dashboard" in text or "kpi" in text:
        comparison_type = "EXECUTIVE_COMPARISON"
        operation = "CROSS_DATASET_KPI_SYNTHESIS"
    elif any(marker in text for marker in ("diversity", "diverse", "variety", "richness")):
        comparison_type = "DIVERSITY_COMPARISON"
        operation = "CROSS_DATASET_DIVERSITY_ANALYSIS"
    elif any(marker in text for marker in ("best suited", "most suitable", "suited for", "support", "supports")) and any(marker in text for marker in ("anomaly", "outlier", "detection", "capability")):
        comparison_type = "CAPABILITY_COMPARISON"
        operation = "DATASET_CAPABILITY_REASONING"
    elif any(marker in text for marker in ("anomaly", "outlier", "unusual", "extreme")):
        comparison_type = "ANOMALY_COMPARISON"
        operation = "CROSS_DATASET_ANOMALY_COMPARISON"
    elif any(marker in text for marker in ("distribution", "side by side", "side-by-side", "histogram")):
        comparison_type = "DISTRIBUTION_COMPARISON"
        operation = "CROSS_DATASET_DISTRIBUTION_COMPARISON"
        artifact_required = True
    elif any(marker in text for marker in ("executive", "summary", "concrete evidence", "management")):
        comparison_type = "EXECUTIVE_COMPARISON"
        operation = "CROSS_DATASET_EXECUTIVE_SYNTHESIS"
    elif any(marker in text for marker in ("operational optimization", "operational optimisation", "better suited", "optimization", "optimisation")):
        comparison_type = "OPERATIONAL_COMPARISON"
        operation = "CROSS_DATASET_OPERATIONAL_OPTIMIZATION"
    elif any(marker in text for marker in ("predictable", "predictability", "feature structure", "feature richness", "feature mix")):
        comparison_type = "PREDICTABILITY_COMPARISON"
        operation = "CROSS_DATASET_PREDICTABILITY_COMPARISON"
    elif any(marker in text for marker in ("relationship", "association", "correlation", "relate to", "relates to", "related to")):
        comparison_type = "RELATIONSHIP_COMPARISON"
        operation = "CROSS_DATASET_RELATIONSHIP_COMPARISON"
    elif any(marker in text for marker in ("inequality", "imbalance", "concentration", "strongest sales cities", "sales cities", "geographic concentration", "geographical concentration")):
        comparison_type = "CONCENTRATION_COMPARISON"
        operation = "CROSS_DATASET_CONCENTRATION_COMPARISON"
    elif any(marker in text for marker in ("profit", "margin", "profitability", "total profit", "highest total profit", "business performance", "health risk", "health-risk", "risk pattern")):
        comparison_type = "PROFITABILITY_COMPARISON"
        operation = "CROSS_DATASET_PROFITABILITY_COMPARISON"
    else:
        comparison_type = "CAPABILITY_COMPARISON"
        operation = "CROSS_DATASET_COMPARISON"
    subquestions = [
        {
            "dataset_name": name,
            "comparison_type": comparison_type,
            "subquestion": _subquestion_for(comparison_type, name),
        }
        for name in (dataset_names or [])
    ]
    return MultiDatasetSynthesisPlan(
        comparison_type=comparison_type,
        operation=operation,
        subquestions=subquestions,
        artifact_required=artifact_required,
        reasoning="Classified multi-dataset comparison type before branch execution.",
    )


def _subquestion_for(comparison_type: str, dataset_name: str) -> str:
    label = f" for {dataset_name}" if dataset_name else ""
    mapping = {
        "DISTRIBUTION_COMPARISON": f"Compute a distribution, range, skew/outlier summary{label}.",
        "ANOMALY_COMPARISON": f"Compute outlier and anomaly evidence from the strongest numeric metric{label}.",
        "EXECUTIVE_COMPARISON": f"Compute executive-level evidence from the strongest KPIs and risks{label}.",
        "PROFITABILITY_COMPARISON": f"Compute domain-appropriate profitability or risk evidence{label}.",
        "CONCENTRATION_COMPARISON": f"Compute imbalance, concentration, or prevalence evidence{label}.",
        "CAPABILITY_COMPARISON": f"Compute operational-actionability evidence{label}.",
        "OPERATIONAL_COMPARISON": f"Compute operational-actionability evidence{label}.",
        "TREND_COMPARISON": f"Compute temporal evidence where true time fields exist{label}.",
        "RELATIONSHIP_COMPARISON": f"Compute relationship evidence from compatible metric pairs{label}.",
        "DIVERSITY_COMPARISON": f"Compute categorical richness and concentration evidence{label}.",
        "PREDICTABILITY_COMPARISON": f"Compute feature-structure and signal richness evidence{label}.",
        "VISUAL_COMPARISON": f"Compute dataset-scoped visual evidence{label}.",
    }
    return mapping.get(comparison_type, f"Compute branch evidence{label}.")


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold())).strip()
