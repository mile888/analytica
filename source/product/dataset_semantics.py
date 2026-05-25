"""Dataset semantic profiling for compatibility checking.

Builds a rich semantic profile from a DataFrame that captures:
- inferred domain
- semantic concepts present in the data
- supported entities (column names + categorical sample values)
- field roles (metric, dimension, timestamp, identifier)
- limitations (what the data does NOT contain)

Reuses existing infrastructure from semantic_layer and llm_semantic_planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from source.product.semantic_layer import (
    build_semantic_dataset_profile,
    _DOMAIN_EVIDENCE,
)


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetSemanticProfile:
    """Rich semantic profile for compatibility checking."""

    domain: str = "general"
    domain_confidence: float = 0.0
    semantic_concepts: list[str] = field(default_factory=list)
    supported_entities: list[str] = field(default_factory=list)
    column_concepts: dict[str, list[str]] = field(default_factory=dict)
    field_roles: dict[str, str] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    column_names: list[str] = field(default_factory=list)


# ── Main entry point ───────────────────────────────────────────────────────

def build_dataset_semantic_profile(df: pd.DataFrame) -> DatasetSemanticProfile:
    """Build a rich semantic profile from a DataFrame.

    Uses existing semantic_layer profiling plus additional concept extraction.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return DatasetSemanticProfile()

    # Get existing semantic profile for domain detection
    base_profile = build_semantic_dataset_profile(df=df)
    domain = base_profile.domain
    domain_confidence = base_profile.domain_confidence

    # Extract concepts from column names
    column_names = [str(col) for col in df.columns]
    column_concepts_map: dict[str, list[str]] = {}
    semantic_concepts: list[str] = []
    supported_entities: list[str] = []

    for col_name in column_names:
        normalized = _normalize(col_name)
        # Add column name itself as a concept
        semantic_concepts.append(normalized)

        # Extract sample values from categorical columns
        series = df[col_name]
        if not pd.api.types.is_numeric_dtype(series):
            nunique = int(series.nunique())
            if 1 < nunique <= 50:
                values = [str(v).strip() for v in series.dropna().unique()[:20]]
                column_concepts_map[col_name] = values
                supported_entities.extend(values)
            elif nunique == 1:
                val = str(series.dropna().iloc[0]).strip() if len(series.dropna()) > 0 else ""
                if val:
                    column_concepts_map[col_name] = [val]
                    supported_entities.append(val)

    # Add domain vocabulary as concepts
    semantic_concepts.extend(base_profile.domain_vocabulary)

    # Build field roles from the base profile
    field_roles: dict[str, str] = {}
    for col_profile in base_profile.columns:
        field_roles[col_profile.name] = col_profile.role.value

    # Detect domain-specific concepts from column names + sample values
    all_text = " ".join(_normalize(c) for c in column_names)
    all_text += " " + " ".join(_normalize(v) for v in supported_entities[:100])

    for evidence_domain, markers in _DOMAIN_EVIDENCE.items():
        for marker in markers:
            if marker in all_text and marker not in semantic_concepts:
                semantic_concepts.append(marker)

    # Build limitations: domains NOT supported
    if domain != "general" and domain_confidence > 0.3:
        non_domains = [d for d in _DOMAIN_EVIDENCE if d != domain and d != "general"]
        limitations = [
            f"No {d}-related data" for d in non_domains[:5]
        ]
    else:
        limitations = []

    # Deduplicate
    semantic_concepts = list(dict.fromkeys(semantic_concepts))
    supported_entities = list(dict.fromkeys(supported_entities))

    return DatasetSemanticProfile(
        domain=domain,
        domain_confidence=domain_confidence,
        semantic_concepts=semantic_concepts,
        supported_entities=supported_entities,
        column_concepts=column_concepts_map,
        field_roles=field_roles,
        limitations=limitations,
        column_names=column_names,
    )


# ── Utilities ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return " ".join(str(text or "").casefold().replace("_", " ").split())
