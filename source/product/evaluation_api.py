"""Public evaluation API for Analytica benchmark notebooks.

This module provides stable public wrappers for the internal functions
used by the evaluation notebook.  Importing private ``_``-prefixed names
from product modules is fragile and looks unprofessional in a research
notebook.  Use this module instead:

    from source.product.evaluation_api import (
        semantic_role,
        metric_subtype,
        dataset_primary_purpose,
        ...
    )

Every symbol re-exported here is covered by the automated test suite
and safe to call from evaluation scripts.
"""

from __future__ import annotations

# ── Dataset registry: semantic classification helpers ────────────────────
from source.product.dataset_registry import (
    # Public model classes
    InvestigationDatasetEntry,
    DatasetScope,
    # Public resolution functions
    resolve_dataset_scope,
    resolve_explicit_metric_dataset,
    analyze_dataset_relationships,
    # Internal helpers exposed as stable evaluation API
    _semantic_role as semantic_role,
    _metric_subtype as metric_subtype,
    _dataset_primary_purpose as dataset_primary_purpose,
    _dataset_semantic_profile as dataset_semantic_profile,
    _concise_role_description as concise_role_description,
    _looks_like_followup as looks_like_followup,
    _strategic_insights_answer as strategic_insights_answer,
)

# ── Dataset profiling ────────────────────────────────────────────────────
from source.product.data_profiling import profile_dataframe

# ── Data source models ───────────────────────────────────────────────────
from source.product.data_sources import DataSource, DataSourceType

# ── Investigation store (for benchmark harness) ─────────────────────────
from source.product.store import InvestigationStore

# ── Execution context (for dataset runtime setup) ───────────────────────
from source.product.execution_context import persist_dataset_runtime

__all__ = [
    # Models
    "InvestigationDatasetEntry",
    "DatasetScope",
    "DataSource",
    "DataSourceType",
    "InvestigationStore",
    # Classification helpers
    "semantic_role",
    "metric_subtype",
    "dataset_primary_purpose",
    "dataset_semantic_profile",
    "concise_role_description",
    "looks_like_followup",
    "strategic_insights_answer",
    # Resolution functions
    "resolve_dataset_scope",
    "resolve_explicit_metric_dataset",
    "analyze_dataset_relationships",
    # Profiling
    "profile_dataframe",
    # Execution
    "persist_dataset_runtime",
]
