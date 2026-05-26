from __future__ import annotations

from typing import Any

from source.product.execution_planner import QueryFilter


def filter_payloads(filters: list[QueryFilter] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in filters or []:
        if isinstance(item, QueryFilter):
            payloads.append(item.to_payload())
        elif isinstance(item, dict):
            payloads.append(dict(item))
    return payloads


def build_histogram_artifact(
    *,
    metric: str,
    bins: list[dict[str, Any]],
    filters: list[QueryFilter] | list[dict[str, Any]] | None,
    title: str,
    query_plan: dict[str, Any] | None = None,
    row_count: int | None = None,
    branch_type: str = "distribution",
) -> dict[str, Any]:
    serialized_filters = filter_payloads(filters)
    if not bins:
        raise ValueError("Histogram artifacts require computed bins.")
    rows = [
        {
            "bin": item.get("label"),
            "count": int(item.get("count") or 0),
            "left": item.get("left"),
            "right": item.get("right"),
        }
        for item in bins
    ]
    edges: list[Any] = []
    if rows:
        edges.append(rows[0].get("left"))
        edges.extend(row.get("right") for row in rows)
    counts = [int(row.get("count") or 0) for row in rows]
    resolved_row_count = int(row_count) if row_count is not None else int(sum(counts))
    if resolved_row_count <= 0 or sum(counts) <= 0:
        raise ValueError("Histogram artifacts require positive computed row counts.")
    if len(edges) != len(counts) + 1:
        raise ValueError("Histogram bin edges must have exactly one more entry than bin counts.")
    if sum(counts) != resolved_row_count:
        raise ValueError("Histogram bin counts must reconcile to the artifact row count.")
    distribution_summary = {
        "row_count": resolved_row_count,
        "bin_count": len(rows),
        "non_empty_bins": sum(1 for count in counts if count > 0),
        "max_bin_count": max(counts) if counts else 0,
    }
    return {
        "artifact_type": "chart",
        "title": title,
        "content": {
            "chart_type": "histogram",
            "visualization_type": "histogram",
            "metric": metric,
            "x": "bin",
            "y": "count",
            "bins": bins,
            "rows": rows,
            "bin_edges": edges,
            "bin_counts": counts,
            "filters": serialized_filters,
            "x_axis": f"{metric} bins",
            "y_axis": "Record count",
            "is_ordered_distribution": True,
            "row_count": resolved_row_count,
            "distribution_summary": distribution_summary,
        },
        "visibility": "user",
        "pinned": True,
        "metadata": {
            "branch_type": branch_type,
            "query_plan": query_plan,
            "chart_type": "histogram",
            "metric": metric,
            "filters": serialized_filters,
            "row_count": resolved_row_count,
            "distribution_summary": distribution_summary,
        },
    }
