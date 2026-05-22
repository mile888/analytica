from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DeliveryDelayAnalysis:
    metric: str
    order_column: str
    delivery_column: str
    row_count: int
    average_delay_days: float
    max_delay_days: float
    delayed_count: int
    bucket_rows: list[dict[str, Any]]
    method_rows: list[dict[str, Any]]


def compute_delivery_delay_analysis(
    df: pd.DataFrame,
    *,
    metric: str,
    order_column: str,
    delivery_column: str,
    method_column: str | None = None,
) -> DeliveryDelayAnalysis | None:
    columns = [metric, order_column, delivery_column]
    if method_column and method_column in df.columns:
        columns.append(method_column)
    working = df[columns].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working[order_column] = pd.to_datetime(working[order_column], errors="coerce")
    working[delivery_column] = pd.to_datetime(working[delivery_column], errors="coerce")
    working = working.dropna(subset=[metric, order_column, delivery_column])
    if working.empty:
        return None
    working["delivery_delay_days"] = (working[delivery_column] - working[order_column]).dt.days
    bucketed = pd.cut(
        working["delivery_delay_days"],
        bins=[float("-inf"), 0, 2, 5, float("inf")],
        labels=["0 days or less", "1-2 days", "3-5 days", "6+ days"],
    )
    working["delay_bucket"] = bucketed.astype(str)
    grouped = (
        working.groupby("delay_bucket", dropna=False, observed=False)
        .agg(record_count=(metric, "count"), average_metric=(metric, "mean"), total_metric=(metric, "sum"), average_delay_days=("delivery_delay_days", "mean"))
        .reset_index()
    )
    bucket_rows = [
        {
            "delay_bucket": str(row["delay_bucket"]),
            "record_count": int(row["record_count"]),
            "average_metric": float(row["average_metric"]),
            "total_metric": float(row["total_metric"]),
            "average_delay_days": float(row["average_delay_days"]),
        }
        for _, row in grouped.iterrows()
    ]
    method_rows: list[dict[str, Any]] = []
    if method_column and method_column in working.columns:
        method_grouped = (
            working.groupby(method_column, dropna=False)
            .agg(record_count=(metric, "count"), average_delay_days=("delivery_delay_days", "mean"), average_metric=(metric, "mean"))
            .reset_index()
            .sort_values("record_count", ascending=False)
        )
        method_rows = [
            {
                method_column: str(row[method_column]),
                "record_count": int(row["record_count"]),
                "average_delay_days": float(row["average_delay_days"]),
                "average_metric": float(row["average_metric"]),
            }
            for _, row in method_grouped.iterrows()
        ]
    return DeliveryDelayAnalysis(
        metric=metric,
        order_column=order_column,
        delivery_column=delivery_column,
        row_count=int(len(working)),
        average_delay_days=float(working["delivery_delay_days"].mean()),
        max_delay_days=float(working["delivery_delay_days"].max()),
        delayed_count=int((working["delivery_delay_days"] > 0).sum()),
        bucket_rows=bucket_rows,
        method_rows=method_rows,
    )


def build_delivery_delay_artifacts(analysis: DeliveryDelayAnalysis, *, query_plan: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = [
        {
            "artifact_type": "table",
            "title": "Delivery delay distribution",
            "content": analysis.bucket_rows,
            "visibility": "user",
            "metadata": {
                "branch_type": "shipping_delay",
                "query_plan": query_plan,
                "metric": analysis.metric,
                "derived_field": "delivery_delay_days",
            },
        }
    ]
    if analysis.method_rows:
        artifacts.append(
            {
                "artifact_type": "table",
                "title": "Delivery delay by shipping category",
                "content": analysis.method_rows,
                "visibility": "user",
                "metadata": {
                    "branch_type": "shipping_delay",
                    "query_plan": query_plan,
                    "metric": analysis.metric,
                    "derived_field": "delivery_delay_days",
                },
            }
        )
    return artifacts
