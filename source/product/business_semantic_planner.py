from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from source.product.fallbacks.narration import _output


BUSINESS_INTENTS = {
    "REVENUE_VOLUME",
    "PROFITABILITY",
    "PROFIT_MARGIN",
    "OPERATIONAL_EFFICIENCY",
    "HIGH_SALES_LOW_PROFIT",
    "LOSS_ANALYSIS",
    "DISCOUNT_SENSITIVITY",
    "SHIPPING_COST_EFFICIENCY",
    "CUSTOMER_VALUE",
    "PRODUCT_CATEGORY_PERFORMANCE",
    "REGIONAL_PERFORMANCE",
    "RELATIONSHIP_ANALYSIS",
    "EXECUTIVE_PRIORITY",
    "BUSINESS_RISK_CONCENTRATION",
}


@dataclass(frozen=True)
class BusinessKPIRegistry:
    revenue: str | None = None
    profit: str | None = None
    discount: str | None = None
    shipping_cost: str | None = None
    customer_id: str | None = None
    order_id: str | None = None
    dimensions: list[str] = field(default_factory=list)

    @property
    def available_kpis(self) -> list[str]:
        kpis: list[str] = []
        if self.revenue:
            kpis.append("total_revenue")
        if self.profit:
            kpis.append("total_profit")
            kpis.append("loss_rate")
        if self.revenue and self.profit:
            kpis.append("profit_margin")
            kpis.append("high_sales_low_profit")
            kpis.append("operational_inefficiency_score")
        if self.discount and self.profit:
            kpis.append("discount_sensitivity")
        if self.shipping_cost and self.revenue:
            kpis.append("shipping_cost_burden")
        if self.order_id and self.profit:
            kpis.append("profit_per_order")
        if self.customer_id and self.revenue:
            kpis.append("revenue_per_customer")
        if self.customer_id and self.profit:
            kpis.append("profit_per_customer")
        return kpis

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BusinessPlan:
    operation: str
    grouping: str | None
    base_metrics: list[str] = field(default_factory=list)
    derived_metrics: list[str] = field(default_factory=list)
    x_metric: str | None = None
    y_metric: str | None = None
    chart_type: str = "bar"
    artifact_required: bool = True
    limit: int = 20
    ranking_logic: str = ""
    constraints_locked: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BusinessEvidencePackage:
    operation: str
    intent: str
    metrics_used: list[str]
    derived_kpis: list[str]
    grouping_fields: list[str]
    computed_results: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    findings: list[str]
    limitations: list[str]
    plan: BusinessPlan
    registry: BusinessKPIRegistry

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "intent": self.intent,
            "metrics_used": self.metrics_used,
            "derived_kpis": self.derived_kpis,
            "grouping_fields": self.grouping_fields,
            "computed_results": self.computed_results,
            "artifacts": self.artifacts,
            "findings": self.findings,
            "limitations": self.limitations,
            "plan": self.plan.to_dict(),
            "kpi_registry": self.registry.to_dict(),
        }


def business_analysis_response(
    question: str,
    df: pd.DataFrame,
    *,
    conversation_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    evidence = execute_business_plan(question, df)
    if not evidence:
        return None
    answer = _business_answer(evidence)
    artifacts = _dedupe_artifacts(evidence.artifacts)
    return _output(
        question=question,
        summary=answer,
        findings=evidence.findings[:5],
        evidence=[
            f"Business semantic plan: operation={evidence.plan.operation}, grouping={evidence.plan.grouping}, derived KPIs={', '.join(evidence.derived_kpis) or 'none'}.",
            f"KPI registry: {', '.join(evidence.registry.available_kpis) or 'no derived KPIs available'}.",
        ],
        limitations=evidence.limitations,
        next_steps=_business_next_steps(evidence),
        code="deterministic business KPI execution",
        result_preview=_preview_from_evidence(evidence),
        timeline=[
            {"tool": "business_semantic_planner", "status": "ok", "operation": evidence.plan.operation, "constraints_locked": evidence.plan.constraints_locked},
            {"tool": "business_kpi_registry", "status": "ok", "available_kpis": evidence.registry.available_kpis},
            {"tool": "deterministic_business_kpi_executor", "status": "ok", "rows": len(df)},
            {"tool": "grounded_business_synthesis", "status": "ok"},
        ],
        artifacts=artifacts,
        trace_metadata={
            "fallback": "business_semantic_planner",
            "analysis_type": evidence.plan.operation.lower(),
            "business_intent": evidence.intent,
            "query_plan": evidence.plan.to_dict(),
            "kpi_registry": evidence.registry.to_dict(),
            "evidence_package": evidence.to_dict(),
            "metrics_used": evidence.metrics_used,
            "derived_kpis": evidence.derived_kpis,
            "grouping_fields": evidence.grouping_fields,
        },
    )


def execute_business_plan(question: str, df: pd.DataFrame) -> BusinessEvidencePackage | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    registry = build_kpi_registry(df)
    plan = plan_business_analysis(question, df, registry)
    if not plan:
        return None
    evidence = _execute_plan(df, plan, registry)
    if not evidence.computed_results and not evidence.artifacts:
        return None
    return evidence


def build_kpi_registry(df: pd.DataFrame) -> BusinessKPIRegistry:
    columns = [str(column) for column in df.columns]
    revenue = _column_by_markers(df, ("sales", "revenue", "turnover", "amount", "net sales", "gross sales"))
    profit = _column_by_markers(df, ("profit", "margin amount", "earnings", "net income"))
    discount = _column_by_markers(df, ("discount", "rebate", "markdown"))
    shipping_cost = _column_by_markers(df, ("shipping cost", "ship cost", "freight", "delivery cost", "logistics cost", "transport cost"))
    customer_id = _column_by_markers(df, ("customer id", "customer", "client id", "client", "account id", "account"), require_identifier=True)
    order_id = _column_by_markers(df, ("order id", "order", "invoice", "transaction id", "transaction"), require_identifier=True)
    dimensions = [column for column in columns if _is_business_dimension(df, column)]
    return BusinessKPIRegistry(
        revenue=revenue,
        profit=profit,
        discount=discount,
        shipping_cost=shipping_cost,
        customer_id=customer_id,
        order_id=order_id,
        dimensions=dimensions,
    )


def plan_business_analysis(question: str, df: pd.DataFrame, registry: BusinessKPIRegistry | None = None) -> BusinessPlan | None:
    registry = registry or build_kpi_registry(df)
    text = _norm(question)
    if not _is_business_question(text, registry):
        return None
    if _should_defer_to_authoritative_aggregation(text):
        return None
    chart_requested = any(marker in text for marker in ("chart", "plot", "graph", "visualization", "visualise", "visualize"))
    limit = _limit_from_question(text)

    if _has_any(text, ("sales versus profit", "sales vs profit", "profit versus sales", "profit vs sales")) or (
        chart_requested and registry.revenue and registry.profit and _mentioned(registry.revenue, text) and _mentioned(registry.profit, text)
    ):
        grouping = _select_grouping(text, df, registry, prefer=("category", "segment", "region"))
        return BusinessPlan(
            operation="RELATIONSHIP_ANALYSIS",
            grouping=grouping,
            base_metrics=[item for item in (registry.revenue, registry.profit) if item],
            derived_metrics=["profit_margin"] if registry.revenue and registry.profit else [],
            x_metric=registry.revenue,
            y_metric=registry.profit,
            chart_type="scatter",
            artifact_required=True,
            limit=limit,
            ranking_logic="Compare revenue volume against profitability by group.",
            constraints_locked={"x_metric": bool(registry.revenue and _mentioned(registry.revenue, text)), "y_metric": bool(registry.profit and _mentioned(registry.profit, text)), "grouping": bool(grouping and _mentioned(grouping, text)), "chart_type": chart_requested},
        )

    if _has_any(text, ("discount sensitivity", "discount against", "discount reduce", "discount affect", "discount impact")):
        if not registry.discount or not registry.profit:
            return None
        grouping = _select_grouping(text, df, registry, prefer=("category", "segment", "region"))
        return BusinessPlan(
            operation="DISCOUNT_SENSITIVITY",
            grouping=grouping,
            base_metrics=[item for item in (registry.discount, registry.profit, registry.revenue) if item],
            derived_metrics=["profit_margin"] if registry.revenue and registry.profit else [],
            x_metric=registry.discount,
            y_metric=registry.profit,
            chart_type="scatter",
            artifact_required=True,
            limit=limit,
            ranking_logic="Measure relationship between discount and profit or margin.",
            constraints_locked={"x_metric": True, "y_metric": True},
        )

    if _has_any(text, ("profit margin", "margin weakest", "weakest margin", "margin improvement", "margin-improvement")):
        if not registry.revenue or not registry.profit:
            return None
        grouping = _select_grouping(text, df, registry, prefer=("category", "region", "segment"))
        operation = "EXECUTIVE_PRIORITY" if _has_any(text, ("executive", "presenting", "3 biggest", "three biggest", "opportunities")) else "PROFIT_MARGIN_ANALYSIS"
        return BusinessPlan(
            operation=operation,
            grouping=grouping,
            base_metrics=[registry.revenue, registry.profit],
            derived_metrics=["profit_margin", "loss_rate"],
            chart_type="bar",
            artifact_required=True,
            limit=3 if operation == "EXECUTIVE_PRIORITY" else limit,
            ranking_logic="Rank groups by weakest profit margin and largest addressable revenue/profit leakage.",
            constraints_locked={"derived_metric": True},
        )

    if _has_any(text, ("high sales but low profit", "high revenue but low profit", "high sales low profit", "high revenue weak margin")):
        if not registry.revenue or not registry.profit:
            return None
        grouping = _select_grouping(text, df, registry, prefer=("category", "region", "segment"))
        return BusinessPlan(
            operation="HIGH_SALES_LOW_PROFIT",
            grouping=grouping,
            base_metrics=[registry.revenue, registry.profit],
            derived_metrics=["profit_margin", "high_sales_low_profit"],
            chart_type="scatter",
            artifact_required=True,
            limit=limit,
            ranking_logic="Find high revenue groups whose profit or margin is weak.",
            constraints_locked={"base_metrics": True},
        )

    if _has_any(text, ("operationally inefficient", "operational inefficiency", "inefficient", "operationally expensive", "expensive operationally", "shipping costs hurt", "shipping cost burden")):
        if not registry.revenue or not registry.profit:
            return None
        grouping = _select_grouping(text, df, registry, prefer=("region", "segment", "category"))
        derived = ["profit_margin", "loss_rate", "operational_inefficiency_score"]
        if registry.shipping_cost:
            derived.append("shipping_cost_burden")
        if registry.discount:
            derived.append("average_discount")
        return BusinessPlan(
            operation="BUSINESS_EFFICIENCY_ANALYSIS",
            grouping=grouping,
            base_metrics=[item for item in (registry.revenue, registry.profit, registry.discount, registry.shipping_cost) if item],
            derived_metrics=derived,
            chart_type="bar",
            artifact_required=True,
            limit=limit,
            ranking_logic="High activity combined with weak margin, losses, high discount, or high shipping burden.",
            constraints_locked={"derived_metrics": True},
        )

    if _has_any(text, ("profitability", "profitable", "loss")) and not _has_any(text, ("histogram", "distribution")):
        if not registry.profit:
            return None
        grouping = _select_grouping(text, df, registry, prefer=("category", "region", "segment"))
        derived = ["profit_margin"] if registry.revenue else []
        return BusinessPlan(
            operation="PROFITABILITY",
            grouping=grouping,
            base_metrics=[item for item in (registry.profit, registry.revenue) if item],
            derived_metrics=derived,
            chart_type="bar",
            artifact_required=chart_requested,
            limit=limit,
            ranking_logic="Rank groups by profit and margin when revenue is available.",
            constraints_locked={"profit_metric": True},
        )

    return None


def _execute_plan(df: pd.DataFrame, plan: BusinessPlan, registry: BusinessKPIRegistry) -> BusinessEvidencePackage:
    limitations: list[str] = []
    computed: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    findings: list[str] = []
    metrics_used = [metric for metric in plan.base_metrics if metric]
    grouping_fields = [plan.grouping] if plan.grouping else []

    if plan.operation == "RELATIONSHIP_ANALYSIS":
        rows = _relationship_rows(df, plan, registry)
        computed.append({"analysis_type": "business_relationship", "rows": rows, "x_metric": plan.x_metric, "y_metric": plan.y_metric, "grouping": plan.grouping})
        if rows:
            leader = max(rows, key=lambda row: float(row.get("total_profit", row.get("profit", 0))))
            findings.append(f"`{leader['group']}` has the strongest profit in the Sales/Profit comparison ({leader.get('total_profit', leader.get('profit')):.2f}).")
        artifacts.extend(_relationship_artifacts(plan, rows))

    elif plan.operation == "DISCOUNT_SENSITIVITY":
        rows, corr = _discount_sensitivity_rows(df, plan, registry)
        computed.append({"analysis_type": "discount_sensitivity", "correlation": corr, "rows": rows, "x_metric": registry.discount, "y_metric": registry.profit})
        findings.append(f"Discount/profit relationship correlation is {corr:.3f}." if corr is not None else "Discount/profit relationship could not be estimated reliably.")
        artifacts.extend(_relationship_artifacts(plan, rows, title="Discount sensitivity vs profitability"))

    else:
        rows = _grouped_business_rows(df, plan, registry)
        if not rows:
            limitations.append("No valid grouped KPI rows could be computed.")
        ranked = _rank_business_rows(rows, plan)
        computed.append({"analysis_type": plan.operation.lower(), "rows": ranked, "grouping": plan.grouping, "ranking_logic": plan.ranking_logic})
        findings.extend(_findings_from_rows(ranked, plan))
        artifacts.extend(_ranking_artifacts(plan, ranked))

    if registry.revenue and registry.profit and "profit_margin" in plan.derived_metrics:
        limitations.append("Profit margin is computed deterministically as total profit divided by total revenue; it is not a causal estimate.")
    if plan.grouping:
        small = _small_group_note(df, plan.grouping)
        if small:
            limitations.append(small)

    return BusinessEvidencePackage(
        operation=plan.operation,
        intent=plan.operation,
        metrics_used=metrics_used,
        derived_kpis=list(plan.derived_metrics),
        grouping_fields=grouping_fields,
        computed_results=computed,
        artifacts=artifacts,
        findings=findings,
        limitations=_unique(limitations),
        plan=plan,
        registry=registry,
    )


def _grouped_business_rows(df: pd.DataFrame, plan: BusinessPlan, registry: BusinessKPIRegistry) -> list[dict[str, Any]]:
    grouping = plan.grouping or _first(registry.dimensions)
    if not grouping or grouping not in df.columns:
        return []
    working = df.copy()
    revenue = _numeric(working, registry.revenue)
    profit = _numeric(working, registry.profit)
    discount = _numeric(working, registry.discount)
    shipping = _numeric(working, registry.shipping_cost)
    working["_revenue"] = revenue
    working["_profit"] = profit
    if discount is not None:
        working["_discount"] = discount
    if shipping is not None:
        working["_shipping_cost"] = shipping
    grouped = working.groupby(grouping, observed=True, dropna=False)
    rows: list[dict[str, Any]] = []
    include_shipping = "shipping_cost_burden" in plan.derived_metrics
    include_discount = "average_discount" in plan.derived_metrics
    include_order = "profit_per_order" in plan.derived_metrics
    include_customer = "revenue_per_customer" in plan.derived_metrics or "profit_per_customer" in plan.derived_metrics
    for key, subset in grouped:
        total_revenue = _safe_sum(subset.get("_revenue"))
        total_profit = _safe_sum(subset.get("_profit"))
        total_shipping = _safe_sum(subset.get("_shipping_cost")) if "_shipping_cost" in subset else None
        avg_discount = _safe_mean(subset.get("_discount")) if "_discount" in subset else None
        row = {
            "group": str(key),
            grouping: str(key),
            "records": int(len(subset)),
            "total_revenue": round(total_revenue, 2) if total_revenue is not None else None,
            "total_profit": round(total_profit, 2) if total_profit is not None else None,
            "profit_margin": _ratio_pct(total_profit, total_revenue),
            "loss_rate": _loss_rate(subset.get("_profit")),
        }
        if include_shipping and total_shipping is not None:
            row["total_shipping_cost"] = round(total_shipping, 2)
            row["shipping_cost_burden"] = _ratio_pct(total_shipping, total_revenue)
        if include_discount and avg_discount is not None:
            row["average_discount"] = round(avg_discount, 4)
        if include_order and registry.order_id and registry.order_id in subset:
            orders = max(1, int(subset[registry.order_id].nunique(dropna=True)))
            if total_profit is not None:
                row["profit_per_order"] = round(total_profit / orders, 2)
        if include_customer and registry.customer_id and registry.customer_id in subset:
            customers = max(1, int(subset[registry.customer_id].nunique(dropna=True)))
            if total_revenue is not None:
                row["revenue_per_customer"] = round(total_revenue / customers, 2)
            if total_profit is not None:
                row["profit_per_customer"] = round(total_profit / customers, 2)
        row["operational_inefficiency_score"] = _inefficiency_score(row)
        rows.append(row)
    return rows


def _rank_business_rows(rows: list[dict[str, Any]], plan: BusinessPlan) -> list[dict[str, Any]]:
    if plan.operation == "PROFIT_MARGIN_ANALYSIS":
        ranked = sorted(rows, key=lambda row: _sort_value(row, "profit_margin", default=math.inf))
    elif plan.operation == "EXECUTIVE_PRIORITY":
        ranked = sorted(rows, key=lambda row: (_sort_value(row, "profit_margin", default=math.inf), -_sort_value(row, "total_revenue", default=0)))
    elif plan.operation == "HIGH_SALES_LOW_PROFIT":
        max_revenue = max((_sort_value(row, "total_revenue", default=0) for row in rows), default=0)
        for row in rows:
            row["high_sales_low_profit_score"] = _high_sales_low_profit_score(row, max_revenue)
        ranked = sorted(rows, key=lambda row: _sort_value(row, "high_sales_low_profit_score", default=0), reverse=True)
    elif plan.operation == "BUSINESS_EFFICIENCY_ANALYSIS":
        ranked = sorted(rows, key=lambda row: _sort_value(row, "operational_inefficiency_score", default=0), reverse=True)
    elif plan.operation == "LOSS_ANALYSIS":
        ranked = sorted(rows, key=lambda row: _sort_value(row, "loss_rate", default=0), reverse=True)
    else:
        ranked = sorted(rows, key=lambda row: _sort_value(row, "total_profit", default=0), reverse=True)
    return ranked[: max(1, int(plan.limit or 20))]


def _relationship_rows(df: pd.DataFrame, plan: BusinessPlan, registry: BusinessKPIRegistry) -> list[dict[str, Any]]:
    if not plan.x_metric or not plan.y_metric or plan.x_metric not in df.columns or plan.y_metric not in df.columns:
        return []
    grouping = plan.grouping
    if grouping and grouping in df.columns:
        rows = _grouped_business_rows(df, plan, registry)
        return [
            {
                "group": row["group"],
                "total_sales": row.get("total_revenue"),
                "total_profit": row.get("total_profit"),
                "profit_margin": row.get("profit_margin"),
                "records": row.get("records"),
                "quadrant": _sales_profit_quadrant(row),
            }
            for row in rows
        ]
    working = pd.DataFrame({
        "sales": pd.to_numeric(df[plan.x_metric], errors="coerce"),
        "profit": pd.to_numeric(df[plan.y_metric], errors="coerce"),
    }).dropna()
    return [{"group": f"row_{idx}", "total_sales": float(row.sales), "total_profit": float(row.profit), "profit_margin": _ratio_pct(row.profit, row.sales)} for idx, row in working.iterrows()]


def _discount_sensitivity_rows(df: pd.DataFrame, plan: BusinessPlan, registry: BusinessKPIRegistry) -> tuple[list[dict[str, Any]], float | None]:
    if not registry.discount or not registry.profit:
        return [], None
    discount = pd.to_numeric(df[registry.discount], errors="coerce")
    profit = pd.to_numeric(df[registry.profit], errors="coerce")
    if registry.revenue:
        revenue = pd.to_numeric(df[registry.revenue], errors="coerce")
        profitability = profit / revenue.replace(0, pd.NA)
        y_name = "profit_margin"
    else:
        profitability = profit
        y_name = "profit"
    working = pd.DataFrame({"discount": discount, y_name: profitability}).dropna()
    corr = float(working["discount"].corr(working[y_name])) if len(working) >= 2 else None
    if working.empty:
        return [], corr
    try:
        working["discount_band"] = pd.qcut(working["discount"], q=min(4, max(2, working["discount"].nunique())), duplicates="drop")
    except ValueError:
        working["discount_band"] = pd.cut(working["discount"], bins=2, duplicates="drop")
    grouped = working.groupby("discount_band", observed=True)
    rows = [
        {
            "group": str(key),
            "average_discount": round(float(subset["discount"].mean()), 4),
            y_name: round(float(subset[y_name].mean() * (100 if y_name == "profit_margin" else 1)), 2),
            "records": int(len(subset)),
        }
        for key, subset in grouped
    ]
    return rows, corr


def _ranking_artifacts(plan: BusinessPlan, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    title = _title_for_plan(plan)
    y_key = _value_key_for_plan(plan)
    return [
        {
            "artifact_type": "table",
            "title": title,
            "content": rows,
            "visibility": "user",
            "metadata": {"analysis_type": plan.operation.lower(), "query_plan": plan.to_dict(), "derived_kpis": plan.derived_metrics, "artifact_signature": _artifact_signature(plan) + "::table"},
        },
        {
            "artifact_type": "chart",
            "title": title,
            "content": {"chart_type": "bar", "x": "group", "y": y_key, "rows": rows},
            "visibility": "user",
            "pinned": True,
            "metadata": {"analysis_type": plan.operation.lower(), "chart_type": "bar", "query_plan": plan.to_dict(), "derived_kpis": plan.derived_metrics, "artifact_signature": _artifact_signature(plan) + "::chart"},
        },
    ]


def _relationship_artifacts(plan: BusinessPlan, rows: list[dict[str, Any]], title: str | None = None) -> list[dict[str, Any]]:
    title = title or _title_for_plan(plan)
    x_key = _relationship_x_key(plan)
    y_key = _relationship_y_key(plan)
    points = _relationship_points(rows, x_key, y_key)
    grouping = plan.grouping or "group"
    chart_content = {
        "chart_type": "scatter",
        "x": "x",
        "y": "y",
        "x_metric": plan.x_metric or x_key,
        "y_metric": plan.y_metric or y_key,
        "grouping": grouping,
        "label": "label",
        "points": points,
        "rows": rows,
    }
    return [
        {
            "artifact_type": "table",
            "title": f"{title} table",
            "content": rows,
            "visibility": "user",
            "metadata": {"analysis_type": plan.operation.lower(), "query_plan": plan.to_dict(), "artifact_signature": _artifact_signature(plan) + "::table"},
        },
        {
            "artifact_type": "chart",
            "title": title,
            "content": chart_content,
            "visibility": "user",
            "pinned": True,
            "metadata": {
                "analysis_type": plan.operation.lower(),
                "chart_type": "scatter",
                "x_metric": chart_content["x_metric"],
                "y_metric": chart_content["y_metric"],
                "grouping": grouping,
                "query_plan": plan.to_dict(),
                "artifact_signature": _artifact_signature(plan),
            },
        },
    ]


def _relationship_x_key(plan: BusinessPlan) -> str:
    if plan.operation in {"RELATIONSHIP_ANALYSIS", "HIGH_SALES_LOW_PROFIT"}:
        return "total_sales"
    if plan.operation == "DISCOUNT_SENSITIVITY":
        return "average_discount"
    return str(plan.x_metric or "x")


def _relationship_y_key(plan: BusinessPlan) -> str:
    if plan.operation in {"RELATIONSHIP_ANALYSIS", "HIGH_SALES_LOW_PROFIT"}:
        return "total_profit"
    if plan.operation == "DISCOUNT_SENSITIVITY":
        return "profit_margin" if "profit_margin" in plan.derived_metrics else "profit"
    return str(plan.y_metric or "y")


def _relationship_points(rows: list[dict[str, Any]], x_key: str, y_key: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        x_value = _number_or_none(row.get(x_key))
        y_value = _number_or_none(row.get(y_key))
        if x_value is None or y_value is None:
            continue
        point: dict[str, Any] = {
            "label": str(row.get("group") or row.get("label") or f"Point {idx + 1}"),
            "x": x_value,
            "y": y_value,
        }
        for key in ("records", "profit_margin", "quadrant"):
            if key in row:
                point[key] = row[key]
        points.append(point)
    return points


def _findings_from_rows(rows: list[dict[str, Any]], plan: BusinessPlan) -> list[str]:
    if not rows:
        return []
    top = rows[0]
    if plan.operation == "PROFIT_MARGIN_ANALYSIS":
        return [f"`{top['group']}` has the weakest profit margin at {top.get('profit_margin')}%."]
    if plan.operation == "EXECUTIVE_PRIORITY":
        return [
            f"Margin opportunity #{idx}: `{row['group']}` has margin {row.get('profit_margin')}% on revenue {row.get('total_revenue')}."
            for idx, row in enumerate(rows[:3], start=1)
        ]
    if plan.operation == "HIGH_SALES_LOW_PROFIT":
        return [f"`{top['group']}` combines high revenue ({top.get('total_revenue')}) with weak profit/margin ({top.get('total_profit')}, {top.get('profit_margin')}%)."]
    if plan.operation == "BUSINESS_EFFICIENCY_ANALYSIS":
        return [f"`{top['group']}` has the highest operational inefficiency score ({top.get('operational_inefficiency_score')})."]
    return [f"`{top['group']}` leads the business KPI ranking."]


def _business_answer(evidence: BusinessEvidencePackage) -> str:
    rows = []
    for result in evidence.computed_results:
        if isinstance(result.get("rows"), list):
            rows = result["rows"]
            break
    if evidence.plan.operation == "EXECUTIVE_PRIORITY":
        return " ".join(evidence.findings[:3]) if evidence.findings else "No margin-improvement opportunities could be ranked from the available KPIs."
    if evidence.plan.operation == "DISCOUNT_SENSITIVITY":
        result = evidence.computed_results[0] if evidence.computed_results else {}
        corr = result.get("correlation")
        if corr is None:
            return "Discount sensitivity could not be estimated because there were too few valid discount/profit rows."
        direction = "lower profitability" if corr < 0 else "higher profitability"
        return f"Discount sensitivity is computed from `{evidence.registry.discount}` against profitability: correlation {corr:.3f}, so higher discounts are associated with {direction} in this dataset."
    if evidence.plan.operation == "RELATIONSHIP_ANALYSIS" and rows:
        top = rows[0]
        x_metric = evidence.plan.x_metric or "x metric"
        y_metric = evidence.plan.y_metric or "y metric"
        return f"`{x_metric}` versus `{y_metric}` uses both metrics. `{top['group']}` has {x_metric} {top.get('total_sales')} and {y_metric} {top.get('total_profit')} ({top.get('quadrant')})."
    return evidence.findings[0] if evidence.findings else "Business KPI analysis completed."


def _business_next_steps(evidence: BusinessEvidencePackage) -> list[str]:
    if any(row.get("records", 0) < 3 for result in evidence.computed_results for row in result.get("rows", []) if isinstance(row, dict)):
        return ["Review sparse groups before making operational decisions from the ranking."]
    return []


def _preview_from_evidence(evidence: BusinessEvidencePackage) -> str:
    for result in evidence.computed_results:
        rows = result.get("rows")
        if isinstance(rows, list) and rows:
            return pd.DataFrame(rows).head(20).to_string(index=False)
    return ""


def _dedupe_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for artifact in artifacts:
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        signature = str(metadata.get("artifact_signature") or artifact.get("title") or "")
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(artifact)
    return deduped


def _is_business_question(text: str, registry: BusinessKPIRegistry) -> bool:
    markers = (
        "profit", "margin", "sales", "revenue", "discount", "shipping cost", "profitability",
        "operational", "inefficient", "efficiency", "expensive", "loss", "executive",
        "customer value", "business risk", "high sales", "low profit",
    )
    return any(marker in text for marker in markers) and bool(registry.revenue or registry.profit)


def _should_defer_to_authoritative_aggregation(text: str) -> bool:
    if any(marker in text for marker in (" versus ", " vs ")) and any(marker in text for marker in ("chart", "plot", "graph", "visualization", "visualise", "visualize")):
        return False
    business_derived_markers = (
        "profit margin",
        "margin weakest",
        "weakest margin",
        "margin improvement",
        "operational",
        "inefficient",
        "efficiency",
        "expensive",
        "shipping costs hurt",
        "shipping cost burden",
        "high sales but low profit",
        "high revenue but low profit",
        "high sales low profit",
        "high revenue weak margin",
        "discount sensitivity",
        "discount against",
        "discount reduce",
        "discount affect",
        "discount impact",
        "sales versus profit",
        "sales vs profit",
        "profit versus sales",
        "profit vs sales",
    )
    if any(marker in text for marker in business_derived_markers):
        return False
    explicit_aggregation_markers = (
        "top ",
        "rank ",
        "ranking",
        "group by",
        "grouping column",
        "numeric metric",
        "by total",
        "by average",
        "average ",
        "total ",
        "sum ",
        "instead",
    )
    return any(marker in text for marker in explicit_aggregation_markers)


def _select_grouping(text: str, df: pd.DataFrame, registry: BusinessKPIRegistry, *, prefer: tuple[str, ...]) -> str | None:
    for column in registry.dimensions:
        if _mentioned(column, text):
            return column
    for marker in prefer:
        for column in registry.dimensions:
            if marker in _norm(column):
                return column
    return _first(registry.dimensions)


def _column_by_markers(df: pd.DataFrame, markers: tuple[str, ...], *, require_identifier: bool = False) -> str | None:
    candidates = []
    for column in df.columns:
        name = str(column)
        norm = _norm(name)
        if any(marker in norm for marker in markers):
            if require_identifier:
                if any(marker in norm for marker in (" id", "id", "uuid", "key", "number", "no", "code", "account")):
                    candidates.append(name)
            elif pd.api.types.is_numeric_dtype(df[column]):
                candidates.append(name)
    if require_identifier and not candidates:
        for column in df.columns:
            name = str(column)
            if any(marker in _norm(name) for marker in markers):
                candidates.append(name)
    return candidates[0] if candidates else None


def _is_business_dimension(df: pd.DataFrame, column: str) -> bool:
    if pd.api.types.is_numeric_dtype(df[column]):
        return False
    norm = _norm(column)
    if any(marker in norm for marker in ("id", "uuid", "key", "postal", "zip")):
        return False
    unique = int(df[column].nunique(dropna=True))
    return 2 <= unique <= min(80, max(3, len(df) // 2 + 10))


def _numeric(df: pd.DataFrame, column: str | None) -> pd.Series | None:
    if not column or column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def _safe_sum(series: Any) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.sum()) if not values.empty else None


def _number_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_mean(series: Any) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _ratio_pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator) * 100.0, 2)


def _loss_rate(series: Any) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return round(float((values < 0).mean() * 100.0), 2)


def _inefficiency_score(row: dict[str, Any]) -> float:
    margin = row.get("profit_margin")
    loss = row.get("loss_rate") or 0
    discount = (row.get("average_discount") or 0) * 100
    burden = row.get("shipping_cost_burden") or 0
    negative_margin = max(0.0, 20.0 - float(margin or 0))
    negative_profit = 25.0 if float(row.get("total_profit") or 0) < 0 else 0.0
    return round(negative_margin + float(loss) * 0.3 + float(discount) * 0.4 + float(burden) * 0.4 + negative_profit, 2)


def _high_sales_low_profit_score(row: dict[str, Any], max_revenue: float) -> float:
    revenue = float(row.get("total_revenue") or 0)
    revenue_component = 100.0 * revenue / max(max_revenue, 1.0)
    margin = row.get("profit_margin")
    weak_margin_component = max(0.0, 20.0 - float(margin or 0)) * 4.0
    profit = float(row.get("total_profit") or 0)
    weak_profit_component = 35.0 if profit < 0 else max(0.0, 10.0 - profit / max(revenue, 1.0) * 100.0) * 2.0
    return round(revenue_component + weak_margin_component + weak_profit_component, 2)


def _sales_profit_quadrant(row: dict[str, Any]) -> str:
    revenue = float(row.get("total_sales") or row.get("total_revenue") or 0)
    profit = float(row.get("total_profit") or 0)
    margin = row.get("profit_margin")
    if revenue > 0 and (profit < 0 or (margin is not None and margin < 5)):
        return "high/active revenue with weak profit"
    if profit > 0 and (margin is not None and margin >= 15):
        return "profitable"
    return "mixed"


def _sort_value(row: dict[str, Any], key: str, *, default: float) -> float:
    value = row.get(key)
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _value_key_for_plan(plan: BusinessPlan) -> str:
    if plan.operation == "PROFIT_MARGIN_ANALYSIS":
        return "profit_margin"
    if plan.operation == "HIGH_SALES_LOW_PROFIT":
        return "high_sales_low_profit_score"
    if plan.operation == "BUSINESS_EFFICIENCY_ANALYSIS":
        return "operational_inefficiency_score"
    if plan.operation == "EXECUTIVE_PRIORITY":
        return "profit_margin"
    return "total_profit"


def _title_for_plan(plan: BusinessPlan) -> str:
    relationship_title = f"{plan.x_metric or 'X metric'} versus {plan.y_metric or 'Y metric'} by group"
    labels = {
        "PROFIT_MARGIN_ANALYSIS": "Weakest profit margin by group",
        "BUSINESS_EFFICIENCY_ANALYSIS": "Operational inefficiency ranking",
        "HIGH_SALES_LOW_PROFIT": "High revenue with weak profit",
        "DISCOUNT_SENSITIVITY": "Discount sensitivity vs profitability",
        "RELATIONSHIP_ANALYSIS": relationship_title,
        "EXECUTIVE_PRIORITY": "Margin-improvement opportunities",
        "PROFITABILITY": "Profitability by group",
    }
    return labels.get(plan.operation, plan.operation.replace("_", " ").title())


def _artifact_signature(plan: BusinessPlan) -> str:
    return "::".join([plan.operation, str(plan.grouping or ""), ",".join(plan.base_metrics), ",".join(plan.derived_metrics), plan.chart_type])


def _small_group_note(df: pd.DataFrame, grouping: str) -> str:
    if grouping not in df.columns:
        return ""
    counts = df[grouping].value_counts(dropna=False)
    if not counts.empty and int(counts.min()) < 3:
        return f"Some `{grouping}` groups are sparse; the smallest group has {int(counts.min())} rows."
    return ""


def _limit_from_question(text: str) -> int:
    match = re.search(r"\btop\s+(\d{1,2})\b", text)
    if match:
        return max(1, min(int(match.group(1)), 50))
    if "3 biggest" in text or "three biggest" in text:
        return 3
    return 20


def _mentioned(column: str, text: str) -> bool:
    norm = _norm(column)
    return bool(norm and norm in text)


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _first(values: list[str]) -> str | None:
    return values[0] if values else None


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold())).strip()
