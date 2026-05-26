from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from source.product.execution_planner import AuthoritativeQueryPlan
from source.product.question_routing import branch_type_for_intent, title_for_intent


class BranchRouteAction(str, Enum):
    CREATE = "create"
    SWITCH = "switch"
    CONTINUE = "continue"
    GLOBAL = "global"
    NONE = "none"


@dataclass(frozen=True)
class BranchIdentity:
    dataset_scope: str | None = None
    dataset_id: str | None = None
    dataset_ids: tuple[str, ...] = ()
    metric: str | None = None
    dimension: str | None = None
    time_axis: str | None = None
    chart_type: str | None = None
    aggregation: str | None = None
    transformation: str | None = None
    active_artifact_id: str | None = None
    filters: tuple[tuple[str, str, str], ...] = ()
    intent: str = "general"

    def key(self) -> str:
        branch_type = _branch_type(self)
        filters = "|".join(f"{col}{op}{value}" for col, op, value in self.filters)
        aggregation = _canonical_aggregation(self)
        chart_family = _canonical_chart_family(self)
        dataset_key = ",".join(self.dataset_ids) or (self.dataset_id or "")
        prefix = (branch_type, dataset_key) if dataset_key else (branch_type,)
        if branch_type == "temporal":
            return "::".join(str(item or "") for item in (*prefix, self.metric, self.time_axis, aggregation, self.transformation, filters))
        if branch_type == "distribution":
            return "::".join(str(item or "") for item in (*prefix, self.metric, chart_family, self.transformation, filters))
        if branch_type == "grouped":
            return "::".join(str(item or "") for item in (*prefix, self.metric, self.dimension, aggregation, self.transformation, filters))
        return "::".join(str(item or "") for item in (*prefix, self.intent, self.metric, self.dimension, self.time_axis, chart_family, aggregation, self.transformation, filters))


@dataclass
class AnalyticalBranch:
    branch_id: str
    identity: BranchIdentity
    title: str
    branch_type: str = ""
    intent_type: str = ""
    parent_branch_id: str | None = None
    created_from_query: str = ""
    last_user_query: str = ""
    last_query_plan: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    active_artifact_id: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["identity"] = asdict(self.identity)
        return payload


@dataclass
class BranchWorkspace:
    branches: dict[str, AnalyticalBranch] = field(default_factory=dict)
    active_branch_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "active_branch_id": self.active_branch_id,
            "branches": {key: branch.to_payload() for key, branch in self.branches.items()},
        }


@dataclass(frozen=True)
class BranchRouteDecision:
    action: BranchRouteAction
    branch_id: str
    reason: str
    identity: BranchIdentity

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        payload["identity"] = asdict(self.identity)
        return payload


class BranchWorkspaceManager:
    FOLLOWUP_INTENTS = {"evidence", "contradiction", "validation", "transformation_followup"}

    @classmethod
    def route(cls, plan: AuthoritativeQueryPlan, workspace: BranchWorkspace | None = None) -> BranchRouteDecision:
        workspace = workspace or BranchWorkspace()
        identity = _identity_from_plan(plan)
        key = identity.key()
        if plan.can_use_active_branch and workspace.active_branch_id:
            return BranchRouteDecision(
                action=BranchRouteAction.CONTINUE,
                branch_id=workspace.active_branch_id,
                reason="follow-up can continue the active analytical branch",
                identity=identity,
            )
        if key in workspace.branches:
            return BranchRouteDecision(
                action=BranchRouteAction.SWITCH,
                branch_id=key,
                reason="explicit task matches an existing branch identity",
                identity=identity,
            )
        return BranchRouteDecision(
            action=BranchRouteAction.CREATE,
            branch_id=key,
            reason="explicit task creates a new authoritative analytical branch",
            identity=identity,
        )

    @staticmethod
    def from_payload(value: Any) -> BranchWorkspace:
        if not isinstance(value, dict):
            return BranchWorkspace()
        branches: dict[str, AnalyticalBranch] = {}
        raw_branches = value.get("branches") if isinstance(value.get("branches"), dict) else {}
        active_input = str(value.get("active_branch_id") or "")
        active_canonical = ""
        for key, raw in raw_branches.items():
            if not isinstance(raw, dict):
                continue
            ident = raw.get("identity") if isinstance(raw.get("identity"), dict) else {}
            identity = BranchIdentity(
                dataset_scope=ident.get("dataset_scope"),
                dataset_id=ident.get("dataset_id"),
                dataset_ids=tuple(str(item) for item in ident.get("dataset_ids", []) if str(item).strip()),
                metric=ident.get("metric"),
                dimension=ident.get("dimension"),
                time_axis=ident.get("time_axis"),
                chart_type=ident.get("chart_type"),
                aggregation=ident.get("aggregation"),
                transformation=ident.get("transformation"),
                active_artifact_id=ident.get("active_artifact_id"),
                filters=tuple(tuple(item) for item in ident.get("filters", [])),
                intent=str(ident.get("intent") or "general"),
            )
            canonical_key = identity.key()
            if str(key) == active_input or str(raw.get("branch_id") or "") == active_input:
                active_canonical = canonical_key
            if canonical_key in branches:
                existing = branches[canonical_key]
                existing.last_query_plan = raw.get("last_query_plan") if isinstance(raw.get("last_query_plan"), dict) else existing.last_query_plan
                existing.findings.extend([str(item) for item in raw.get("findings", []) if str(item).strip()] if isinstance(raw.get("findings"), list) else [])
                existing.artifacts.extend([item for item in raw.get("artifacts", []) if isinstance(item, dict)] if isinstance(raw.get("artifacts"), list) else [])
                continue
            branches[canonical_key] = AnalyticalBranch(
                branch_id=canonical_key,
                identity=identity,
                title=_title_for_identity(identity),
                branch_type=str(raw.get("branch_type") or _branch_type(identity)),
                intent_type=str(raw.get("intent_type") or identity.intent),
                parent_branch_id=raw.get("parent_branch_id"),
                created_from_query=str(raw.get("created_from_query") or ""),
                last_user_query=str(raw.get("last_user_query") or ""),
                last_query_plan=raw.get("last_query_plan") if isinstance(raw.get("last_query_plan"), dict) else {},
                findings=[str(item) for item in raw.get("findings", []) if str(item).strip()] if isinstance(raw.get("findings"), list) else [],
                artifacts=[item for item in raw.get("artifacts", []) if isinstance(item, dict)] if isinstance(raw.get("artifacts"), list) else [],
                active_artifact_id=str(raw.get("active_artifact_id") or identity.active_artifact_id or ""),
            )
        active = active_canonical or active_input
        if active not in branches:
            active = next((bid for bid, branch in branches.items() if str(branch.branch_id) == active), active)
        return BranchWorkspace(branches=branches, active_branch_id=active if active in branches else None)


def branch_workspace_from_investigation(investigation: Any) -> BranchWorkspace:
    metadata = getattr(investigation, "metadata", {}) if isinstance(getattr(investigation, "metadata", {}), dict) else {}
    workspace = BranchWorkspaceManager.from_payload(metadata.get("branch_workspace"))
    if not workspace.branches:
        workspace = _workspace_from_outputs(investigation)
    return workspace


def branch_dtos_for_investigation(investigation: Any) -> list[dict[str, Any]]:
    workspace = branch_workspace_from_investigation(investigation)
    active = workspace.active_branch_id
    artifacts = list(getattr(investigation, "artifacts", []) or [])
    findings = list(getattr(investigation, "findings", []) or [])
    out = []
    for branch_id, branch in workspace.branches.items():
        artifact_count = max(_count_for_branch(artifacts, branch_id), len(branch.artifacts))
        finding_count = max(_count_for_branch(findings, branch_id), len(branch.findings))
        title = branch.title or _title_for_identity(branch.identity)
        out.append(
            {
                "branch_id": branch_id,
                "title": title,
                "branch_type": branch.branch_type or _branch_type(branch.identity),
                "intent_type": branch.intent_type or branch.identity.intent,
                "dataset_scope": branch.identity.dataset_scope,
                "dataset_id": branch.identity.dataset_id,
                "dataset_ids": list(branch.identity.dataset_ids),
                "active_artifact_id": branch.active_artifact_id or branch.identity.active_artifact_id,
                "metric": branch.identity.metric,
                "dimension": branch.identity.dimension,
                "filters": [
                    {"column": col, "operator": op, "value": value}
                    for col, op, value in branch.identity.filters
                ],
                "chart_type": branch.identity.chart_type,
                "artifact_count": artifact_count,
                "finding_count": finding_count,
                "updated_at": _branch_updated_at(branch),
                "is_active": branch_id == active,
                "subtitle": _branch_subtitle(branch.identity, artifact_count, finding_count),
            }
        )
    return sorted(out, key=lambda item: (not item["is_active"], item["title"]))


def activate_branch(investigation: Any, branch_id: str) -> dict[str, Any]:
    workspace = branch_workspace_from_investigation(investigation)
    if branch_id not in workspace.branches:
        raise KeyError("Branch not found")
    workspace.active_branch_id = branch_id
    metadata = dict(getattr(investigation, "metadata", {}) or {})
    metadata["branch_workspace"] = workspace.to_payload()
    branch = workspace.branches[branch_id]
    state = dict(metadata.get("conversation_state") or {})
    state["active_metric"] = branch.identity.metric
    state["active_dimension"] = branch.identity.dimension
    state["active_time_axis"] = branch.identity.time_axis
    state["active_chart_type"] = branch.identity.chart_type
    state["active_branch_type"] = _branch_type(branch.identity)
    state["active_branch_id"] = branch_id
    state["active_dataset_scope"] = branch.identity.dataset_scope
    state["active_dataset_id"] = branch.identity.dataset_id
    state["active_dataset_ids"] = list(branch.identity.dataset_ids)
    active_artifact = _latest_artifact_for_branch(investigation, branch_id)
    if active_artifact:
        state.update(active_artifact)
    if _branch_type(branch.identity) == "distribution" and branch.identity.metric:
        state["distribution_state"] = {
            "branch_type": "distribution",
            "metric": branch.identity.metric,
            "filters": [
                {"column": col, "operator": op, "value": value}
                for col, op, value in branch.identity.filters
            ],
            "chart_type": branch.identity.chart_type or "histogram",
        }
    metadata["conversation_state"] = state
    investigation.metadata = metadata
    return metadata


def _latest_artifact_for_branch(investigation: Any, branch_id: str) -> dict[str, Any]:
    for artifact in reversed(list(getattr(investigation, "artifacts", []) or [])):
        metadata = getattr(artifact, "metadata", None)
        content = getattr(artifact, "content", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        content = content if isinstance(content, dict) else {}
        nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        artifact_branch_id = str(metadata.get("branch_id") or nested.get("branch_id") or "")
        if artifact_branch_id != branch_id:
            continue
        chart_type = str(metadata.get("chart_type") or nested.get("chart_type") or content.get("chart_type") or "")
        artifact_id = str(getattr(artifact, "artifact_id", "") or "")
        metric = str(metadata.get("metric") or nested.get("metric") or content.get("metric") or "")
        dimension = str(metadata.get("dimension") or nested.get("dimension") or content.get("dimension") or content.get("x") or "")
        aggregation = str(metadata.get("aggregation") or nested.get("aggregation") or content.get("aggregation") or "")
        filters = metadata.get("filters") or nested.get("filters") or content.get("filters") or []
        payload = {
            "active_artifact_id": artifact_id,
            "active_dataset_id": metadata.get("dataset_id") or nested.get("dataset_id") or "",
            "active_dataset_ids": metadata.get("dataset_ids") or nested.get("dataset_ids") or [],
            "active_dataset_scope": metadata.get("dataset_scope") or nested.get("dataset_scope") or "",
            "active_metric": metric,
            "active_dimension": dimension,
            "active_aggregation": aggregation,
            "active_filters": filters if isinstance(filters, list) else [],
            "active_chart_type": chart_type,
        }
        if chart_type == "histogram":
            payload["active_distribution_context"] = {
                "artifact_id": artifact_id,
                "metric": metric,
                "dimension": dimension,
                "filters": payload["active_filters"],
                "chart_type": chart_type,
            }
        derived = metadata.get("derived_field") or nested.get("derived_field")
        if isinstance(derived, dict):
            payload["active_bin_context"] = derived
        transformation = str(metadata.get("transformation_type") or nested.get("transformation_type") or "")
        if transformation:
            payload["active_transformation"] = transformation
        return {key: value for key, value in payload.items() if value not in ("", None, [])}
    return {}


def upsert_branch_from_plan(
    workspace: BranchWorkspace,
    plan_payload: dict[str, Any],
    *,
    artifact_count: int = 0,
    finding_count: int = 0,
) -> BranchWorkspace:
    plan = _plan_like(plan_payload)
    identity = _identity_from_plan(plan, plan_payload)
    branch_id = identity.key()
    branch = workspace.branches.get(branch_id) or AnalyticalBranch(
        branch_id=branch_id,
        identity=identity,
        title=_title_for_identity(identity),
        branch_type=_branch_type(identity),
        intent_type=identity.intent,
    )
    branch.last_query_plan = dict(plan_payload)
    branch.artifacts = [{}] * max(0, artifact_count)
    branch.findings = [""] * max(0, finding_count)
    branch.active_artifact_id = str(plan_payload.get("active_artifact_id") or branch.active_artifact_id or "")
    branch.created_from_query = branch.created_from_query or str(plan_payload.get("raw_question") or "")
    branch.last_user_query = str(plan_payload.get("raw_question") or branch.last_user_query)
    workspace.branches[branch_id] = branch
    workspace.active_branch_id = branch_id
    return workspace


def upsert_branch_for_intent(
    workspace: BranchWorkspace,
    *,
    intent_type: str,
    dataset_scope: str = "",
    dataset_id: str = "",
    dataset_ids: list[str] | tuple[str, ...] | None = None,
    created_from_query: str = "",
    parent_branch_id: str | None = None,
) -> BranchWorkspace:
    identity = BranchIdentity(
        dataset_scope=dataset_scope or None,
        dataset_id=dataset_id or None,
        dataset_ids=tuple(str(item) for item in (dataset_ids or []) if str(item).strip()),
        intent=str(intent_type or "analysis"),
    )
    branch_id = identity.key()
    branch = workspace.branches.get(branch_id) or AnalyticalBranch(
        branch_id=branch_id,
        identity=identity,
        title=title_for_intent(intent_type),
        branch_type=branch_type_for_intent(intent_type),
        intent_type=str(intent_type),
        parent_branch_id=parent_branch_id,
        created_from_query=created_from_query,
    )
    branch.title = title_for_intent(intent_type)
    branch.branch_type = branch_type_for_intent(intent_type)
    branch.intent_type = str(intent_type)
    branch.last_user_query = created_from_query
    workspace.branches[branch_id] = branch
    workspace.active_branch_id = branch_id
    return workspace


def _identity_from_plan(plan: AuthoritativeQueryPlan, plan_payload: dict[str, Any] | None = None) -> BranchIdentity:
    payload = plan_payload or {}
    return BranchIdentity(
        dataset_scope=payload.get("dataset_scope"),
        dataset_id=payload.get("dataset_id"),
        dataset_ids=tuple(str(item) for item in payload.get("dataset_ids", []) if str(item).strip()) if isinstance(payload.get("dataset_ids"), list) else (),
        metric=plan.metric,
        dimension=plan.dimension,
        time_axis=plan.time_axis,
        chart_type=plan.chart_type,
        aggregation=plan.aggregation,
        transformation=plan.transformation,
        active_artifact_id=payload.get("active_artifact_id"),
        filters=tuple((item.column, item.operator, str(item.value)) for item in plan.filters),
        intent=_canonical_intent(plan),
    )


def _workspace_from_outputs(investigation: Any) -> BranchWorkspace:
    workspace = BranchWorkspace()
    for artifact in list(getattr(investigation, "artifacts", []) or []):
        metadata = getattr(artifact, "metadata", {}) if isinstance(getattr(artifact, "metadata", {}), dict) else {}
        content = getattr(artifact, "content", {}) if isinstance(getattr(artifact, "content", {}), dict) else {}
        plan_payload = metadata.get("query_plan") if isinstance(metadata.get("query_plan"), dict) else content.get("query_plan")
        if isinstance(plan_payload, dict):
            workspace = upsert_branch_from_plan(workspace, plan_payload, artifact_count=1)
    return workspace


def _plan_like(payload: dict[str, Any]) -> AuthoritativeQueryPlan:
    from source.product.execution_planner import QueryFilter

    filters = [
        QueryFilter(column=str(item.get("column")), operator=str(item.get("operator") or "equals"), value=str(item.get("value")))
        for item in payload.get("filters", [])
        if isinstance(item, dict)
    ]
    return AuthoritativeQueryPlan(
        raw_question=str(payload.get("raw_question") or ""),
        intent=str(payload.get("intent") or "general"),
        metric=payload.get("metric"),
        dimension=payload.get("dimension"),
        time_axis=payload.get("time_axis"),
        chart_type=payload.get("chart_type"),
        aggregation=payload.get("aggregation"),
        transformation=payload.get("transformation"),
        filters=filters,
        extremum_scope=payload.get("extremum_scope"),
        target_entity=payload.get("target_entity"),
        aggregate_function=payload.get("aggregate_function"),
    )


def _branch_type(identity: BranchIdentity) -> str:
    routed = branch_type_for_intent(identity.intent)
    if routed != "analysis":
        return routed
    if identity.intent in {"temporal", "distribution", "grouped", "quality", "hypothesis", "shipping", "transformation"}:
        return identity.intent
    if identity.intent in {"shipping_delay"}:
        return "shipping"
    if identity.intent in {"histogram", "binning"} or identity.chart_type == "histogram":
        return "distribution"
    if identity.intent in {"seasonality_heatmap", "growth", "temporal_trend", "chart_request"} or identity.chart_type in {"line", "heatmap", "seasonality_heatmap"}:
        return "temporal"
    if identity.intent in {"quality"}:
        return "quality"
    if identity.intent in {"hypothesis"}:
        return "hypothesis"
    if identity.dimension:
        return "grouped"
    return "generic"


def _canonical_intent(plan: AuthoritativeQueryPlan) -> str:
    if plan.intent in {"temporal_trend", "chart_request", "growth", "seasonality_heatmap"} or plan.chart_type in {"line", "heatmap", "seasonality_heatmap"}:
        return "temporal"
    if plan.intent in {"histogram", "binning"} or plan.chart_type == "histogram":
        return "distribution"
    if plan.intent in {"rank_groups"} or plan.dimension:
        return "grouped"
    return str(plan.intent or "general")


def _canonical_aggregation(identity: BranchIdentity) -> str:
    if identity.aggregation:
        return str(identity.aggregation)
    if _branch_type(identity) == "temporal":
        return "mean"
    if _branch_type(identity) == "grouped":
        return "sum"
    return ""


def _canonical_chart_family(identity: BranchIdentity) -> str:
    if _branch_type(identity) == "temporal":
        return "trend"
    if _branch_type(identity) == "distribution":
        return "histogram"
    return str(identity.chart_type or "")


def _title_for_identity(identity: BranchIdentity) -> str:
    routed = title_for_intent(identity.intent)
    if routed != "Analysis" and routed != str(identity.intent or "").replace("_", " ").title():
        return routed
    if identity.intent == "shipping_delay":
        return "Shipping Delay Investigation"
    if identity.intent == "binning":
        return f"{identity.metric or 'Metric'} Bins"
    if identity.intent == "quality":
        return "Duplicate Quality Analysis"
    prefix = "Outlier-Adjusted " if identity.transformation in {"remove_extreme_records", "remove_outliers"} else ""
    if identity.metric and identity.chart_type == "histogram":
        filter_value = _first_filter_value(identity)
        return f"{prefix}{identity.metric} Distribution in {filter_value}" if filter_value else f"{prefix}{identity.metric} Distribution"
    if identity.metric and identity.time_axis:
        return f"{prefix}{identity.metric} Trend over Time"
    if identity.metric and identity.dimension:
        return f"{prefix}{identity.metric} by {identity.dimension}"
    if identity.metric and identity.chart_type:
        return f"{identity.metric} {identity.chart_type}"
    return identity.intent.replace("_", " ").title()


def _first_filter_value(identity: BranchIdentity) -> str | None:
    for _column, _op, value in identity.filters:
        if value:
            return value
    return None


def _branch_subtitle(identity: BranchIdentity, artifacts: int, findings: int) -> str:
    kind = _branch_type(identity).replace("_", " ")
    outputs = []
    if artifacts:
        outputs.append(f"{artifacts} chart/output" if artifacts == 1 else f"{artifacts} outputs")
    if findings:
        outputs.append(f"{findings} finding" if findings == 1 else f"{findings} findings")
    return f"{kind} · {', '.join(outputs) if outputs else 'latest'}"


def _branch_updated_at(branch: AnalyticalBranch) -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_for_branch(items: list[Any], branch_id: str) -> int:
    count = 0
    for item in items:
        metadata = getattr(item, "metadata", {}) if isinstance(getattr(item, "metadata", {}), dict) else {}
        if metadata.get("branch_id") == branch_id:
            count += 1
    return count
