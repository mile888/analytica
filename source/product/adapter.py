from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from source.product.investigation import Artifact, ArtifactType, ArtifactVisibility, DecisionReport, Finding
from source.product.insight_quality import build_insight_metadata


@dataclass
class InvestigationUpdate:
    artifacts: list[Artifact] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    report: DecisionReport | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


def agent_output_to_investigation_update(output: dict[str, Any] | None) -> InvestigationUpdate:
    """Translate current agent output into product-facing investigation objects.

    The active agent runtime returns a compatibility dict. This adapter is
    deliberately defensive because older/newer runners may omit fields.
    """

    data = output if isinstance(output, dict) else {}
    structured = data.get("structured_report") if isinstance(data.get("structured_report"), dict) else {}
    summary = _first_text(data.get("summary"), structured.get("summary"), data.get("final_answer"))
    raw_key_findings = _text_list(structured.get("key_findings") or data.get("key_findings"))
    suppress_key_findings = _suppresses_key_findings(data, structured)
    key_findings = [] if suppress_key_findings else [
        item for item in raw_key_findings if not _is_profile_observation(item)
    ]
    limitations = _text_list(structured.get("limitations") or data.get("limitations"))
    next_steps = _text_list(structured.get("next_steps") or data.get("next_steps"))
    generated_code = _first_text(data.get("generated_code"), structured.get("generated_code"), data.get("code"))
    sql_metadata = data.get("sql_metadata") if isinstance(data.get("sql_metadata"), dict) else {}
    trace_metadata = data.get("trace_metadata") if isinstance(data.get("trace_metadata"), dict) else {}
    tool_timeline = data.get("tool_timeline") if isinstance(data.get("tool_timeline"), list) else []
    result_preview = _first_text(data.get("result_preview"))
    agent_artifacts = _coerce_agent_artifacts(data.get("artifacts") or structured.get("artifacts"))

    update = InvestigationUpdate()
    execution_context_unavailable = bool(trace_metadata.get("execution_context_unavailable"))

    if summary or key_findings or limitations or next_steps:
        content = _report_content(summary, key_findings, limitations, next_steps)
        update.report = DecisionReport(
            summary=summary,
            question=_first_text(data.get("query"), structured.get("question")),
            answer=summary,
            key_findings=key_findings,
            evidence=_text_list(structured.get("evidence") or data.get("evidence")),
            limitations=limitations,
            next_steps=next_steps,
            content=content,
            metadata={"source": "agent_output", "trace_metadata": trace_metadata},
        )
        if not execution_context_unavailable:
            update.artifacts.append(
                Artifact(
                    artifact_type=ArtifactType.REPORT,
                    title="Decision report",
                    content=content,
                    visibility=ArtifactVisibility.USER,
                    pinned=True,
                    metadata={"source": "agent_output"},
                )
            )

    if summary and not execution_context_unavailable:
        update.artifacts.append(
            Artifact(
                artifact_type=ArtifactType.TEXT,
                title="Summary",
                content=summary,
                visibility=ArtifactVisibility.USER,
            )
        )

    user_artifact_ids = [artifact.artifact_id for artifact in update.artifacts if artifact.visibility == ArtifactVisibility.USER]
    for item in key_findings:
        update.findings.append(
            Finding(
                text=item,
                title=_insight_title(item),
                confidence=_confidence_value(
                    build_insight_metadata(
                        item,
                        evidence=update.report.evidence if update.report else [],
                        limitations=limitations,
                        next_steps=next_steps,
                        artifact_ids=user_artifact_ids,
                        analysis_context=trace_metadata,
                    ).get("confidence_level")
                ),
                metadata=build_insight_metadata(
                    item,
                    evidence=update.report.evidence if update.report else [],
                    limitations=limitations,
                    next_steps=next_steps,
                    artifact_ids=user_artifact_ids,
                    analysis_context=trace_metadata,
                ),
            )
        )

    if not suppress_key_findings:
        for item in limitations:
            update.findings.append(
                Finding(
                    text=item,
                    title="Limitation",
                    confidence=0.35,
                    metadata=build_insight_metadata(
                        item,
                        evidence=update.report.evidence if update.report else [],
                        limitations=[item],
                        next_steps=next_steps,
                        artifact_ids=user_artifact_ids,
                        kind="limitation",
                        analysis_context=trace_metadata,
                    ),
                )
            )

    if execution_context_unavailable:
        return update

    if generated_code:
        update.artifacts.append(
            Artifact(
                artifact_type=ArtifactType.PYTHON_CODE,
                title="Generated Python",
                content=generated_code,
                visibility=ArtifactVisibility.TECHNICAL,
            )
        )

    sql_artifact = _sql_artifact(sql_metadata)
    if sql_artifact:
        update.artifacts.append(sql_artifact)

    has_table_artifact = any(artifact.artifact_type == ArtifactType.TABLE for artifact in agent_artifacts)
    if result_preview and not has_table_artifact:
        update.artifacts.append(
            Artifact(
                artifact_type=ArtifactType.TABLE,
                title="Result preview",
                content=result_preview,
                visibility=ArtifactVisibility.USER,
            )
        )

    for artifact in agent_artifacts:
        update.artifacts.append(artifact)

    if tool_timeline:
        update.trace.extend(_dict_list(tool_timeline))
        update.artifacts.append(
            Artifact(
                artifact_type=ArtifactType.VALIDATION,
                title="Run trace",
                content=tool_timeline,
                visibility=ArtifactVisibility.TECHNICAL,
                metadata={"source": "tool_timeline"},
            )
        )

    return update


def _insight_title(text: str) -> str:
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return "Insight"
    if len(cleaned) <= 78:
        return cleaned.rstrip(".")
    return cleaned[:75].rstrip(" .,") + "..."


def _confidence_value(label: Any) -> float | None:
    normalized = str(label or "").lower()
    if normalized == "high":
        return 0.85
    if normalized == "medium":
        return 0.65
    if normalized == "low":
        return 0.4
    return None


def _suppresses_key_findings(data: dict[str, Any], structured: dict[str, Any]) -> bool:
    trace_metadata = data.get("trace_metadata") if isinstance(data.get("trace_metadata"), dict) else {}
    analysis_type = str(
        trace_metadata.get("analysis_type")
        or structured.get("analysis_type")
        or data.get("analysis_type")
        or ""
    ).lower()
    return bool(trace_metadata.get("suppress_key_findings")) or analysis_type in {"overview", "profile", "suggestion"}


def _is_profile_observation(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    markers = (
        "dataset has enough structure",
        "enough structure for analytical",
        "good candidates for quantitative analysis",
        "candidate metric",
        "candidate metrics",
        "can anchor quantitative analysis",
        "can explain differences between groups",
        "useful for segmentation",
        "supports trend analysis",
        "enables trend",
        "identifier columns should not",
        "should not be treated as metrics",
        "provides the grouping",
        "provides the comparison metric",
        "can be reviewed for distribution shape",
        "has enough numeric companions",
    )
    return any(marker in normalized for marker in markers)


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        else:
            text = str(value).strip()
            if text:
                return text
    return ""


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [_first_text(item) for item in value if _first_text(item)]
    return [_first_text(value)] if _first_text(value) else []


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _report_content(
    summary: str,
    key_findings: list[str],
    limitations: list[str],
    next_steps: list[str],
) -> str:
    lines: list[str] = []
    if summary:
        lines.extend(["## Summary", summary])
    if key_findings:
        lines.append("## Key insights")
        lines.extend(f"- {item}" for item in key_findings)
    if limitations:
        lines.append("## Limitations")
        lines.extend(f"- {item}" for item in limitations)
    if next_steps:
        lines.append("## Next steps")
        lines.extend(f"- {item}" for item in next_steps)
    return "\n\n".join(lines).strip()


def _sql_artifact(sql_metadata: dict[str, Any]) -> Artifact | None:
    if not sql_metadata:
        return None
    sql = _first_text(
        sql_metadata.get("query"),
        sql_metadata.get("sql"),
        sql_metadata.get("last_query"),
        sql_metadata.get("checked_query"),
    )
    return Artifact(
        artifact_type=ArtifactType.SQL,
        title="SQL context",
        content=sql or sql_metadata,
        visibility=ArtifactVisibility.TECHNICAL,
        metadata=sql_metadata,
    )


def _coerce_agent_artifacts(value: Any) -> list[Artifact]:
    if not isinstance(value, list):
        return []
    artifacts: list[Artifact] = []
    for idx, item in enumerate(value, start=1):
        if isinstance(item, Artifact):
            artifacts.append(item)
            continue
        if not isinstance(item, dict):
            artifacts.append(
                Artifact(
                    artifact_type=ArtifactType.UNKNOWN,
                    title=f"Agent artifact {idx}",
                    content=item,
                    visibility=ArtifactVisibility.TECHNICAL,
                )
            )
            continue
        artifact_type = _artifact_type_from_value(item.get("artifact_type") or item.get("type"))
        artifacts.append(
            Artifact(
                artifact_type=artifact_type,
                title=str(item.get("title") or item.get("path") or f"Agent artifact {idx}"),
                content=item.get("content") or item.get("data") or item,
                path=str(item["path"]) if item.get("path") else None,
                visibility=_visibility_from_value(item.get("visibility"), artifact_type),
                pinned=bool(item.get("pinned", False)),
                metadata={k: v for k, v in item.items() if k not in {"content", "data"}},
            )
        )
    return artifacts


def _artifact_type_from_value(value: Any) -> ArtifactType:
    if isinstance(value, ArtifactType):
        return value
    text = str(value or "").strip().lower()
    aliases = {
        "code": ArtifactType.PYTHON_CODE,
        "python": ArtifactType.PYTHON_CODE,
        "markdown": ArtifactType.REPORT,
        "md": ArtifactType.REPORT,
    }
    if text in aliases:
        return aliases[text]
    try:
        return ArtifactType(text)
    except ValueError:
        return ArtifactType.UNKNOWN


def _visibility_for_type(artifact_type: ArtifactType) -> ArtifactVisibility:
    if artifact_type in {ArtifactType.TABLE, ArtifactType.CHART, ArtifactType.REPORT, ArtifactType.TEXT}:
        return ArtifactVisibility.USER
    if artifact_type in {ArtifactType.PYTHON_CODE, ArtifactType.SQL, ArtifactType.VALIDATION}:
        return ArtifactVisibility.TECHNICAL
    return ArtifactVisibility.TECHNICAL


def _visibility_from_value(value: Any, artifact_type: ArtifactType) -> ArtifactVisibility:
    try:
        return ArtifactVisibility(str(value))
    except (TypeError, ValueError):
        return _visibility_for_type(artifact_type)
