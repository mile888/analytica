"""Create final answers from computed evidence only."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from source.product.plan_executor import EvidencePackage


@dataclass
class SynthesisResult:
    """Stores the final answer, findings, limits, and next steps."""

    answer: str
    findings: list[dict[str, str]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


def synthesize(
    question: str,
    evidence: EvidencePackage,
    language: str = "auto",
) -> SynthesisResult:
    """Use the LLM when available, otherwise use mechanical synthesis."""
    try:
        from source.llm.factory import make_llm
        llm = make_llm("semantic_planner")
        prompt = _build_synthesis_prompt(question, evidence, language)
        response = llm.invoke(prompt)
        text = _response_text(response)
        result = _parse_synthesis(text, evidence)
        if result:
            return result
    except Exception:
        pass

    return mechanical_synthesis(question, evidence)


_SYNTHESIS_SYSTEM = """\
You are a data analyst writing a concise, evidence-grounded answer.

STRICT RULES:
1. Use ONLY the computed evidence provided below. Do NOT invent numbers.
2. Do NOT reference columns or fields not present in the evidence.
3. If the evidence is insufficient to answer fully, say so explicitly.
4. Write in the user's language (detect from the question).
5. Be concise — prioritize the most important finding first.
6. Use exact numbers from the evidence.
7. Do NOT make causal claims unless the evidence supports them.
8. Output ONLY valid JSON matching the schema below. No markdown, no prose.

OUTPUT SCHEMA:
{
  "answer": "concise analytical answer with specific numbers",
  "findings": [
    {"title": "short finding title", "evidence": "supporting data from evidence", "importance": "high|medium|low"}
  ],
  "limitations": ["..."],
  "next_steps": ["..."]
}
"""


def _build_synthesis_prompt(
    question: str,
    evidence: EvidencePackage,
    language: str = "auto",
) -> str:
    evidence_json = json.dumps(evidence.to_dict(), ensure_ascii=False, default=str, indent=2)
    return (
        f"{_SYNTHESIS_SYSTEM}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"COMPUTED EVIDENCE:\n{evidence_json}\n\n"
        f"OUTPUT (JSON only):"
    )


# ── Parse LLM synthesis ────────────────────────────────────────────────────

def _parse_synthesis(text: str, evidence: EvidencePackage) -> SynthesisResult | None:
    import re
    stripped = text.strip()

    # Try to extract JSON
    if stripped.startswith("{"):
        json_str = stripped
    else:
        match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", stripped, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", stripped, re.DOTALL)
            json_str = match.group(0) if match else None

    if not json_str:
        return None

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict) or not data.get("answer"):
        return None

    findings = []
    for item in (data.get("findings") or []):
        if isinstance(item, dict) and item.get("title"):
            findings.append({
                "title": str(item.get("title", "")),
                "evidence": str(item.get("evidence", "")),
                "importance": str(item.get("importance", "medium")),
            })

    return SynthesisResult(
        answer=str(data["answer"]),
        findings=findings,
        limitations=[str(l) for l in (data.get("limitations") or evidence.limitations)],
        next_steps=[str(s) for s in (data.get("next_steps") or [])],
    )


# ── Mechanical synthesis (no LLM needed) ────────────────────────────────────

def mechanical_synthesis(question: str, evidence: EvidencePackage) -> SynthesisResult:
    """Generate answer directly from evidence without LLM."""
    answer_parts: list[str] = []
    findings: list[dict[str, str]] = []

    # Process computed tables
    for table in evidence.computed_tables:
        rows = table.get("rows", [])
        if not rows:
            continue
        name = table.get("name", "result")
        dim = table.get("dimension") or table.get("category_field") or table.get("group_column", "")

        if table.get("aggregation") == "count" or "count" in name.lower():
            # Count distribution summary
            top = rows[0]
            top_val = top.get("value", top.get("group", top.get("category", "")))
            top_count = top.get("count", 0)
            total = evidence.record_count
            share = round(top_count / max(total, 1) * 100, 1) if total else 0
            answer_parts.append(
                f"The most common {dim} is \"{top_val}\" with {top_count} records ({share}% of total)."
            )
            if len(rows) >= 3:
                top3 = ", ".join(
                    f"\"{r.get('value', r.get('group', r.get('category', '')))}\""
                    for r in rows[:3]
                )
                findings.append({
                    "title": f"Top {dim} categories",
                    "evidence": f"Top 3: {top3}",
                    "importance": "high",
                })
        elif "outcome_breakdown" in name.lower():
            top = max(rows, key=lambda row: float(row.get("share", 0)))
            target = table.get("target_column", "outcome")
            groups = table.get("grouping_columns", [])
            group_label = " / ".join(str(top.get(group, "")) for group in groups if group in top) or str(top.get("group", ""))
            answer_parts.append(
                f"The strongest observed {target} share is {top.get('share')}% for {group_label} ({top.get(target)})."
            )
            findings.append({
                "title": f"{target} varies by group",
                "evidence": f"{group_label}: {top.get('share')}% ({top.get('count')} of {top.get('group_total')})",
                "importance": "high",
            })
        elif "ordered_risk" in name.lower():
            top = rows[0]
            target = table.get("target_column", "outcome")
            ordered = table.get("ordered_column", "ordered field")
            answer_parts.append(
                f"The strongest increase in {target} risk as {ordered} decreases is for "
                f"{top.get('grouping_column')}=\"{top.get('group')}\": "
                f"{top.get('increase_as_order_decreases')} percentage points."
            )
            findings.append({
                "title": "Strongest ordered risk gradient",
                "evidence": (
                    f"{top.get('group')} changes from {top.get('risk_at_highest')}% at "
                    f"{top.get('highest_ordered_value')} to {top.get('risk_at_lowest')}% at "
                    f"{top.get('lowest_ordered_value')}"
                ),
                "importance": "high",
            })
        elif "shift" in name.lower():
            # Temporal shift
            shifts = table.get("shifts", [])
            if shifts:
                biggest = shifts[0]
                answer_parts.append(
                    f"Biggest category change: \"{biggest.get('category')}\" "
                    f"shifted by {biggest.get('change', 0):+.1f}pp."
                )
                findings.append({
                    "title": "Category mix shift",
                    "evidence": f"\"{biggest.get('category')}\" changed from {biggest.get('before_share', 0):.1f}% to {biggest.get('after_share', 0):.1f}%",
                    "importance": "high",
                })
        else:
            # Generic metric aggregation
            if rows:
                top = rows[0]
                group = top.get("group", top.get("value", ""))
                agg = table.get("aggregation", "value")
                grouping_columns = set(table.get("grouping_columns") or [])
                metric_val = top.get(agg)
                if metric_val is None:
                    metric_val = next(
                        (
                            v for k, v in top.items()
                            if k not in {"group", "value", "category"} and k not in grouping_columns
                        ),
                        None,
                    )
                if group and metric_val is not None:
                    answer_parts.append(f"Top {dim}: \"{group}\" with {agg} = {metric_val}.")
                    findings.append({
                        "title": f"Leading {dim}",
                        "evidence": f"\"{group}\": {metric_val}",
                        "importance": "high",
                    })

    # Process summary statistics
    stats = evidence.summary_statistics
    if stats and not answer_parts:
        if "metric" in stats:
            answer_parts.append(
                f"{stats['metric']}: mean={stats.get('mean')}, median={stats.get('median')}, "
                f"range=[{stats.get('min')}..{stats.get('max')}]"
            )
        elif "dataset_overview" in stats:
            overview = stats["dataset_overview"]
            answer_parts.append(
                f"Dataset contains {overview.get('rows', 0)} rows and {overview.get('columns', 0)} columns."
            )

    answer = " ".join(answer_parts) if answer_parts else "Analysis computed but no clear summary could be generated."

    # Ensure at least one finding if we have data
    if not findings and evidence.computed_tables:
        findings.append({
            "title": "Analysis completed",
            "evidence": f"Computed {len(evidence.computed_tables)} table(s) from {evidence.record_count} records",
            "importance": "medium",
        })

    return SynthesisResult(
        answer=answer,
        findings=findings,
        limitations=evidence.limitations,
        next_steps=["Drill down into the top categories", "Check for outliers or data quality issues"],
    )


# ── Utility ─────────────────────────────────────────────────────────────────

def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(str(item.get("text") if isinstance(item, dict) else item) for item in content)
    return str(content or "")
