from __future__ import annotations

import pandas as pd

from source.product.evidence_resolution import (
    ActiveAnalyticalTarget,
    EvidenceGraph,
    EvidenceTarget,
    ClaimEvidence,
    evidence_response_text,
    resolve_evidence_subject,
)
from source.product.fallback_analysis import deterministic_investigation_fallback


def test_sparse_city_evidence_followup_stays_attached_to_latest_hypothesis() -> None:
    df = pd.DataFrame(
        {
            "City": ["Jamestown", "Cheyenne", "Bellingham", "Boston"],
            "Sales": [2354.39, 1603.14, 1263.41, 200.0],
        }
    )
    active_target = ActiveAnalyticalTarget(
        branch_id="run_latest",
        branch_type="hypothesis_validation",
        metric="Sales",
        dimension="City",
        hypothesis="Some cities look strong only because of sparse data.",
        evidence_subject="The hypothesis is supported: some strong-looking `City` groups are sparse.",
        active_entities=["Sales", "City"],
        active_mechanism="sparse_group_instability",
        response_language="ru",
    ).to_payload()
    data_context = {
        "conversation_context": {
            "conversation_state": {
                "active_metric": "Sales",
                "active_dimension": "City",
                "active_branch_type": "hypothesis_validation",
                "active_analytical_target": active_target,
            },
            "latest_findings": ["`Home Office` нужно проверить отдельно: широкий разброс может отражать выбросы."],
            "recent_artifacts": [
                {
                    "artifact_type": "table",
                    "content": [
                        {"City": "Jamestown", "count": 2, "mean": 2354.39, "is_sparse": True},
                        {"City": "Cheyenne", "count": 1, "mean": 1603.14, "is_sparse": True},
                    ],
                    "metadata": {
                        "metric": "Sales",
                        "dimension": "City",
                        "mechanism": "sparse_group_instability",
                    },
                }
            ],
        }
    }

    result = deterministic_investigation_fallback("Какие evidence supports this?", df, data_context=data_context)
    answer = result["final_answer"]

    assert "City" in answer
    assert "sample-size instability" in answer
    assert "Home Office" not in answer
    assert answer.startswith("`")
    assert result["trace_metadata"]["fallback"] == "evidence_resolution"


def test_evidence_resolution_prefers_active_target_over_stale_findings() -> None:
    active_target = ActiveAnalyticalTarget(
        branch_id="new",
        branch_type="hypothesis_validation",
        metric="Sales",
        dimension="City",
        active_entities=["Sales", "City"],
        active_mechanism="sparse_group_instability",
    )

    resolution = resolve_evidence_subject(
        "What evidence supports this?",
        active_target.to_payload(),
        branch_state={"active_branch_type": "hypothesis_validation"},
        recent_findings=["Home Office has wide spread."],
    )

    assert resolution.target is not None
    assert resolution.target.dimension == "City"
    assert resolution.reason == "active_target"
    assert resolution.score > 1.0


def test_standard_class_target_stays_grounded_for_evidence_text() -> None:
    target = ActiveAnalyticalTarget(
        branch_type="hypothesis_validation",
        metric="Sales",
        dimension="Ship Mode",
        active_entities=["Sales", "Ship Mode", "Standard Class"],
        active_mechanism="volume_effect",
        evidence_subject="`Standard Class` dominance was tested by volume and value.",
    )

    answer = evidence_response_text(
        "What evidence supports this?",
        target,
        artifact_rows=[
            {"Ship Mode": "Standard Class", "record_count": 4, "total": 1000.0, "mean": 250.0},
            {"Ship Mode": "Same Day", "record_count": 1, "total": 400.0, "mean": 400.0},
        ],
    )

    assert "`Ship Mode`" in answer
    assert "volume-driven" in answer
    assert "`Segment`" not in answer


def test_duplicate_quality_next_check_continues_quality_branch() -> None:
    df = pd.DataFrame({"Order ID": ["a", "a", "b"], "Sales": [10.0, 10.0, 20.0]})
    active_target = ActiveAnalyticalTarget(
        branch_type="data_quality",
        metric="Sales",
        active_entities=["Sales", "Order ID"],
        active_mechanism="duplicate_inflation",
        response_language="ru",
    ).to_payload()

    result = deterministic_investigation_fallback(
        "Что проверить дополнительно?",
        df,
        data_context={
            "conversation_context": {
                "conversation_state": {
                    "active_branch_type": "data_quality",
                    "active_metric": "Sales",
                    "active_analytical_target": active_target,
                }
            }
        },
    )

    assert "duplicate-quality branch" in result["final_answer"]
    assert "duplicates" in result["final_answer"]
    assert "Segment" not in result["final_answer"]
    assert result["trace_metadata"]["fallback"] == "quality_branch_continuity"


def test_no_unrelated_branch_leakage_for_target_evidence() -> None:
    target = ActiveAnalyticalTarget(
        branch_type="hypothesis_validation",
        metric="Defect Rate",
        dimension="Factory",
        active_entities=["Defect Rate", "Factory"],
        active_mechanism="sparse_group_instability",
    )

    answer = evidence_response_text("Which evidence supports this?", target)

    assert "`Factory`" in answer
    assert "`Defect Rate`" in answer
    assert "`City`" not in answer
    assert "`Sales`" not in answer


def test_evidence_graph_structures_link_claims_to_artifacts() -> None:
    graph = EvidenceGraph(
        active_target=EvidenceTarget(
            target_id="claim_1",
            target_type="hypothesis",
            metric="Sales",
            dimension="Ship Mode",
            mechanism="volume_effect",
            text="Standard Class dominance is volume-driven.",
        ),
        claim_evidence=[
            ClaimEvidence(
                claim_id="claim_1",
                evidence_id="artifact_1",
                evidence_type="table",
                role="supports",
                metric="Sales",
                dimension="Ship Mode",
                mechanism="volume_effect",
                strength="medium",
            )
        ],
    )

    payload = graph.to_payload()

    assert payload["active_target"]["dimension"] == "Ship Mode"
    assert payload["claim_evidence"][0]["evidence_id"] == "artifact_1"
