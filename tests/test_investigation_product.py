from __future__ import annotations

import pytest
import pandas as pd

from source.product.adapter import agent_output_to_investigation_update
from source.product.investigation import Artifact, ArtifactType, InvestigationStatus
from source.product.service import InvestigationService
from source.product.store import InvestigationStore


FORBIDDEN_ANALYSIS_NARRATION = (
    "to identify",
    "you can compare",
    "a useful next step",
    "should focus",
    "to investigate",
    "can be analyzed",
    "supports analysis",
    "potential grouping",
    "next analytical move",
    "useful analysis",
    "should be compared",
)


def assert_no_methodology_narration(text: str) -> None:
    lowered = text.lower()
    for phrase in FORBIDDEN_ANALYSIS_NARRATION:
        assert phrase not in lowered


def test_create_investigation() -> None:
    store = InvestigationStore()

    investigation = store.create_investigation("Why did conversion drop?")

    assert investigation.user_question == "Why did conversion drop?"
    assert investigation.title == "Why did conversion drop?"
    assert investigation.status == InvestigationStatus.DRAFT
    assert store.get_investigation(investigation.investigation_id) == investigation


def test_add_artifact() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Explain the metric change")
    artifact = Artifact(
        artifact_type=ArtifactType.TEXT,
        title="Summary",
        content="The metric increased in one group.",
    )

    updated = store.add_artifact(investigation.investigation_id, artifact)

    assert updated.artifacts == [artifact]
    assert updated.updated_at >= updated.created_at


def test_adapter_handles_empty_output() -> None:
    update = agent_output_to_investigation_update({})

    assert update.artifacts == []
    assert update.findings == []
    assert update.report is None
    assert update.trace == []


def test_adapter_maps_summary_key_findings_generated_code() -> None:
    update = agent_output_to_investigation_update(
        {
            "structured_report": {
                "summary": "Conversion dropped in paid traffic.",
                "key_findings": ["Paid traffic CVR fell by 12%."],
                "limitations": ["No campaign spend data."],
            },
            "generated_code": "result = df.head()",
            "tool_timeline": [{"tool": "inspect_dataset_schema", "status": "ok"}],
        }
    )

    assert update.report is not None
    assert update.report.summary == "Conversion dropped in paid traffic."
    assert [finding.text for finding in update.findings] == [
        "Paid traffic CVR fell by 12%.",
        "No campaign spend data.",
    ]
    assert any(artifact.artifact_type == ArtifactType.PYTHON_CODE for artifact in update.artifacts)
    assert any(artifact.artifact_type == ArtifactType.REPORT for artifact in update.artifacts)
    assert any(artifact.artifact_type == ArtifactType.VALIDATION for artifact in update.artifacts)
    assert update.trace == [{"tool": "inspect_dataset_schema", "status": "ok"}]


def test_adapter_adds_analyst_quality_metadata_to_insights() -> None:
    update = agent_output_to_investigation_update(
        {
            "structured_report": {
                "summary": "Salary differs by role.",
                "key_findings": ["AI engineering roles show wider salary variance."],
                "evidence": ["Grouped salary distribution by role."],
                "limitations": ["Some groups have small samples."],
                "next_steps": ["Compare variance by industry."],
            },
            "artifacts": [
                {
                    "artifact_type": "chart",
                    "title": "Average Salary_LPA by Job_Title",
                    "content": {"chart_type": "bar", "rows": []},
                    "visibility": "user",
                }
            ],
        }
    )

    finding = update.findings[0]

    assert finding.title == "AI engineering roles show wider salary variance"
    assert finding.metadata["confidence_level"] in {"High", "Medium"}
    assert finding.metadata["evidence_strength"] in {"high", "medium"}
    assert finding.metadata["business_impact"] == "High"
    assert finding.metadata["conclusion"] == "AI engineering roles show wider salary variance."
    assert finding.metadata["business_implication"]
    assert finding.metadata["confidence_reason"]
    assert finding.metadata["evidence_reason"]
    assert finding.metadata["limitation"] == "Some groups have small samples."
    assert finding.metadata["recommended_validation"] == "Compare variance by industry."
    assert finding.metadata["analysis_type"] in {"grouped_metric", "overview"}
    assert finding.metadata["recommended_next_step"] == "Compare variance by industry."
    assert finding.metadata["confidence"] == finding.metadata["confidence_level"]
    assert finding.metadata["related_outputs"]
    assert "evidence" in finding.metadata


def test_adapter_filters_profile_observations_out_of_findings() -> None:
    update = agent_output_to_investigation_update(
        {
            "structured_report": {
                "summary": "This dataset contains useful fields for analysis.",
                "key_findings": [
                    "The dataset has enough structure for analytical questions.",
                    "Metric values differ strongly across category labels.",
                ],
            },
            "trace_metadata": {"analysis_type": "grouped_metric"},
        }
    )

    assert [finding.text for finding in update.findings] == [
        "Metric values differ strongly across category labels."
    ]


def test_adapter_suppresses_overview_findings() -> None:
    update = agent_output_to_investigation_update(
        {
            "structured_report": {
                "summary": "The dataset is ready for exploration.",
                "key_findings": ["metric_value can anchor quantitative analysis."],
            },
            "trace_metadata": {"analysis_type": "overview", "suppress_key_findings": True},
        }
    )

    assert update.findings == []
    assert update.report is not None
    assert update.report.key_findings == []


def test_service_marks_failed_on_runner_exception() -> None:
    def failing_runner(**kwargs):
        raise RuntimeError("runner exploded")

    service = InvestigationService(runner=failing_runner)
    investigation = service.create_investigation("Why did churn spike?")

    updated = service.run_investigation(investigation.investigation_id, df=None)

    assert updated.status == InvestigationStatus.FAILED
    assert updated.runs[-1].status == InvestigationStatus.FAILED
    assert updated.runs[-1].error == "RuntimeError: runner exploded"
    assert updated.artifacts[-1].artifact_type == ArtifactType.VALIDATION
    assert "runner exploded" in updated.artifacts[-1].content


def test_service_marks_failed_on_runner_error_without_fallback() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Why did churn spike?")

    updated = service.run_investigation(investigation.investigation_id, df=None)

    assert updated.status == InvestigationStatus.FAILED
    assert "analytics tool result" in updated.runs[-1].error


def test_service_uses_deterministic_fallback_for_metric_by_category() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "category_label": ["Group A", "Group A", "Group B"],
            "metric_value": [10.0, 5.0, 30.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Почему значение метрики отличается по группам?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    assert updated.status == InvestigationStatus.NEEDS_REVIEW
    assert updated.findings
    assert "Group B" in updated.findings[0].text
    assert "не доказывает причинность" in updated.report.limitations[0] or "does not prove causality" in updated.report.limitations[0]
    assert any(artifact.artifact_type == ArtifactType.TABLE for artifact in updated.artifacts)
    assert updated.trace[0]["tool"] == "deterministic_pandas_fallback"


def test_service_fallback_handles_single_numeric_metric_without_dimension() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "metric_value": [100.0, 250.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Суммаризируй этот датасет.")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    assert updated.status == InvestigationStatus.NEEDS_REVIEW
    assert updated.findings == []
    assert updated.report is not None
    assert "`metric_value`" in updated.report.summary
    assert "useful first analysis" in updated.report.summary.lower()
    assert updated.trace[0]["tool"] == "dataset_overview"


def test_service_fallback_handles_metric_by_dimension() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "dimension_label": ["Group A", "Group B", "Group A"],
            "metric_value": [100.0, 250.0, 50.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Как группы связаны со значением метрики?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    assert updated.status == InvestigationStatus.NEEDS_REVIEW
    assert updated.findings
    assert "`dimension_label`" in updated.findings[0].text
    assert "`metric_value`" in updated.findings[0].text
    table_artifact = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.TABLE)
    assert isinstance(table_artifact.content, list)
    assert table_artifact.content[0]["dimension_label"] == "Group B"
    assert updated.trace[0]["tool"] == "deterministic_pandas_fallback"


def test_service_uses_context_fallback_when_raw_dataframe_is_unavailable() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "OperationalError: near ')': syntax error", "critic_verdict": "ERROR"}

    data_context = {
        "data_source_usage_contexts": [
            {
                "name": "generic dataset",
                "schema_summary": {"row_count": 5000, "column_count": 4},
                "column_summaries": [
                    {"name": "entity_id", "dtype": "int64", "inferred_role": "identifier"},
                    {"name": "category_label", "dtype": "object", "inferred_role": "dimension"},
                    {"name": "metric_value", "dtype": "float64", "inferred_role": "metric"},
                    {"name": "event_date", "dtype": "object", "inferred_role": "timestamp"},
                ],
                "caveats": ["Raw CSV file is not linked for execution."],
            }
        ]
    }
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Что ты можешь сказать об этом датасете?")

    updated = service.run_investigation(investigation.investigation_id, df=None, data_context=data_context)

    assert updated.status == InvestigationStatus.NEEDS_REVIEW
    assert updated.findings == []
    assert "5,000" in updated.report.summary
    assert "metric_value" in updated.report.summary
    assert updated.trace[0]["tool"] == "data_source_context_fallback"


def test_overview_question_does_not_create_profile_key_findings() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "entity_id": [1, 2, 3, 4],
            "category_label": ["A", "A", "B", "C"],
            "metric_value": [10.0, 11.0, 30.0, 5.0],
            "event_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("что ты можешь сказать о данных")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    assert updated.findings == []
    assert updated.report is not None
    assert "structured tabular dataset" in updated.report.summary
    assert "Strong starting directions" in updated.report.summary
    assert "enough structure" not in updated.report.summary.lower()


def test_context_fallback_answers_follow_up_about_findings_evidence() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "OperationalError: near ')': syntax error", "critic_verdict": "ERROR"}

    data_context = {
        "data_source_usage_contexts": [
            {
                "name": "generic dataset",
                "schema_summary": {"row_count": 120, "column_count": 3},
                "column_summaries": [
                    {"name": "entity_id", "dtype": "int64", "inferred_role": "identifier"},
                    {"name": "category_label", "dtype": "object", "inferred_role": "dimension"},
                    {"name": "metric_value", "dtype": "float64", "inferred_role": "metric"},
                ],
            }
        ],
        "conversation_context": {
            "latest_findings": [
                "Metric values differ strongly across category labels.",
                "Some entity records appear unusual compared with peers.",
            ],
            "artifact_titles": ["Metric by category table"],
        },
    }
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Which current findings need more evidence?")

    updated = service.run_investigation(investigation.investigation_id, df=None, data_context=data_context)

    assert updated.status == InvestigationStatus.NEEDS_REVIEW
    assert "need evidence" in updated.report.summary.lower()
    assert "Metric values differ" in updated.findings[0].text
    assert updated.trace[0]["tool"] == "data_source_context_findings_review"


def test_context_fallback_continues_follow_up_instead_of_repeating_overview() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "runner unavailable", "critic_verdict": "ERROR"}

    data_context = {
        "data_source_usage_contexts": [
            {
                "name": "generic dataset",
                "schema_summary": {"row_count": 120, "column_count": 3},
                "column_summaries": [
                    {"name": "category_label", "dtype": "object", "inferred_role": "dimension"},
                    {"name": "metric_value", "dtype": "float64", "inferred_role": "metric"},
                ],
            }
        ],
        "conversation_context": {
            "active_message_id": "msg_followup",
            "latest_findings": ["Metric values differ strongly across category labels."],
            "artifact_titles": ["Metric by category table"],
        },
    }
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Show this as a chart.")

    updated = service.run_investigation(investigation.investigation_id, df=None, data_context=data_context)

    assert updated.status == InvestigationStatus.NEEDS_REVIEW
    assert "continuation" not in updated.report.summary.lower()
    assert "orchestration" not in updated.report.summary.lower()
    assert "workflow" not in updated.report.summary.lower()
    assert "focused comparison" in updated.report.summary.lower() or "chart" in updated.report.summary.lower()
    assert "120 rows" not in updated.report.summary
    assert updated.trace[0]["tool"] == "profile_based_analytical_answer"


def test_context_fallback_answers_unusual_group_question_without_meta_narration() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "runner unavailable", "critic_verdict": "ERROR"}

    data_context = {
        "data_source_usage_contexts": [
            {
                "name": "careers",
                "schema_summary": {"row_count": 5000, "column_count": 17},
                "column_summaries": [
                    {"name": "Job_Title", "dtype": "object", "inferred_role": "dimension"},
                    {"name": "Salary_LPA", "dtype": "float64", "inferred_role": "metric"},
                ],
            }
        ],
        "conversation_context": {
            "active_message_id": "msg_followup",
            "latest_findings": ["Salary_LPA varies across Job_Title."],
        },
    }
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Which `Job_Title` groups have unusual `Salary_LPA` values?")

    updated = service.run_investigation(investigation.investigation_id, df=None, data_context=data_context)

    summary = updated.report.summary
    assert "Job_Title" in summary
    assert "Salary_LPA" in summary
    assert "outlier" in summary.lower() or "unusual" in summary.lower()
    assert "most relevant current insight" not in summary.lower()
    assert "context" not in summary.lower()
    assert "workflow" not in summary.lower()
    assert updated.trace[0]["tool"] == "profile_based_analytical_answer"


def test_deterministic_fallback_answers_salary_by_job_title_without_orchestration() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "Job_Title": ["AI Engineer", "AI Engineer", "Designer", "Analyst"],
            "Salary_LPA": [80.0, 120.0, 30.0, 55.0],
            "Date_Posted": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("How does Salary_LPA vary across Job_Title?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    summary = updated.report.summary
    assert "Salary_LPA" in summary
    assert "Job_Title" in summary
    assert "AI Engineer" in summary
    assert "100.00" in summary
    assert "30.00" in summary
    assert "Fallback analysis" not in summary
    assert "continuation" not in summary.lower()
    assert_no_methodology_narration(summary)
    assert any(artifact.artifact_type == ArtifactType.CHART for artifact in updated.artifacts)
    assert updated.findings[0].metadata["conclusion"]
    assert updated.findings[0].metadata["recommended_validation"]
    assert updated.findings[0].metadata["business_implication"]
    assert updated.findings[0].metadata["analysis_type"] == "grouped_metric"


def test_deterministic_fallback_adds_reasoning_depth_to_grouped_analysis() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "Job_Title": ["AI Engineer", "AI Engineer", "Designer", "Designer", "Analyst", "Analyst"],
            "Salary_LPA": [80.0, 120.0, 30.0, 35.0, 55.0, 60.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("How does Salary_LPA vary across Job_Title?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    summary = updated.report.summary
    assert "may reflect" in summary
    assert "hidden subgroup" in summary
    assert "Evidence" in " ".join(updated.report.limitations)
    assert_no_methodology_narration(summary)


def test_deterministic_fallback_answers_unusual_groups_with_results_not_methodology() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "Job_Title": [
                "Research Scientist",
                "Research Scientist",
                "Research Scientist",
                "NLP Engineer",
                "NLP Engineer",
                "NLP Engineer",
                "UI Designer",
                "UI Designer",
                "UI Designer",
            ],
            "Salary_LPA": [75.0, 150.0, 82.0, 88.0, 90.0, 92.0, 40.0, 42.0, 43.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Which Job_Title groups have unusual Salary_LPA values?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    summary = updated.report.summary
    assert "Research Scientist" in summary
    assert "widest spread" in summary.lower()
    assert "150.00" in summary
    assert_no_methodology_narration(summary)
    assert updated.trace[0]["tool"] == "group_unusual_values_check"
    assert any(artifact.artifact_type == ArtifactType.CHART for artifact in updated.artifacts)


def test_deterministic_fallback_answers_affects_question_with_correlation() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "Salary_LPA": [50.0, 60.0, 70.0, 90.0, 110.0],
            "Company_Rating": [2.0, 2.5, 3.0, 4.0, 5.0],
            "Applicants": [100, 90, 70, 45, 20],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("What affects Salary_LPA most?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    summary = updated.report.summary
    assert "Company_Rating" in summary
    assert "correlation" in summary.lower()
    assert "not a causal result" in summary
    assert "segment mix" in summary
    assert_no_methodology_narration(summary)
    assert updated.trace[0]["tool"] == "correlation_check"
    assert "causality" in updated.findings[0].metadata["limitation"].lower()


def test_chart_intent_ranking_uses_grouped_bar_even_with_date_column() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "Job_Title": ["AI Engineer", "AI Engineer", "Designer", "Analyst"],
            "Salary_LPA": [80.0, 120.0, 30.0, 55.0],
            "Date_Posted": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("build a chart of top professions by salary")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)
    assert chart.content["chart_type"] == "bar"
    assert chart.content["x"] == "Job_Title"
    assert chart.content["metric"] == "Salary_LPA"
    assert updated.trace[0]["tool"] == "deterministic_pandas_fallback"


def test_russian_chart_request_creates_grouped_bar_not_overview() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "City": ["Moscow", "Moscow", "Kazan", "Sochi"],
            "Sales": [1200.0, 800.0, 500.0, 700.0],
            "Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("построй график по sales в разных городах")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)
    assert chart.content["chart_type"] == "bar"
    assert chart.content["x"] == "City"
    assert chart.content["metric"] == "Sales"
    assert "Moscow" in updated.report.summary
    assert "ready for investigation" not in updated.report.summary
    assert "structure" not in updated.report.summary.lower()
    assert updated.trace[0]["tool"] == "deterministic_pandas_fallback"


def test_russian_city_chart_request_prefers_city_over_other_categories() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "City": ["Moscow", "Moscow", "Kazan", "Sochi"],
            "Ship Mode": ["Second Class", "Same Day", "Standard Class", "First Class"],
            "Sales": [1200.0, 800.0, 500.0, 700.0],
            "Order Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("построй график sales по городам")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)
    assert chart.content["chart_type"] == "bar"
    assert chart.content["x"] == "City"
    assert chart.content["metric"] == "Sales"
    assert "Moscow" in updated.report.summary
    assert "Ship Mode" not in chart.title


def test_follow_up_uses_latest_chart_context_for_metric_and_city_dimension() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "City": ["Moscow", "Moscow", "Kazan", "Sochi"],
            "Ship Mode": ["Second Class", "Same Day", "Standard Class", "First Class"],
            "Sales": [1200.0, 800.0, 500.0, 700.0],
            "Profit": [200.0, 100.0, 50.0, 80.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Какие города самые сильные?")

    updated = service.run_investigation(
        investigation.investigation_id,
        df=df,
        data_context={
            "conversation_context": {
                "latest_chart_context": {
                    "metric": "Sales",
                    "dimension": "City",
                    "chart_type": "bar",
                    "title": "Average Sales by City",
                }
            }
        },
    )

    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)
    assert chart.content["x"] == "City"
    assert chart.content["metric"] == "Sales"
    assert "Moscow" in updated.report.summary
    assert "Ship Mode" not in updated.report.summary


def test_follow_up_anomaly_uses_latest_chart_context() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "City": ["Moscow", "Moscow", "Kazan", "Sochi"],
            "Ship Mode": ["Second Class", "Same Day", "Standard Class", "First Class"],
            "Sales": [1200.0, 800.0, 500.0, 700.0],
            "Profit": [200.0, 100.0, 50.0, 80.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Есть ли аномалии?")

    updated = service.run_investigation(
        investigation.investigation_id,
        df=df,
        data_context={
            "conversation_context": {
                "latest_chart_context": {
                    "metric": "Sales",
                    "dimension": "City",
                    "chart_type": "bar",
                    "title": "Average Sales by City",
                }
            }
        },
    )

    assert "City" in updated.report.summary
    assert "Sales" in updated.report.summary
    assert "Ship Mode" not in updated.report.summary
    assert updated.trace[0]["tool"] == "group_unusual_values_check"


def test_follow_up_volume_question_continues_latest_chart_thread() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "City": ["Moscow", "Moscow", "Moscow", "Kazan", "Kazan", "Sochi"],
            "Sales": [1000.0, 900.0, 1100.0, 300.0, 350.0, 800.0],
            "Order_ID": [1, 2, 3, 4, 5, 6],
            "Ship Mode": ["A", "A", "B", "A", "B", "C"],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Could this be related to order volume?")

    updated = service.run_investigation(
        investigation.investigation_id,
        df=df,
        data_context={
            "conversation_context": {
                "latest_chart_context": {
                    "metric": "Sales",
                    "dimension": "City",
                    "chart_type": "bar",
                    "title": "Average Sales by City",
                }
            }
        },
    )

    summary = updated.report.summary
    assert "Sales" in summary
    assert "City" in summary
    assert "record volume" in summary
    assert "Moscow" in summary
    assert "Ship Mode" not in summary
    assert updated.trace[0]["tool"] == "thread_count_relationship_check"
    assert any(artifact.artifact_type == ArtifactType.CHART for artifact in updated.artifacts)
    forbidden = ("using previous context", "continuing investigation state", "thread memory")
    assert not any(phrase in summary.lower() for phrase in forbidden)


def test_chart_request_overrides_successful_overview_without_chart() -> None:
    def overview_runner(**kwargs):
        return {
            "structured_report": {
                "summary": "The dataset has 4 rows and 3 columns. Useful quantitative fields include Sales.",
                "key_findings": ["The dataset has enough structure for analytical questions."],
            },
            "critic_verdict": "OK",
        }

    df = pd.DataFrame(
        {
            "City": ["Moscow", "Moscow", "Kazan", "Sochi"],
            "Sales": [1200.0, 800.0, 500.0, 700.0],
            "Order_Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    service = InvestigationService(runner=overview_runner)
    investigation = service.create_investigation("построй график топ городов по sales")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)
    assert chart.content["chart_type"] == "bar"
    assert chart.content["x"] == "City"
    assert chart.content["metric"] == "Sales"
    assert "Moscow" in updated.report.summary
    assert "enough structure" not in updated.report.summary.lower()


def test_follow_up_overrides_successful_generic_overview_when_dataframe_available() -> None:
    def overview_runner(**kwargs):
        return {
            "structured_report": {
                "summary": "The dataset has 4 rows and 3 columns. Useful quantitative fields include Sales.",
                "key_findings": ["The dataset has enough structure for analytical questions."],
            },
            "critic_verdict": "OK",
        }

    df = pd.DataFrame(
        {
            "City": ["Moscow", "Moscow", "Kazan", "Sochi"],
            "Sales": [1200.0, 800.0, 500.0, 700.0],
            "Order_Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    service = InvestigationService(runner=overview_runner)
    investigation = service.create_investigation("Which City groups have unusual Sales values?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    assert "Moscow" in updated.report.summary
    assert "Sales" in updated.report.summary
    assert "enough structure" not in updated.report.summary.lower()
    assert "useful quantitative fields" not in updated.report.summary.lower()
    assert any(artifact.artifact_type == ArtifactType.CHART for artifact in updated.artifacts)


def test_chart_intent_trend_uses_line_chart() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "group_label": ["A", "B", "A", "B"],
            "metric_value": [10.0, 20.0, 15.0, 25.0],
            "event_date": ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15"],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("build a chart of metric_value over time")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)
    assert chart.content["chart_type"] == "line"
    assert chart.content["timestamp"] == "event_date"
    assert updated.trace[0]["tool"] == "trend_check"


def test_chart_intent_distribution_uses_histogram() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "category_label": ["A", "B", "C", "D"],
            "metric_value": [10.0, 20.0, 30.0, 40.0],
            "event_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("build a histogram of metric_value distribution")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)
    assert chart.content["chart_type"] == "histogram"
    assert updated.trace[0]["tool"] == "deterministic_pandas_fallback"


def test_chart_intent_relationship_uses_scatter() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "category_label": ["A", "B", "C", "D", "E"],
            "Salary_LPA": [50.0, 60.0, 70.0, 90.0, 110.0],
            "Company_Rating": [2.0, 2.5, 3.0, 4.0, 5.0],
            "Date_Posted": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("build a chart showing the relationship between salary and rating")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)
    assert chart.content["chart_type"] == "scatter"
    assert updated.trace[0]["tool"] == "correlation_check"


def test_chart_intent_column_inference_is_dataset_agnostic() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "category_name": ["Alpha", "Alpha", "Beta", "Gamma"],
            "quality_score": [0.8, 0.9, 0.4, 0.7],
            "created_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("show top categories by score")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    chart = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.CHART)
    assert chart.content["chart_type"] == "bar"
    assert chart.content["x"] == "category_name"
    assert chart.content["metric"] == "quality_score"


def test_deterministic_fallback_explains_data_quality_implications() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "category_label": ["A", "A", "B", "B"],
            "metric_value": [10.0, None, 30.0, 30.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Is the data reliable?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    summary = updated.report.summary
    assert "missingness can bias segment comparisons" in summary
    assert "duplicates can inflate counts" in summary
    assert "distort averages" in summary
    assert updated.findings[0].metadata["analysis_type"] == "data_quality"


def test_deterministic_fallback_detects_outliers() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame({"entity_id": range(10), "metric_value": [10, 11, 12, 12, 13, 14, 15, 16, 17, 120]})
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Find outliers in metric_value")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    assert "outlier" in updated.report.summary.lower()
    assert "metric_value" in updated.report.summary
    assert "orchestration" not in updated.report.summary.lower()
