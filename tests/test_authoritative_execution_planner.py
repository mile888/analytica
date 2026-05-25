from __future__ import annotations

import pandas as pd

from source.product.execution_planner import (
    AuthoritativeExecutionPlanner,
    FilterConstraintValidator,
    TemporalSanityValidator,
)
from source.product.fallback_analysis import deterministic_investigation_fallback
from source.product.fallbacks.artifact_builders import build_histogram_artifact
from source.product.grounding_critic import dataframe_operation_precedence_check
from source.product.investigation import InvestigationMessage, InvestigationMessageRole
from source.product.run_service import InvestigationRunService
from source.product.service import InvestigationService
from source.product.store import InvestigationStore


def _retail_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sales": [100.0, 25.0, 300.0, 80.0, 50.0, 500.0],
            "City": ["Los Angeles", "Los Angeles", "San Francisco", "San Francisco", "Boston", "Los Angeles"],
            "Customer Name": ["A", "B", "C", "D", "E", "F"],
            "Order ID": ["o1", "o2", "o3", "o4", "o5", "o6"],
            "Product Name": ["Desk", "Chair", "Desk", "Lamp", "Chair", "Desk"],
            "Order Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-15", "2024-03-01", "2024-03-20"]),
            "Ship Date": pd.to_datetime(["2024-01-03", "2024-01-05", "2024-02-04", "2024-02-17", "2024-03-02", "2024-03-25"]),
        }
    )


def _generic_business_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Amount": [120.0, 80.0, 210.0, 60.0, 175.0, 95.0],
            "Location": ["East", "East", "West", "West", "North", "North"],
            "Region": ["East", "East", "West", "West", "North", "North"],
            "Client": ["Client A", "Client B", "Client C", "Client D", "Client E", "Client F"],
            "EventDate": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-02-01", "2024-02-05", "2024-03-01", "2024-03-03"]),
            "Category": ["Alpha", "Beta", "Alpha", "Gamma", "Beta", "Gamma"],
            "Channel": ["Standard", "Express", "Standard", "Express", "Standard", "Standard"],
            "DeliveryDate": pd.to_datetime(["2024-01-03", "2024-01-12", "2024-02-04", "2024-02-10", "2024-03-02", "2024-03-08"]),
            "Status": ["closed", "closed", "open", "closed", "open", "closed"],
        }
    )


def _generic_transaction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionValue": [10.0, 15.0, 40.0, 42.0, 120.0, 130.0, 18.0, 44.0],
            "Region": ["East", "East", "West", "West", "East", "West", "North", "North"],
            "Account": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "TransactionDate": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02", "2024-03-01", "2024-03-02", "2024-04-01", "2024-04-02"]),
            "ProductGroup": ["One", "Two", "One", "Two", "Three", "Three", "One", "Two"],
        }
    )


def _generic_operations_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": [4.0, 7.0, 9.0, 6.0, 12.0, 15.0],
            "Site": ["North", "North", "South", "South", "East", "East"],
            "Operator": ["A", "B", "A", "C", "B", "C"],
            "Timestamp": pd.to_datetime(["2024-01-01", "2024-01-12", "2024-02-01", "2024-02-12", "2024-03-01", "2024-03-12"]),
            "DelayDays": [0, 2, 1, 4, 0, 6],
            "Priority": ["low", "high", "low", "medium", "high", "medium"],
        }
    )


def _salary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Salary_LPA": [12.0, 15.0, 18.0, 10.0, 22.0, 16.0, 14.0, 30.0],
            "City": ["Bangalore", "Remote", "Hyderabad", "Remote", "Hyderabad", "Bangalore", "Remote", "Remote"],
            "Company": ["Apex", "Nira", "Orbit", "Zen", "Pulse", "Nova", "Luna", "Max"],
            "Date_Posted": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-02-01", "2024-02-15", "2024-03-01", "2024-03-10", "2024-04-01", "2024-04-15"]),
            "Region": ["South", "Remote", "South", "Remote", "South", "South", "Remote", "Remote"],
        }
    )


def _assert_executed_with_artifact(result: dict) -> None:
    assert result["trace_metadata"]["fallback"] in {"authoritative_query_plan", "direct_query_executor"}
    assert result.get("artifacts")
    assert "likely to vary" not in result["final_answer"]
    assert "analytical picture is unchanged" not in result["final_answer"].casefold()
    assert "strongest groups will" not in result["final_answer"].casefold()


def test_salary_grouped_ranking_commits_to_execution() -> None:
    result = deterministic_investigation_fallback("top cities by salary", _salary_df())
    plan = result["trace_metadata"]["query_plan"]

    _assert_executed_with_artifact(result)
    assert plan["intent"] == "rank_groups"
    assert plan["metric"] == "Salary_LPA"
    assert plan["dimension"] == "City"
    assert "`Salary_LPA` by `City` is led by" in result["final_answer"]
    assert result["artifacts"][1]["metadata"]["artifact_type"] == "chart"


def test_missing_requested_dimension_does_not_silently_substitute() -> None:
    df = _salary_df()
    result = deterministic_investigation_fallback("Top industries by salary", df)

    assert result["trace_metadata"]["query_plan"]["requires_user_confirmation"] is True
    assert result["trace_metadata"]["query_plan"]["requested_dimension_type"] == "industries"
    assert "could not find" in result["final_answer"].lower()
    assert "`industries`-like field" in result["final_answer"]
    assert "`Company`" in result["final_answer"]
    assert "by `Company` is led by" not in result["final_answer"]
    assert result.get("artifacts") == []


def test_salary_grouped_aggregation_commits_to_execution() -> None:
    result = deterministic_investigation_fallback("average salary by city", _salary_df())
    plan = result["trace_metadata"]["query_plan"]

    _assert_executed_with_artifact(result)
    assert plan["aggregation"] == "mean"
    assert plan["dimension"] == "City"
    assert "average `Salary_LPA`" in result["final_answer"]


def test_salary_extremum_preserves_remote_filter() -> None:
    result = deterministic_investigation_fallback("Which company in Remote has the highest salary?", _salary_df())
    plan = result["trace_metadata"]["query_plan"]

    _assert_executed_with_artifact(result)
    assert plan["intent"] == "extremum"
    assert plan["metric"] == "Salary_LPA"
    assert plan["dimension"] == "Company"
    assert plan["filters"][0]["value"] == "Remote"
    assert "`Company` is `Max`" in result["final_answer"]


def test_salary_distribution_and_temporal_queries_create_artifacts() -> None:
    histogram = deterministic_investigation_fallback("Build histogram of Salary_LPA in Remote", _salary_df())
    temporal = deterministic_investigation_fallback("salary trend over Date_Posted", _salary_df())

    _assert_executed_with_artifact(histogram)
    _assert_executed_with_artifact(temporal)
    assert histogram["artifacts"][0]["content"]["chart_type"] == "histogram"
    assert histogram["artifacts"][0]["content"]["row_count"] == 4
    assert temporal["trace_metadata"]["analysis_type"] == "temporal_trend"
    assert temporal["artifacts"][0]["content"]["chart_type"] == "line"


def test_service_replaces_generic_runner_narration_for_executable_query() -> None:
    def generic_runner(**_: object) -> dict:
        return {
            "summary": "`Salary_LPA` is likely to vary across `City`. The strongest groups will be the ones with enough rows.",
            "structured_report": {"summary": "`Salary_LPA` is likely to vary across `City`.", "key_findings": []},
            "artifacts": [],
        }

    store = InvestigationStore()
    service = InvestigationService(store=store, runner=generic_runner)
    investigation = service.create_investigation("top cities by salary")
    updated = service.run_investigation(investigation.investigation_id, df=_salary_df())

    assert "`Salary_LPA` by `City` is led by" in updated.report.summary
    assert "likely to vary" not in updated.report.summary
    assert len(updated.artifacts) >= 2


def test_run_service_commits_followups_to_execution_not_generic_continuity() -> None:
    def generic_runner(**kwargs: object) -> dict:
        question = str(kwargs.get("question") or "")
        if "top" in question.casefold():
            summary = "`Salary_LPA` is likely to vary across `City`. The strongest groups will be the ones with enough rows."
        else:
            summary = "The analytical picture is unchanged: Small samples and extreme values can make a group look stronger than it really is."
        return {"summary": summary, "structured_report": {"summary": summary, "key_findings": []}, "artifacts": []}

    store = InvestigationStore()
    service = InvestigationService(store=store, runner=generic_runner)
    run_service = InvestigationRunService(store, service)
    investigation = service.create_investigation("top cities by salary")

    run_service.run_investigation(investigation.investigation_id, df=_salary_df())
    message = store.add_investigation_message(InvestigationMessage(investigation_id=investigation.investigation_id, content="average salary by city"))
    run_service.run_investigation(investigation.investigation_id, df=_salary_df(), message_id=message.message_id)
    message = store.add_investigation_message(InvestigationMessage(investigation_id=investigation.investigation_id, content="Which company in Remote has the highest salary?"))
    run_service.run_investigation(investigation.investigation_id, df=_salary_df(), message_id=message.message_id)

    answers = [item.content for item in store.list_investigation_messages(investigation.investigation_id) if item.role == InvestigationMessageRole.ASSISTANT]
    assert "`Salary_LPA` by `City` is led by" in answers[-3]
    assert "average `Salary_LPA`" in answers[-2]
    assert "`Company` is `Max`" in answers[-1]
    assert all("analytical picture is unchanged" not in answer.casefold() for answer in answers)
    assert all("likely to vary" not in answer.casefold() for answer in answers)


def test_filter_preservation_lowest_customer_in_los_angeles() -> None:
    df = _retail_df()
    result = deterministic_investigation_fallback("Which customer in Los Angeles has the lowest Sales?", df)

    answer = result["final_answer"]
    assert "`City` = `Los Angeles`" in answer
    assert "`B`" in answer
    assert "not a global customer ranking" in answer


def test_safe_business_alias_revenue_by_city() -> None:
    df = _retail_df()
    result = deterministic_investigation_fallback("Посчитай выручку в каждом городе", df)
    plan = result["trace_metadata"]["query_plan"]

    assert plan["metric"] == "Sales"
    assert plan["dimension"] == "City"
    assert result["final_answer"].startswith("`выручку` is treated as `Sales`")


def test_geo_dimension_priority_prefers_city_over_country_for_city_intent() -> None:
    df = pd.DataFrame(
        {
            "Sales": [10.0, 20.0, 30.0, 40.0],
            "City": ["Northport", "Southport", "Northport", "Eastport"],
            "Country": ["Aland", "Aland", "Borduria", "Borduria"],
        }
    )
    result = deterministic_investigation_fallback("выручка по городам", df)
    plan = result["trace_metadata"]["query_plan"]

    assert plan["metric"] == "Sales"
    assert plan["dimension"] == "City"
    assert plan["dimension"] != "Country"
    assert "`Sales` by `City`" in result["final_answer"]


def test_customer_intent_prefers_human_readable_entity_over_identifier() -> None:
    df = pd.DataFrame(
        {
            "Sales": [30.0, 20.0, 10.0],
            "Customer ID": ["C-001", "C-002", "C-003"],
            "Customer Name": ["Aster Lane", "Briar Stone", "Cedar Vale"],
        }
    )
    result = deterministic_investigation_fallback("Какой customer имеет минимальный Sales", df)
    plan = result["trace_metadata"]["query_plan"]

    assert plan["dimension"] == "Customer Name"
    assert plan["target_entity"] == "Customer Name"
    assert "`Customer Name` is `Cedar Vale`" in result["final_answer"]
    assert "Customer ID" not in result["final_answer"]


def test_safe_business_alias_revenue_by_product_after_cleanup() -> None:
    result = deterministic_investigation_fallback("top products by revenue", _retail_df())
    plan = result["trace_metadata"]["query_plan"]

    assert plan["metric"] == "Sales"
    assert plan["dimension"] == "Product Name"
    assert plan["aggregation"] == "sum"
    assert "cannot" not in result["final_answer"].casefold()


def test_generic_business_fixture_aggregation_and_temporal_flow() -> None:
    df = _generic_business_df()
    aggregation = deterministic_investigation_fallback("top locations by amount", df)
    temporal = deterministic_investigation_fallback("average amount by eventdate", df)

    assert aggregation["trace_metadata"]["query_plan"]["metric"] == "Amount"
    assert aggregation["trace_metadata"]["query_plan"]["dimension"] == "Location"
    assert "`Amount` by `Location`" in aggregation["final_answer"]
    assert temporal["trace_metadata"]["analysis_type"] == "temporal_trend"
    assert temporal["trace_metadata"]["query_plan"]["time_axis"] == "EventDate"


def test_generic_location_ontology_resolves_location_dimension() -> None:
    result = deterministic_investigation_fallback("top locations by amount", _generic_business_df())
    plan = result["trace_metadata"]["query_plan"]

    _assert_executed_with_artifact(result)
    assert plan["metric"] == "Amount"
    assert plan["dimension"] == "Location"
    assert "`Amount` by `Location`" in result["final_answer"]


def test_location_ontology_prefers_city_for_generic_locations() -> None:
    df = pd.DataFrame(
        {
            "Amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "City": ["A", "B", "A", "C", "B"],
            "State": ["North", "North", "South", "South", "West"],
            "Region": ["East", "East", "East", "West", "West"],
            "Country": ["US", "US", "US", "US", "US"],
        }
    )

    result = deterministic_investigation_fallback("top locations by amount", df)

    assert result["trace_metadata"]["query_plan"]["dimension"] == "City"
    assert "`Amount` by `City`" in result["final_answer"]


def test_location_ontology_resolves_available_geo_levels() -> None:
    cases = [
        ("Region", pd.DataFrame({"Amount": [10.0, 20.0], "Region": ["East", "West"]})),
        ("State", pd.DataFrame({"Amount": [10.0, 20.0], "State": ["CA", "NY"]})),
        ("Country", pd.DataFrame({"Amount": [10.0, 20.0], "Country": ["US", "CA"]})),
    ]

    for expected, df in cases:
        result = deterministic_investigation_fallback("top locations by amount", df)
        assert result["trace_metadata"]["query_plan"]["dimension"] == expected
        assert f"`Amount` by `{expected}`" in result["final_answer"]


def test_generic_fixture_executable_queries_do_not_enter_exploration() -> None:
    df = _generic_business_df()
    ranking = deterministic_investigation_fallback("top locations by amount", df)
    aggregation = deterministic_investigation_fallback("average amount by region", df)
    distribution = deterministic_investigation_fallback("distribution of amount", df)

    for result in (ranking, aggregation, distribution):
        _assert_executed_with_artifact(result)
        assert result["trace_metadata"]["analysis_type"] != "exploration"
    assert aggregation["trace_metadata"]["query_plan"]["dimension"] == "Region"
    assert distribution["trace_metadata"]["analysis_type"] == "histogram"


def test_exploratory_mode_stays_isolated_from_executable_queries() -> None:
    exploratory = deterministic_investigation_fallback("What business questions can we investigate?", _generic_business_df())
    executable = deterministic_investigation_fallback("top products by revenue", _retail_df())

    assert exploratory["trace_metadata"]["analysis_type"] == "exploration"
    _assert_executed_with_artifact(executable)
    assert executable["trace_metadata"]["query_plan"]["intent"] == "rank_groups"
    assert executable["trace_metadata"]["query_plan"]["dimension"] == "Product Name"


def test_client_intent_prefers_readable_entity_over_code() -> None:
    df = pd.DataFrame(
        {
            "Revenue": [10.0, 90.0, 40.0],
            "Client Code": ["CL-001", "CL-002", "CL-003"],
            "Client Name": ["North Account", "West Account", "East Account"],
        }
    )
    result = deterministic_investigation_fallback("Which client has the highest revenue?", df)
    plan = result["trace_metadata"]["query_plan"]

    assert plan["dimension"] == "Client Name"
    assert "`Client Name` is `West Account`" in result["final_answer"]
    assert "Client Code" not in result["final_answer"]


def test_generic_operations_fixture_temporal_query_uses_timestamp() -> None:
    result = deterministic_investigation_fallback("metric trend over timestamp", _generic_operations_df())

    assert result["trace_metadata"]["analysis_type"] == "temporal_trend"
    assert result["trace_metadata"]["query_plan"]["metric"] == "Metric"
    assert result["trace_metadata"]["query_plan"]["time_axis"] == "Timestamp"


def test_generic_transaction_histogram_followup_preserves_distribution_context() -> None:
    df = _generic_transaction_df()
    first = deterministic_investigation_fallback("build histogram of transactionvalue in east", df)
    context = {
        "conversation_state": {"distribution_state": first["trace_metadata"]["distribution_state"]},
        "recent_artifacts": first["artifacts"],
    }
    result = deterministic_investigation_fallback("compare against west", df, data_context={"conversation_context": context})

    assert first["artifacts"][0]["content"]["chart_type"] == "histogram"
    assert first["artifacts"][0]["content"]["filters"][0]["value"] == "East"
    assert "East" in result["final_answer"]
    assert "West" in result["final_answer"]
    assert result["trace_metadata"]["analysis_type"] == "distribution_comparison"


def test_generic_amount_bins_are_created_without_retail_schema() -> None:
    result = deterministic_investigation_fallback("create bins automatically from amount", _generic_business_df())

    assert result["trace_metadata"]["analysis_type"] == "binning"
    assert result["trace_metadata"]["derived_field"]["source_metric"] == "Amount"
    assert "Amount_bin" in result["final_answer"]


def test_generic_amount_bin_followups_have_distinct_reasoning_modes() -> None:
    df = _generic_business_df()
    first = deterministic_investigation_fallback("create bins automatically from amount", df)
    context = {"conversation_context": {"conversation_state": {"derived_field": first["trace_metadata"]["derived_field"]}}}

    contribution = deterministic_investigation_fallback("Which bin contributes the most revenue?", df, data_context=context)
    driver = deterministic_investigation_fallback("Are high-value bins driven by volume or order value?", df, data_context=context)

    assert contribution["trace_metadata"]["analysis_type"] == "bins_contribution"
    assert driver["trace_metadata"]["analysis_type"] == "bins_volume_value_driver"
    assert contribution["final_answer"] != driver["final_answer"]
    assert "contributes the most" in contribution["final_answer"]
    assert "record volume" in driver["final_answer"]


def test_balanced_quantile_bins_are_not_labeled_sparse() -> None:
    df = _generic_business_df()
    first = deterministic_investigation_fallback("create bins automatically from amount", df)
    context = {"conversation_context": {"conversation_state": {"derived_field": first["trace_metadata"]["derived_field"]}}}

    result = deterministic_investigation_fallback("which bins are sparse?", df, data_context=context)

    assert result["trace_metadata"]["analysis_type"] == "bins_sparsity_balanced"
    assert "No `Amount_bin` bins are materially sparse" in result["final_answer"]
    assert "quantile binning produced balanced groups" in result["final_answer"]


def test_specific_relationship_does_not_substitute_strongest_numeric_field() -> None:
    df = pd.DataFrame(
        {
            "Amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Rating": [5.0, 4.0, 3.0, 2.0, 1.0],
            "OtherMetric": [100.0, 90.0, 70.0, 45.0, 20.0],
        }
    )
    result = deterministic_investigation_fallback("Is amount related to rating?", df)

    assert result["trace_metadata"]["analysis_type"] == "specific_relationship"
    assert result["trace_metadata"]["metric"] == "Amount"
    assert result["trace_metadata"]["comparison"] == "Rating"
    assert "`Amount` and `Rating`" in result["final_answer"]
    assert "OtherMetric" not in result["final_answer"]


def test_growth_relationship_requires_time_target_without_substitution() -> None:
    df = pd.DataFrame(
        {
            "Salary_LPA": [50.0, 60.0, 70.0, 90.0, 110.0],
            "Company_Rating": [2.0, 2.5, 3.0, 4.0, 5.0],
            "Applicants": [100, 90, 70, 45, 20],
        }
    )
    result = deterministic_investigation_fallback("Is salary growth related to company rating?", df)

    assert result["trace_metadata"]["analysis_type"] == "relationship_target_unavailable"
    assert "not `Salary_LPA` growth" in result["final_answer"]
    assert "no reliable time field" in result["final_answer"]
    assert "Applicants has the strongest" not in result["final_answer"]


def test_histogram_artifact_rejects_row_count_mismatch() -> None:
    try:
        build_histogram_artifact(
            metric="Metric",
            bins=[{"left": 0, "right": 1, "label": "0 to 1", "count": 2}],
            filters=[],
            title="Metric distribution",
            row_count=3,
        )
    except ValueError as exc:
        assert "reconcile" in str(exc)
    else:
        raise AssertionError("Expected histogram row-count mismatch to fail validation")


def test_schema_profile_question_bypasses_grouped_ranking() -> None:
    result = deterministic_investigation_fallback("What fields are most important here?", _generic_business_df())

    assert result["trace_metadata"]["analysis_type"] == "exploration"
    assert result["trace_metadata"]["fallback"] == "important_fields"
    assert "The most important fields" in result["final_answer"]
    assert " is led by " not in result["final_answer"]


def test_generic_delivery_delay_fixture_uses_delay_semantics() -> None:
    result = deterministic_investigation_fallback("late events", _generic_business_df())

    assert result["trace_metadata"]["query_plan"]["intent"] == "shipping_delay"
    assert result["trace_metadata"]["analysis_type"] == "shipping_delay"
    assert "delivery_delay_days" in result["final_answer"]
    assert "Average delay" in result["final_answer"]


def test_delay_analysis_warns_instead_of_refusing_ambiguous_dates() -> None:
    df = pd.DataFrame(
        {
            "Amount": [10.0, 20.0, 30.0],
            "EventDate": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "DeliveryDate": pd.to_datetime(["2024-01-02", "2024-04-15", "2024-01-05"]),
        }
    )
    result = deterministic_investigation_fallback("late events", df)

    assert result["trace_metadata"]["analysis_type"] == "shipping_delay"
    assert "Average delay" in result["final_answer"]
    assert "Interpret" in result["final_answer"] or result["limitations"]


def test_generic_delivery_delays_query_uses_delay_semantics() -> None:
    result = deterministic_investigation_fallback("delivery delays", _generic_business_df())

    assert result["trace_metadata"]["query_plan"]["intent"] == "shipping_delay"
    assert result["trace_metadata"]["analysis_type"] == "shipping_delay"
    assert result["artifacts"][0]["title"] == "Delivery delay distribution"


def test_generic_artifact_explanation_uses_selected_chart_metadata() -> None:
    chart = {
        "artifact_id": "generic_chart",
        "artifact_type": "chart",
        "title": "Amount by Location",
        "content": {
            "chart_type": "bar",
            "metric": "Amount",
            "dimension": "Location",
            "aggregation": "sum",
            "x": "Location",
            "y": "sum",
            "rows": [{"Location": "East", "sum": 200.0}, {"Location": "West", "sum": 270.0}],
        },
        "metadata": {"metric": "Amount", "dimension": "Location", "aggregation": "sum", "branch_id": "grouped::Amount::Location::sum::::"},
    }
    result = deterministic_investigation_fallback(
        "Explain this chart.",
        _generic_business_df(),
        data_context={
            "conversation_context": {
                "active_message_metadata": {"action": "explain_artifact", "artifact_id": "generic_chart"},
                "recent_artifacts": [chart],
                "conversation_state": {},
            }
        },
    )

    assert result["trace_metadata"]["analysis_type"] == "chart_explanation"
    assert "total `Amount`" in result["final_answer"]
    assert "`Location`" in result["final_answer"]


def test_location_abbreviation_only_applies_when_observed_value_exists() -> None:
    df = pd.DataFrame({"Sales": [10.0, 20.0], "City": ["Lagos", "Lisbon"], "Order Date": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    result = deterministic_investigation_fallback("Build histogram of Sales in LA", df)

    assert result["trace_metadata"]["query_plan"]["filters"] == []
    # LA should NOT be silently expanded to Los Angeles.
    # The system should detect that "LA" is an unresolved filter and report it,
    # NOT silently build an unfiltered histogram.
    assert result["trace_metadata"]["analysis_type"] == "filter_not_found"
    assert result.get("artifacts") == []
    assert "la" in result["final_answer"].lower()


def test_histogram_filter_resolves_validated_location_abbreviation() -> None:
    df = _retail_df()
    result = deterministic_investigation_fallback("Build histogram of Sales in LA", df)
    chart = result["artifacts"][0]

    assert result["trace_metadata"]["analysis_type"] == "histogram"
    assert result["trace_metadata"]["query_plan"]["filters"][0]["column"] == "City"
    assert result["trace_metadata"]["query_plan"]["filters"][0]["value"] == "Los Angeles"
    assert chart["content"]["row_count"] == 3
    assert chart["content"]["filters"] == [{"column": "City", "operator": "equals", "value": "Los Angeles", "source": "observed_abbreviation"}]
    assert "Filter: `City` = `Los Angeles`" in result["final_answer"]


def test_histogram_filter_resolves_generic_region_value() -> None:
    df = _generic_business_df()
    result = deterministic_investigation_fallback("Build histogram of Amount in East", df)
    chart = result["artifacts"][0]

    assert result["trace_metadata"]["analysis_type"] == "histogram"
    assert result["trace_metadata"]["query_plan"]["filters"][0]["column"] in {"Location", "Region"}
    assert result["trace_metadata"]["query_plan"]["filters"][0]["value"] == "East"
    assert chart["content"]["row_count"] == 2
    assert chart["content"]["filters"]


def test_no_structural_substitution_for_bin() -> None:
    df = _retail_df()
    result = deterministic_investigation_fallback("Does record volume explain differences across bin?", df)

    assert "There is no saved `bin` field" in result["final_answer"]


def test_histogram_preserves_alias_and_filter() -> None:
    df = _retail_df()
    result = deterministic_investigation_fallback("Build a histogram of revenue in Los Angeles", df)
    chart = result["artifacts"][0]

    assert "`Sales` distribution" in result["final_answer"]
    assert "`City` = `Los Angeles`" in result["final_answer"]
    assert chart["content"]["chart_type"] == "histogram"
    assert chart["content"]["filters"][0]["value"] == "Los Angeles"


def test_histogram_comparison_continuation_resolves_abbreviation_to_same_dimension() -> None:
    df = _retail_df()
    first = deterministic_investigation_fallback("Build histogram of Sales in LA", df)
    context = {
        "latest_chart_context": {"chart_type": "histogram", "metric": "Sales"},
        "recent_artifacts": first["artifacts"],
        "conversation_state": {"distribution_state": first["trace_metadata"]["distribution_state"]},
    }

    result = deterministic_investigation_fallback("Compare against SF", df, data_context={"conversation_context": context})
    chart = result["artifacts"][0]

    assert result["trace_metadata"]["analysis_type"] == "distribution_comparison"
    assert chart["content"]["chart_type"] == "histogram"
    assert chart["content"]["comparison_groups"][0]["group"] == "Los Angeles"
    assert chart["content"]["comparison_groups"][1]["group"] == "San Francisco"
    assert "Segment" not in result["final_answer"]
    assert "Sub-Category" not in result["final_answer"]


def test_seasonality_heatmap_does_not_fallback_to_bar() -> None:
    df = _retail_df()
    result = deterministic_investigation_fallback("Seasonality heatmap", df)
    chart = result["artifacts"][0]

    assert "Seasonality heatmap plan preserved" in result["final_answer"]
    assert chart["content"]["chart_type"] == "heatmap"


def test_bin_creation_derives_sales_bin() -> None:
    df = _retail_df()
    result = deterministic_investigation_fallback("Create bins automatically from Sales", df)

    assert "Sales_bin" in result["final_answer"]
    assert "ready for contribution" in result["final_answer"]


def test_plan_validator_preserves_filter() -> None:
    df = _retail_df()
    plan = AuthoritativeExecutionPlanner.plan("Which customer in Los Angeles has the lowest Sales?", df)

    assert plan.filters[0].column == "City"
    assert plan.filters[0].value == "Los Angeles"
    assert FilterConstraintValidator.validate(plan, df) == []


def test_delivery_delay_sanity_flags_impossible_delay() -> None:
    df = pd.DataFrame(
        {
            "Order Date": pd.to_datetime(["2024-01-01", "2024-01-05"]),
            "Ship Date": pd.to_datetime(["2023-12-31", "2025-04-01"]),
        }
    )

    issues = TemporalSanityValidator.delivery_delay_issues(df, "Order Date", "Ship Date")
    assert any("negative" in issue for issue in issues)
    assert any("year" in issue for issue in issues)


def test_dataframe_operation_dtype_request_does_not_group_rank() -> None:
    df = pd.DataFrame({"Quantity": ["1", "2", "bad"], "UnitPrice": [10.0, 20.0, 30.0]})

    result = deterministic_investigation_fallback("Check whether Quantity is numeric", df)

    assert result["trace_metadata"]["analysis_type"] == "dataframe_operation"
    assert result["trace_metadata"]["operation"] == "dtype_inspection"
    assert result["trace_metadata"]["fallback"] != "authoritative_query_plan"
    assert result["artifacts"][0]["content"][0]["column"] == "Quantity"
    assert result["artifacts"][0]["content"][0]["conversion_failures"] == 1


def test_explicit_group_metric_ranking_beats_dtype_inspection() -> None:
    df = pd.DataFrame(
        {
            "City": ["New York", "Boston", "New York", "Seattle", "Austin", "Boston", "Miami", "Denver"],
            "Sales": [100.0, 200.0, 150.0, 90.0, 300.0, 50.0, 120.0, 80.0],
            "Row ID": list(range(8)),
            "Postal Code": ["10001", "02101", "10002", "98101", "73301", "02102", "33101", "80201"],
        }
    )

    result = deterministic_investigation_fallback(
        "Show the top 10 real cities by total Sales. Use City as the grouping column and Sales as the numeric metric.",
        df,
    )
    plan = result["trace_metadata"]["query_plan"]

    assert result["trace_metadata"]["analysis_type"] == "rank_groups"
    assert result["trace_metadata"].get("operation") != "dtype_inspection"
    assert plan["intent"] == "rank_groups"
    assert plan["dimension"] == "City"
    assert plan["metric"] == "Sales"
    assert plan["aggregation"] == "sum"
    assert plan["limit"] == 10
    assert result["artifacts"][0]["artifact_type"] == "table"
    assert result["artifacts"][1]["artifact_type"] == "chart"
    assert result["artifacts"][0]["content"][0]["City"] == "Austin"
    assert result["artifacts"][0]["content"][0]["total"] == 300.0


def test_explicit_category_profit_ranking_beats_dtype_inspection() -> None:
    df = pd.DataFrame(
        {
            "Product Category": ["Furniture", "Technology", "Furniture", "Office", "Technology", "Office"],
            "Profit": [20.0, 150.0, 30.0, 40.0, 80.0, 35.0],
            "Product ID": ["P-1", "P-2", "P-3", "P-4", "P-5", "P-6"],
        }
    )

    result = deterministic_investigation_fallback(
        "Use Product Category as the grouping column and Profit as the numeric metric. Show top 5 by total Profit.",
        df,
    )
    plan = result["trace_metadata"]["query_plan"]

    assert result["trace_metadata"]["analysis_type"] == "rank_groups"
    assert result["trace_metadata"].get("operation") != "dtype_inspection"
    assert plan["dimension"] == "Product Category"
    assert plan["metric"] == "Profit"
    assert plan["aggregation"] == "sum"
    assert plan["limit"] == 5
    assert result["artifacts"][0]["content"][0]["Product Category"] == "Technology"
    assert result["artifacts"][0]["content"][0]["total"] == 230.0


def test_explicit_dtype_inspection_still_works_with_mentioned_columns() -> None:
    df = pd.DataFrame({"City": ["New York", "Boston"], "Sales": [100.0, 200.0], "Profit": [20.0, 30.0]})

    result = deterministic_investigation_fallback("Inspect dtypes for City and Sales.", df)

    assert result["trace_metadata"]["analysis_type"] == "dataframe_operation"
    assert result["trace_metadata"]["operation"] == "dtype_inspection"
    assert [row["column"] for row in result["artifacts"][0]["content"]] == ["City", "Sales"]


def test_numeric_convertibility_inspection_still_works() -> None:
    df = pd.DataFrame({"Quantity": ["1", "2", "bad"], "Revenue": ["10.5", "20.0", "30.1"]})

    result = deterministic_investigation_fallback("Which columns can be converted to numeric?", df)

    assert result["trace_metadata"]["analysis_type"] == "dataframe_operation"
    assert result["trace_metadata"]["operation"] == "dtype_inspection"
    assert {row["column"] for row in result["artifacts"][0]["content"]} == {"Quantity", "Revenue"}


def test_average_duration_chart_request_beats_dtype_inspection() -> None:
    df = pd.DataFrame(
        {
            "genre": ["Drama", "Drama", "Comedy", "Comedy", "Action"],
            "duration": ["90 min", "110 min", "80 min", "100 min", "120 min"],
        }
    )

    result = deterministic_investigation_fallback("Build a chart of average duration by genre.", df)

    assert result["trace_metadata"]["analysis_type"] != "dataframe_operation"
    assert result["trace_metadata"].get("operation") != "dtype_inspection"
    assert any(artifact["artifact_type"] == "chart" for artifact in result["artifacts"])


def test_critic_rejects_dtype_answer_for_analytical_ranking_request() -> None:
    refusal = dataframe_operation_precedence_check(
        "Show the top 10 real cities by total Sales. Use City as the grouping column and Sales as the numeric metric.",
        "Dataframe dtype inspection: checked 2 columns.",
    )

    assert refusal is not None
    assert "analytical planner" in refusal
    assert dataframe_operation_precedence_check(
        "Inspect dtypes for City and Sales.",
        "Dataframe dtype inspection: checked 2 columns.",
    ) is None


def _followup_context_from_result(result: dict) -> dict:
    chart = next(artifact for artifact in result["artifacts"] if artifact.get("artifact_type") == "chart")
    chart_context = {
        "metric": chart.get("metadata", {}).get("metric") or chart.get("content", {}).get("metric"),
        "dimension": chart.get("metadata", {}).get("dimension") or chart.get("content", {}).get("dimension"),
        "aggregation": chart.get("metadata", {}).get("aggregation") or chart.get("content", {}).get("aggregation"),
        "chart_type": chart.get("metadata", {}).get("chart_type") or chart.get("content", {}).get("chart_type"),
        "top_n": chart.get("metadata", {}).get("top_n"),
        "filters": chart.get("metadata", {}).get("filters") or chart.get("content", {}).get("filters") or [],
    }
    return {
        "conversation_context": {
            "conversation_state": {
                "active_metric": chart_context["metric"],
                "active_dimension": chart_context["dimension"],
                "active_aggregation": chart_context["aggregation"],
                "active_chart_type": chart_context["chart_type"],
                "active_filters": chart_context["filters"],
            },
            "latest_chart_context": chart_context,
            "recent_artifacts": result["artifacts"],
        }
    }


def test_followup_correction_patches_average_sales_to_total_sales() -> None:
    df = pd.DataFrame(
        {
            "City": ["Austin", "Austin", "Boston", "Boston", "Sale"],
            "Sales": [100.0, 50.0, 80.0, 90.0, 200.0],
        }
    )
    first = deterministic_investigation_fallback("Which cities are performing best overall?", df)

    result = deterministic_investigation_fallback(
        "don't use average sales, use total sales",
        df,
        data_context=_followup_context_from_result(first),
    )
    plan = result["trace_metadata"]["query_plan"]

    assert result["trace_metadata"]["fallback"] == "followup_plan_patch"
    assert plan["metric"] == "Sales"
    assert plan["dimension"] == "City"
    assert plan["aggregation"] == "sum"
    assert "using total `Sales`" in result["final_answer"]


def test_followup_correction_patches_product_profit_average_to_total() -> None:
    df = pd.DataFrame(
        {
            "Product": ["Desk", "Desk", "Chair", "Chair", "Lamp"],
            "Profit": [20.0, 30.0, 35.0, 25.0, 40.0],
        }
    )
    first = deterministic_investigation_fallback("Top products by average Profit", df)

    result = deterministic_investigation_fallback(
        "use total Profit instead",
        df,
        data_context=_followup_context_from_result(first),
    )
    plan = result["trace_metadata"]["query_plan"]

    assert plan["metric"] == "Profit"
    assert plan["dimension"] == "Product"
    assert plan["aggregation"] == "sum"
    assert result["artifacts"][0]["content"][0]["Product"] == "Chair"
    assert result["artifacts"][0]["content"][0]["total"] == 60.0


def test_followup_correction_does_not_create_metric_word_filter() -> None:
    df = pd.DataFrame(
        {
            "City": ["Sale", "Sale", "Boston", "Austin"],
            "Sales": [10.0, 20.0, 100.0, 80.0],
        }
    )
    first = deterministic_investigation_fallback("Which cities are performing best overall?", df)

    result = deterministic_investigation_fallback(
        "don't use average sales, use total sales",
        df,
        data_context=_followup_context_from_result(first),
    )

    assert result["trace_metadata"]["query_plan"]["filters"] == []
    assert "City` = `Sale" not in result["final_answer"]


def test_followup_correction_chart_title_changes_to_total() -> None:
    df = pd.DataFrame({"City": ["A", "A", "B"], "Sales": [1.0, 2.0, 4.0]})
    first = deterministic_investigation_fallback("Which cities are performing best overall?", df)

    result = deterministic_investigation_fallback(
        "don't use average sales, use total sales",
        df,
        data_context=_followup_context_from_result(first),
    )

    chart = next(artifact for artifact in result["artifacts"] if artifact["artifact_type"] == "chart")
    assert chart["title"] == "Total Sales by City"
    assert chart["metadata"]["aggregation"] == "sum"


def test_followup_correction_does_not_trigger_dtype_inspection() -> None:
    df = pd.DataFrame({"City": ["A", "B"], "Sales": [1.0, 2.0]})
    first = deterministic_investigation_fallback("Which cities are performing best overall?", df)

    result = deterministic_investigation_fallback(
        "don't use average sales, use total sales",
        df,
        data_context=_followup_context_from_result(first),
    )

    assert result["trace_metadata"]["analysis_type"] == "rank_groups"
    assert result["trace_metadata"].get("operation") == "FOLLOWUP_PLAN_PATCH"
    assert result["trace_metadata"].get("operation") != "dtype_inspection"


def test_run_service_followup_correction_uses_active_chart_plan() -> None:
    def error_runner(**_: object) -> dict:
        return {"exec_error": "force deterministic fallback"}

    df = pd.DataFrame(
        {
            "City": ["Sale", "Sale", "Boston", "Austin"],
            "Sales": [10.0, 20.0, 100.0, 80.0],
        }
    )
    store = InvestigationStore()
    investigation_service = InvestigationService(store=store, runner=error_runner)
    service = InvestigationRunService(store, investigation_service)
    investigation = investigation_service.create_investigation("Which cities are performing best overall?")

    service.run_investigation(investigation.investigation_id, df=df)
    message = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="don't use average sales, use total sales")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)
    updated = store.get_investigation(investigation.investigation_id)
    answers = [item.content for item in store.list_investigation_messages(investigation.investigation_id) if item.role == InvestigationMessageRole.ASSISTANT]
    chart_titles = [artifact.title for artifact in updated.artifacts if artifact.artifact_type.value == "chart"]

    assert "using total `Sales`" in answers[-1]
    assert "City` = `Sale" not in answers[-1]
    assert "Total Sales by City" in chart_titles


def test_dataframe_operation_formula_validation_does_not_group_rank() -> None:
    df = pd.DataFrame({"Subtotal": [100.0, 200.0], "Tax": [5.0, 11.0]})

    result = deterministic_investigation_fallback("Validate whether Tax equals 5% of Subtotal", df)

    assert result["trace_metadata"]["analysis_type"] == "dataframe_operation"
    assert result["trace_metadata"]["operation"] == "formula_validation"
    assert "Mismatches: 1" in result["final_answer"]
    assert " by " not in result["trace_metadata"].get("operation", "")


def test_derived_column_lineage_can_feed_followup_metric() -> None:
    df = pd.DataFrame({"Price": [10.0, 4.0, 5.0], "Quantity": [2, 10, 3], "Location": ["A", "B", "A"]})
    first = deterministic_investigation_fallback("Create Revenue = Price * Quantity", df)
    context = {"conversation_context": {"conversation_state": {"derived_columns": first["trace_metadata"]["derived_columns"]}}}

    second = deterministic_investigation_fallback("Find highest revenue groups by Location", df, data_context=context)

    assert first["trace_metadata"]["operation"] == "derived_column"
    assert second["trace_metadata"]["query_plan"]["metric"] == "Revenue"
    assert "Revenue" in second["final_answer"]


def test_derived_column_lineage_persists_through_run_service_followup() -> None:
    df = pd.DataFrame({"Price": [10.0, 4.0, 5.0], "Quantity": [2, 10, 3], "Location": ["A", "B", "A"]})
    def generic_runner(**_: object) -> dict:
        return {"summary": "Generic dataframe narration.", "structured_report": {"summary": "Generic dataframe narration.", "key_findings": []}, "artifacts": []}

    store = InvestigationStore()
    investigation_service = InvestigationService(store=store, runner=generic_runner)
    service = InvestigationRunService(store, investigation_service)
    investigation = investigation_service.create_investigation("Create Revenue = Price * Quantity")

    service.run_investigation(investigation.investigation_id, df=df)
    message = store.add_investigation_message(
        InvestigationMessage(investigation_id=investigation.investigation_id, content="Find highest revenue groups by Location")
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)
    updated = store.get_investigation(investigation.investigation_id)
    answers = [item.content for item in store.list_investigation_messages(investigation.investigation_id) if item.role == InvestigationMessageRole.ASSISTANT]

    assert updated.metadata["conversation_state"]["derived_columns"][0]["column"] == "Revenue"
    assert "Revenue" in answers[-1]
    assert "`B`" in answers[-1]


def test_russian_ingestion_diagnostic_detects_glued_columns_before_metric_fallback() -> None:
    df = pd.DataFrame({"Income;Age;Region": ["100;35;North", "200;44;South"]})

    result = deterministic_investigation_fallback(
        "Проверь, правильно ли загружен датасет. Если все данные оказались в одной колонке или колонки выглядят склеенными, определи правильный разделитель, перезагрузи данные и покажи первые 5 строк.",
        df,
    )

    assert result["trace_metadata"]["analysis_type"] == "csv_ingestion_diagnostic"
    assert result["trace_metadata"]["delimiter"] == ";"
    assert "usable numeric metric" not in result["final_answer"]
    assert "Income" in result["result_preview"]
    assert result["artifacts"][0]["content"][0]["Income"] == 100


def test_russian_missing_values_request_uses_quality_branch_not_active_metric_fallback() -> None:
    df = pd.DataFrame({"Income": [100.0, None, 300.0], "Region": ["A", "B", None]})

    result = deterministic_investigation_fallback(
        "Найди все пропуски в датасете. Особенно проверь колонку Income. Предложи безопасный способ обработки пропусков и объясни, почему он подходит.",
        df,
        data_context={
            "conversation_context": {
                "conversation_state": {
                    "active_metric": "Revenue",
                    "active_dimension": "City",
                    "active_branch_type": "grouped_comparison",
                }
            }
        },
    )

    assert result["trace_metadata"]["analysis_type"] == "data_quality"
    assert "usable numeric metric" not in result["final_answer"]
    assert "active metric" not in result["final_answer"]
    assert any(row["column"] == "Income" and row["missing_values"] == 1 for row in result["artifacts"][0]["content"])


# ===========================================================================
# HARD EXECUTION STOP — Regression tests
# ===========================================================================


def test_missing_metric_and_filter_value_hard_stop_no_chart() -> None:
    """Dataset without Sales/LA must refuse, produce no chart, no Z_Revenue."""
    df = pd.DataFrame(
        {
            "Z_Revenue": [100.0, 200.0, 300.0],
            "Education": ["High School", "College", "Graduate"],
            "ID": [1, 2, 3],
        }
    )
    result = deterministic_investigation_fallback("Build histogram of Sales in LA", df)

    assert result["trace_metadata"]["analysis_type"] == "hard_stop"
    assert result["trace_metadata"]["fallback"] == "hard_stop_missing_fields"
    assert result.get("artifacts") == []
    # The refusal may list Z_Revenue as available, but must NOT produce a Z_Revenue distribution chart
    assert "distribution" not in result["final_answer"].lower() or "cannot" in result["final_answer"].lower()
    assert "cannot" in result["final_answer"].lower() or "does not contain" in result["final_answer"].lower()
    assert "sales" in result["final_answer"].lower()


def test_stale_active_metric_does_not_override_missing_fields() -> None:
    """Even with Z_Revenue in active state, missing Sales/LA must still refuse."""
    df = pd.DataFrame(
        {
            "Z_Revenue": [100.0, 200.0, 300.0],
            "Education": ["High School", "College", "Graduate"],
        }
    )
    result = deterministic_investigation_fallback(
        "Build histogram of Sales in LA",
        df,
        data_context={
            "conversation_context": {
                "conversation_state": {
                    "active_metric": "Z_Revenue",
                    "active_dimension": "Education",
                    "active_branch_type": "grouped_comparison",
                }
            }
        },
    )

    assert result["trace_metadata"]["analysis_type"] == "hard_stop"
    assert result.get("artifacts") == []
    # Must NOT produce a Z_Revenue grouped chart or distribution
    assert "Z_Revenue` by `Education` is led by" not in result["final_answer"]


def test_joinability_query_does_not_execute_grouped_analysis() -> None:
    """Multi-dataset joinability query must produce joinability reasoning, never grouped analysis."""
    df = pd.DataFrame(
        {
            "Z_Revenue": [100.0, 200.0, 300.0],
            "Education": ["High School", "College", "Graduate"],
        }
    )
    result = deterministic_investigation_fallback("Can these datasets be joined reliably?", df)

    assert result["trace_metadata"]["fallback"] == "insufficient_dataset_scope"
    assert result["trace_metadata"]["analysis_type"] == "insufficient_dataset_scope"
    assert " is led by " not in result["final_answer"]
    assert "by `Education`" not in result["final_answer"]


def test_warehouse_design_does_not_execute_grouped_analysis() -> None:
    """Warehouse design query must produce entity/fact/dimension reasoning, not grouped analysis."""
    df = pd.DataFrame(
        {
            "Amount": [10.0, 20.0, 30.0],
            "Category": ["A", "B", "C"],
            "Region": ["East", "West", "North"],
        }
    )
    result = deterministic_investigation_fallback("Design a unified data warehouse from this dataset", df)

    assert result["trace_metadata"]["fallback"] == "insufficient_dataset_scope"
    assert result["trace_metadata"]["analysis_type"] == "insufficient_dataset_scope"
    assert " is led by " not in result["final_answer"]


def test_explain_chart_without_chart_returns_safe_refusal() -> None:
    """Explain chart with no chart available must not fall through to grouped analysis."""
    from source.product.conversation_engine import _chart_explanation_response

    response = _chart_explanation_response(
        "Explain this chart",
        [],
        {},
        {},
    )

    assert response is not None
    assert "do not currently have a chart" in response.text
    assert response.response_kind == "chart_explanation"


def test_russian_missing_metric_hard_stop_contains_available_fields() -> None:
    """Russian query on dataset without requested metric must refuse and list available fields."""
    df = pd.DataFrame(
        {
            "Metric_A": [10.0, 20.0, 30.0],
            "Metric_B": [40.0, 50.0, 60.0],
            "Category": ["X", "Y", "Z"],
        }
    )
    result = deterministic_investigation_fallback("Построй гистограмму Sales по городам", df)

    assert result["trace_metadata"]["analysis_type"] in {"hard_stop", "fallback"}
    assert result.get("artifacts") == []
    assert " is led by " not in result["final_answer"]
    # Must not silently substitute with Metric_A or Metric_B
    assert "Metric_A` by `Category` is led by" not in result["final_answer"]


# ─── Multi-dataset prerequisite validation tests ────────────────────────────────


def test_joinability_single_dataset_returns_prerequisite_failure() -> None:
    """Single dataset loaded: joinability query must refuse, not run grouped analysis."""
    df = pd.DataFrame(
        {
            "Sales": [100.0, 200.0, 300.0],
            "City": ["NYC", "LA", "Chicago"],
        }
    )
    result = deterministic_investigation_fallback("Can these datasets be joined reliably?", df)

    assert result["trace_metadata"]["fallback"] == "insufficient_dataset_scope"
    assert result["trace_metadata"]["analysis_type"] == "insufficient_dataset_scope"
    assert result["trace_metadata"]["loaded_dataset_count"] == 1
    assert "only 1 dataset" in result["final_answer"].lower()
    assert "join" in result["final_answer"].lower()
    assert result.get("artifacts") == []
    # Must NOT contain grouped analysis
    assert "`Sales` by `City` is led by" not in result["final_answer"]
    assert "Sales by City" not in result["final_answer"]


def test_warehouse_design_single_dataset_returns_prerequisite_failure() -> None:
    """Single dataset loaded: warehouse design must say multiple datasets required."""
    df = pd.DataFrame(
        {
            "Revenue": [10.0, 20.0, 30.0],
            "Region": ["East", "West", "East"],
        }
    )
    result = deterministic_investigation_fallback("Design a warehouse across these datasets", df)

    assert result["trace_metadata"]["fallback"] == "insufficient_dataset_scope"
    assert "multiple datasets" in result["final_answer"].lower() or "warehouse" in result["final_answer"].lower()
    assert result.get("artifacts") == []


def test_compare_both_datasets_single_dataset_returns_prerequisite_failure() -> None:
    """Single dataset loaded: comparing both datasets must ask for another dataset."""
    df = pd.DataFrame(
        {
            "Sales": [100.0, 200.0],
            "Category": ["A", "B"],
        }
    )
    result = deterministic_investigation_fallback("Compare both datasets", df)

    assert result["trace_metadata"]["fallback"] == "insufficient_dataset_scope"
    assert "1 dataset" in result["final_answer"] or "only" in result["final_answer"].lower()
    assert result.get("artifacts") == []
    # Must NOT fall through to grouped analysis
    assert "`Sales` by `Category`" not in result["final_answer"]


def test_joinability_two_datasets_does_not_trigger_prerequisite_failure() -> None:
    """Two datasets loaded: joinability query should proceed to real analysis."""
    df = pd.DataFrame(
        {
            "Sales": [100.0, 200.0, 300.0],
            "City": ["NYC", "LA", "Chicago"],
        }
    )
    result = deterministic_investigation_fallback(
        "Can these datasets be joined reliably?",
        df,
        data_context={"data_source_ids": ["ds_1", "ds_2"]},
    )

    # Should NOT be a prerequisite failure
    assert result["trace_metadata"]["fallback"] != "insufficient_dataset_scope"
    # Should go through to the global intent bypass (joinability reasoning)
    assert result["trace_metadata"]["fallback"] in {"global_intent_bypass", "authoritative_query_plan", "direct_query_executor"}


def test_no_grouped_fallback_after_prerequisite_failure() -> None:
    """After prerequisite failure, no grouped metric analysis should be produced."""
    df = pd.DataFrame(
        {
            "Sales": [100.0, 200.0, 300.0, 400.0, 500.0],
            "City": ["NYC", "LA", "Chicago", "Houston", "Phoenix"],
            "Segment": ["A", "B", "A", "C", "B"],
        }
    )
    queries = [
        "Can these datasets be joined reliably?",
        "What shared entities exist between datasets?",
        "Design a warehouse across these datasets",
        "Compare both datasets",
    ]
    for query in queries:
        result = deterministic_investigation_fallback(query, df)
        assert " is led by " not in result["final_answer"], f"Grouped fallback leaked for: {query}"
        assert "distribution" not in result["final_answer"].lower() or "dataset" in result["final_answer"].lower(), f"Distribution leaked for: {query}"
        assert result["trace_metadata"]["analysis_type"] != "grouped_comparison", f"Grouped comparison type for: {query}"


# ─── Product polish regression tests ────────────────────────────────────────────


def test_continuation_after_refusal_returns_clarification() -> None:
    """After a hard-stop refusal, a continuation question must clarify, not fall into stale analysis."""
    df = pd.DataFrame(
        {
            "Salary_LPA": [10.0, 22.0, 14.0, 18.0],
            "City": ["Bengaluru", "Delhi", "Mumbai", "Bengaluru"],
        }
    )
    # Simulate prior refusal state in conversation_context
    conversation_context = {
        "conversation_state": {
            "analysis_type": "hard_stop_missing_fields",
            "active_metric": "",
            "active_dimension": "",
        }
    }
    result = deterministic_investigation_fallback(
        "Compare against married customers",
        df,
        data_context={"conversation_context": conversation_context},
    )

    assert result["trace_metadata"]["fallback"] == "continuation_after_refusal"
    assert "no active analysis" in result["final_answer"].lower()
    assert "`Salary_LPA` by `City` is led by" not in result["final_answer"]
    assert " is led by " not in result["final_answer"]


def test_executive_risk_answer_has_no_planning_exposure_filler() -> None:
    """Risk analysis should not contain 'planning exposure' or 'decision uncertainty' filler."""
    from source.product.llm_reasoning import _local_grounded_synthesis, ReasoningIntent

    evidence = {"computed_summary": "Sales is dominated by a single customer.", "data_profile": {}, "artifacts": []}
    intent = ReasoningIntent(mode="risk_analysis", kind="business", requires_llm=False)
    result = _local_grounded_synthesis(evidence, intent)

    assert "planning exposure" not in result
    assert "decision uncertainty" not in result
    assert "not broadly repeatable" not in result
    assert "Secondary signal" not in result
    # Post-cleanup: local synthesis restates evidence rather than using filler templates
    assert "dominated" in result or "customer" in result.lower() or "sales" in result.lower()


def test_distant_alias_refusal_uses_human_readable_language() -> None:
    """Distant alias refusal must not contain resolver jargon like 'reliable alias'."""
    df = pd.DataFrame(
        {
            "Z_Revenue": [100.0, 200.0, 300.0],
            "Education": ["High School", "College", "Graduate"],
        }
    )
    result = deterministic_investigation_fallback("Build histogram of Sales in LA", df)

    assert "reliable alias" not in result["final_answer"]
    # Should explain in human-readable terms
    assert "does not appear to be" in result["final_answer"] or "does not contain" in result["final_answer"]


def test_english_only_refusal_for_russian_query() -> None:
    """Russian query requesting a field that doesn't exist must produce English refusal."""
    df = pd.DataFrame(
        {
            "Z_Revenue": [100.0, 200.0, 300.0],
            "Education": ["High School", "College", "Graduate"],
        }
    )
    # "Построй график зарплат по профессиям" = "Build chart of salaries by professions"
    # Neither salaries/professions exist in this dataset
    result = deterministic_investigation_fallback("Построй график зарплат по профессиям", df)

    # Refusal must be in English
    answer = result["final_answer"]
    assert any(eng in answer for eng in ("does not contain", "cannot", "not ", "I found")), f"Expected English refusal, got: {answer[:200]}"


def test_joinability_global_intent_produces_findings_and_chart() -> None:
    """Joinability intent should produce at least one finding and one chart artifact."""
    df = pd.DataFrame(
        {
            "Sales": [100.0, 200.0, 300.0],
            "City": ["NYC", "LA", "Chicago"],
            "Region": ["East", "West", "Midwest"],
        }
    )
    result = deterministic_investigation_fallback(
        "Can these datasets be joined reliably?",
        df,
        data_context={"data_source_ids": ["ds_1", "ds_2"]},
    )

    assert result["trace_metadata"]["fallback"] == "global_intent_bypass"
    key_findings = result["structured_report"]["key_findings"]
    assert len(key_findings) >= 1
    assert any("numeric" in f.lower() or "joinab" in f.lower() for f in key_findings)
    artifacts = result.get("artifacts", [])
    assert len(artifacts) >= 1
    assert any(a.get("artifact_type") == "chart" for a in artifacts)
    assert result["trace_metadata"].get("suppress_key_findings") is not True


def test_warehouse_global_intent_produces_findings() -> None:
    """Warehouse design intent should produce findings with fact/dimension classification."""
    df = pd.DataFrame(
        {
            "Revenue": [100.0, 200.0],
            "Cost": [50.0, 80.0],
            "Department": ["Sales", "Marketing"],
            "Quarter": ["Q1", "Q2"],
        }
    )
    result = deterministic_investigation_fallback(
        "Design a warehouse across these datasets",
        df,
        data_context={"data_source_ids": ["ds_1", "ds_2"]},
    )

    assert result["trace_metadata"]["fallback"] == "global_intent_bypass"
    key_findings = result["structured_report"]["key_findings"]
    assert len(key_findings) >= 1
    assert any("fact" in f.lower() or "dimension" in f.lower() or "candidate" in f.lower() for f in key_findings)
    artifacts = result.get("artifacts", [])
    assert len(artifacts) >= 1


def test_schema_comparison_produces_heatmap() -> None:
    """Schema/semantic reasoning intent should produce a heatmap artifact."""
    df = pd.DataFrame(
        {
            "Order_ID": [1, 2, 3],
            "Amount": [100.0, 200.0, 300.0],
            "Category": ["A", "B", "C"],
            "Order_Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )
    result = deterministic_investigation_fallback(
        "What conceptual overlap exists between datasets?",
        df,
        data_context={"data_source_ids": ["ds_1", "ds_2"]},
    )

    assert result["trace_metadata"]["fallback"] == "global_intent_bypass"
    artifacts = result.get("artifacts", [])
    heatmaps = [a for a in artifacts if isinstance(a.get("content"), dict) and a["content"].get("chart_type") == "heatmap"]
    assert len(heatmaps) >= 1, f"Expected at least one heatmap, got artifacts: {[a.get('title') for a in artifacts]}"


def test_global_intent_findings_not_suppressed() -> None:
    """Global intent responses must not set suppress_key_findings."""
    df = pd.DataFrame(
        {
            "Sales": [100.0, 200.0, 300.0],
            "City": ["NYC", "LA", "Chicago"],
        }
    )
    result = deterministic_investigation_fallback(
        "What shared entities exist between these datasets?",
        df,
        data_context={"data_source_ids": ["ds_1", "ds_2"]},
    )

    assert result["trace_metadata"]["fallback"] == "global_intent_bypass"
    assert result["trace_metadata"].get("suppress_key_findings") is not True
    assert len(result["structured_report"]["key_findings"]) >= 1


def _survey_income_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Occupation": ["Engineer", "Teacher", "Engineer", "Artist", "Teacher", "Doctor", "Doctor", "Artist"],
            "Monthly Income": [
                "10001 to 25000",
                "Below Rs.10000",
                "More than 50000",
                "No Income",
                "10001 to 25000",
                "25001 to 50000",
                "More than 50000",
                "Below Rs.10000",
            ],
            "Age": [34, 29, 41, 22, 31, 50, 45, 27],
            "Educational Qualifications": ["Graduate", "Graduate", "Graduate", "High School", "High School", "Postgraduate", "Postgraduate", "High School"],
            "Gender": ["Female", "Male", "Male", "Female", "Male", "Male", "Female", "Female"],
            "Feedback": ["Positive", "Negative", "Positive", "Negative", "Negative", "Positive", "Negative", "Positive"],
        }
    )


def test_explicit_locked_total_income_by_occupation_uses_range_midpoints_not_age() -> None:
    question = (
        "Show the top 10 occupations by total monthly income.\n"
        "Use Occupation as the grouping column and Monthly Income as the metric.\n"
        "Do NOT use averages."
    )
    result = deterministic_investigation_fallback(question, _survey_income_df())
    plan = result["trace_metadata"]["query_plan"]

    assert result["trace_metadata"]["fallback"] == "explicit_constraint_semantic_planner"
    assert plan["operation"] == "GROUPED_AGGREGATION"
    assert plan["grouping_columns"] == ["Occupation"]
    assert plan["metric_columns"] == ["Monthly Income"]
    assert plan["aggregation"] == "sum"
    assert plan["constraints_locked"]["grouping_columns"] is True
    assert plan["constraints_locked"]["metric_columns"] is True
    assert plan["constraints_locked"]["aggregation"] is True
    assert plan["metric"] != "Age"
    assert "Average" not in result["artifacts"][0]["title"]
    report = result["structured_report"]
    assert any("midpoint" in item.lower() for item in report["limitations"] + report["evidence"])
    assert any((artifact.get("artifact_type") or artifact.get("type")) == "chart" for artifact in result["artifacts"])


def test_feedback_visualization_uses_feedback_by_education_and_gender_not_age() -> None:
    result = deterministic_investigation_fallback(
        "Build a visualization of positive vs negative feedback by educational qualification and gender.",
        _survey_income_df(),
    )
    plan = result["trace_metadata"]["query_plan"]

    assert result["trace_metadata"]["analysis_type"] == "categorical_outcome_breakdown"
    assert plan["target_column"] == "Feedback"
    assert plan["grouping_columns"] == ["Educational Qualifications", "Gender"]
    assert plan["metric"] is None
    assert "Age" not in str(plan)
    assert any((artifact.get("artifact_type") or artifact.get("type")) == "chart" for artifact in result["artifacts"])


def test_negative_feedback_risk_as_income_decreases_uses_ordered_range_analysis() -> None:
    result = deterministic_investigation_fallback(
        "Which demographic groups show the strongest increase in negative feedback risk as income decreases?",
        _survey_income_df(),
    )
    plan = result["trace_metadata"]["query_plan"]

    assert result["trace_metadata"]["analysis_type"] == "ordered_outcome_risk"
    assert plan["target_column"] == "Feedback"
    assert plan["ordered_column"] == "Monthly Income"
    assert "Age" not in str(plan)
    table = next(artifact for artifact in result["artifacts"] if (artifact.get("artifact_type") or artifact.get("type")) == "table")
    assert table["content"]
    assert "increase_as_order_decreases" in table["content"][0]


def test_explicit_constraint_critic_rejects_lost_metric_group_or_aggregation() -> None:
    from source.product.grounding_critic import explicit_constraint_plan_check

    question = (
        "Show the top 10 occupations by total monthly income. "
        "Use Occupation as the grouping column and Monthly Income as the metric. "
        "Do NOT use averages."
    )
    bad_output = {
        "summary": "Age by Gender was averaged.",
        "trace_metadata": {
            "query_plan": {
                "metric": "Age",
                "dimension": "Gender",
                "grouping_columns": ["Gender"],
                "metric_columns": ["Age"],
                "aggregation": "mean",
            }
        },
        "artifacts": [],
    }

    issue = explicit_constraint_plan_check(question, bad_output, _survey_income_df())
    assert issue is not None
    assert "metric" in issue.lower() or "aggregation" in issue.lower()
