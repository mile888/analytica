from source.skills.registry import get_skill_content, list_skill_summaries, load_skill_registry, skill_descriptions_text


def test_skill_registry_contains_expected_skills():
    registry = load_skill_registry()

    expected = {
        "business_analysis",
        "code_execution_safety",
        "csv_dataframe_analysis",
        "data_analysis",
        "reporting",
        "sql_querying",
        "visualization",
    }
    assert expected.issubset(set(registry))


def test_skill_metadata_and_content_are_separate():
    summaries = list_skill_summaries()
    descriptions = skill_descriptions_text()
    sql_skill = get_skill_content("sql_querying")

    assert summaries
    assert sql_skill is not None
    assert sql_skill.description
    assert "Workflow" in sql_skill.content
    assert "SQL Querying" in sql_skill.content
    assert "SQL Querying" not in descriptions


def test_skill_lookup_accepts_hyphen_variant():
    skill = get_skill_content("business-analysis")

    assert skill is not None
    assert skill.name == "business_analysis"
