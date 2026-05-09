from source.tools.skill_tools import build_skill_tools


def _skill_tools(context=None):
    context = context if context is not None else {"loaded_skills": []}
    return context, {tool.__name__: tool for tool in build_skill_tools(context)}


def test_list_available_skills_returns_lightweight_descriptions():
    context, tools = _skill_tools()

    result = tools["list_available_skills"]()

    assert "sql_querying" in result["skills"]
    assert "Workflow" not in result["skills"]
    assert context["tool_timeline"][0]["tool"] == "list_available_skills"


def test_load_skill_returns_full_content_and_tracks_loaded_skill():
    context, tools = _skill_tools()

    result = tools["load_skill"]("sql_querying")

    assert result["loaded"] == "true"
    assert result["skill_name"] == "sql_querying"
    assert "Workflow" in result["content"]
    assert "sql_querying" in context["loaded_skills"]
    assert context["tool_timeline"][-1]["tool"] == "load_skill"


def test_load_missing_skill_returns_error():
    _, tools = _skill_tools()

    result = tools["load_skill"]("missing_skill")

    assert result["loaded"] == "false"
    assert "not found" in result["error"]
