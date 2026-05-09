from __future__ import annotations

from typing import Any, Callable

from source.skills.registry import get_skill_content, list_skill_summaries


def _record_tool_event(run_context: dict[str, Any], tool_name: str, status: str, **metadata: Any) -> None:
    from datetime import datetime

    event = {
        "tool": tool_name,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    event.update({key: value for key, value in metadata.items() if value not in (None, "")})
    run_context.setdefault("tool_timeline", []).append(event)


def build_skill_tools(run_context: dict[str, Any]) -> list[Callable[..., dict[str, str]]]:
    """Create progressive-disclosure skill tools bound to one run context."""

    def list_available_skills() -> dict[str, str]:
        """List available skills with lightweight descriptions only."""
        skills = list_skill_summaries()
        _record_tool_event(run_context, "list_available_skills", "ok", count=len(skills))
        return {
            "skills": "\n".join(
                f"- {skill['name']}: {skill['description']}"
                for skill in skills
            )
        }

    def load_skill(skill_name: str) -> dict[str, str]:
        """Load the full content of one skill on demand.

        Use this before applying specialized rules for data analysis, CSV/DataFrame
        work, SQL querying, visualization, reporting, business analysis, or code
        execution safety.
        """
        skill = get_skill_content(skill_name)
        if skill is None:
            available = ", ".join(skill["name"] for skill in list_skill_summaries())
            _record_tool_event(run_context, "load_skill", "error", skill_name=skill_name)
            return {
                "loaded": "false",
                "skill_name": skill_name,
                "content": "",
                "error": f"Skill '{skill_name}' not found. Available skills: {available}",
            }

        loaded = run_context.setdefault("loaded_skills", [])
        if skill.name not in loaded:
            loaded.append(skill.name)
        _record_tool_event(run_context, "load_skill", "ok", skill_name=skill.name)
        return {
            "loaded": "true",
            "skill_name": skill.name,
            "allowed_tools": ", ".join(skill.allowed_tools),
            "content": skill.content,
            "error": "",
        }

    return [list_available_skills, load_skill]
