from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from source.config import SKILL_DIR


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    """Progressively disclosed skill metadata and full content."""

    name: str
    description: str
    content: str
    allowed_tools: tuple[str, ...] = ()
    path: Path | None = None


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text.strip()

    metadata: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, text[match.end() :].strip()


def _parse_allowed_tools(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


def _read_skill(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    metadata, content = _parse_frontmatter(text)
    fallback_name = path.parent.name.replace("-", "_")
    name = (metadata.get("name") or fallback_name).strip()
    description = (metadata.get("description") or "").strip()
    return Skill(
        name=name,
        description=description,
        content=content,
        allowed_tools=_parse_allowed_tools(metadata.get("allowed-tools", "")),
        path=path,
    )


@lru_cache(maxsize=1)
def load_skill_registry() -> dict[str, Skill]:
    """Load skills from disk once per process."""
    registry: dict[str, Skill] = {}
    if not SKILL_DIR.exists():
        return registry

    for path in sorted(SKILL_DIR.rglob("SKILL.md")):
        if not path.is_file():
            continue
        skill = _read_skill(path)
        registry[skill.name] = skill
    return registry


def skill_descriptions_text() -> str:
    """Return lightweight skill descriptions for the system prompt."""
    registry = load_skill_registry()
    if not registry:
        return "- no skills registered"
    lines = []
    for skill in registry.values():
        tools = ", ".join(skill.allowed_tools) if skill.allowed_tools else "direct answer"
        lines.append(f"- `{skill.name}`: {skill.description} Tools: {tools}.")
    return "\n".join(lines)


def list_skill_summaries() -> list[dict[str, str]]:
    """Return metadata-only skill summaries safe for upfront context."""
    return [
        {
            "name": skill.name,
            "description": skill.description,
            "allowed_tools": ", ".join(skill.allowed_tools),
        }
        for skill in load_skill_registry().values()
    ]


def get_skill_content(skill_name: str) -> Skill | None:
    """Look up a skill by canonical name or simple hyphen/underscore variant."""
    registry = load_skill_registry()
    key = (skill_name or "").strip()
    if key in registry:
        return registry[key]
    normalized = key.replace("-", "_")
    for name, skill in registry.items():
        if name.replace("-", "_") == normalized:
            return skill
    return None
