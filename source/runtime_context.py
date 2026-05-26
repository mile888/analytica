from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalyticaContext:
    """Typed DeepAgents runtime context for one user/session/run."""

    user_id: str = "local"
    thread_id: str = ""
    session_id: str = ""
    run_id: str = ""
    artifact_dir: str = ""
    memory_file: str = "/memories/AGENTS.md"
    run_state: dict[str, Any] = field(default_factory=dict)
