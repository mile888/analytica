from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from source.config import ANALYTICA_CHECKPOINTER_PATH, ANALYTICA_CHECKPOINTER_TYPE


@dataclass
class CheckpointerHandle:
    saver: Any
    checkpointer_type: str
    path: Path | None = None
    context: AbstractContextManager[Any] | None = None

    def close(self) -> None:
        if self.context is not None:
            self.context.__exit__(None, None, None)
            self.context = None


_CHECKPOINTER_HANDLE: CheckpointerHandle | None = None


def _create_memory_checkpointer() -> CheckpointerHandle:
    from langgraph.checkpoint.memory import InMemorySaver

    return CheckpointerHandle(
        saver=InMemorySaver(),
        checkpointer_type="memory",
    )


def _create_sqlite_checkpointer(path: Path) -> CheckpointerHandle:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SQLite checkpointer requested but `langgraph-checkpoint-sqlite` is not installed. "
            "Install project requirements or run `python3 -m pip install langgraph-checkpoint-sqlite`."
        ) from exc

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        context = SqliteSaver.from_conn_string(str(path))
        saver = context.__enter__()
        if hasattr(saver, "setup"):
            saver.setup()
        return CheckpointerHandle(
            saver=saver,
            checkpointer_type="sqlite",
            path=path,
            context=context,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not start SQLite checkpointer at {path}: {exc}") from exc


def create_checkpointer(
    checkpointer_type: str | None = None,
    path: str | Path | None = None,
) -> CheckpointerHandle:
    resolved_type = (checkpointer_type or ANALYTICA_CHECKPOINTER_TYPE or "memory").strip().lower()
    if resolved_type in {"", "memory"}:
        return _create_memory_checkpointer()
    if resolved_type == "sqlite":
        return _create_sqlite_checkpointer(Path(path) if path is not None else ANALYTICA_CHECKPOINTER_PATH)
    raise ValueError(
        "Invalid ANALYTICA_CHECKPOINTER_TYPE. Expected `memory` or `sqlite`, "
        f"got `{resolved_type}`."
    )


def get_checkpointer_handle() -> CheckpointerHandle:
    global _CHECKPOINTER_HANDLE
    expected_type = (ANALYTICA_CHECKPOINTER_TYPE or "memory").strip().lower()
    expected_path = ANALYTICA_CHECKPOINTER_PATH if expected_type == "sqlite" else None
    if (
        _CHECKPOINTER_HANDLE is None
        or _CHECKPOINTER_HANDLE.checkpointer_type != expected_type
        or _CHECKPOINTER_HANDLE.path != expected_path
    ):
        reset_checkpointer_for_tests()
        _CHECKPOINTER_HANDLE = create_checkpointer(expected_type, expected_path)
    return _CHECKPOINTER_HANDLE


def get_checkpointer() -> Any:
    return get_checkpointer_handle().saver


def reset_checkpointer_for_tests() -> None:
    global _CHECKPOINTER_HANDLE
    if _CHECKPOINTER_HANDLE is not None:
        _CHECKPOINTER_HANDLE.close()
        _CHECKPOINTER_HANDLE = None
