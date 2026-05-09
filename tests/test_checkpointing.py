from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver

import source.checkpointing as checkpointing


def test_default_memory_checkpointer_works():
    handle = checkpointing.create_checkpointer("memory")

    try:
        assert handle.checkpointer_type == "memory"
        assert isinstance(handle.saver, InMemorySaver)
    finally:
        handle.close()


def test_sqlite_checkpointer_can_be_created(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    handle = checkpointing.create_checkpointer("sqlite", path)

    try:
        assert handle.checkpointer_type == "sqlite"
        assert handle.path == path
        assert path.exists()
    finally:
        handle.close()


def test_sqlite_checkpointer_restores_same_thread_after_reinitialization(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "demo", "checkpoint_ns": ""}}

    first = checkpointing.create_checkpointer("sqlite", path)
    try:
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"memory_probe": "persisted"}
        checkpoint["channel_versions"] = {"memory_probe": "1"}
        first.saver.put(config, checkpoint, {"source": "test", "step": 1}, {"memory_probe": "1"})
    finally:
        first.close()

    second = checkpointing.create_checkpointer("sqlite", path)
    try:
        restored = second.saver.get_tuple(config)
        assert restored is not None
        assert restored.checkpoint["channel_values"]["memory_probe"] == "persisted"
    finally:
        second.close()


def test_invalid_checkpointer_config_gives_clear_error():
    try:
        checkpointing.create_checkpointer("bad-type")
    except ValueError as exc:
        assert "ANALYTICA_CHECKPOINTER_TYPE" in str(exc)
        assert "memory" in str(exc)
        assert "sqlite" in str(exc)
    else:
        raise AssertionError("Invalid checkpointer type did not raise")
