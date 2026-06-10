from pathlib import Path

from minicode.checkpoints import CheckpointManager, format_checkpoint_list, format_checkpoint_show


def test_checkpoint_rollback_restores_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("old\n", encoding="utf-8")
    manager = CheckpointManager(tmp_path)

    record = manager.create([target], reason="before edit", tool_name="test")
    target.write_text("new\n", encoding="utf-8")
    manager.rollback(record.checkpoint_id)

    assert target.read_text(encoding="utf-8") == "old\n"


def test_checkpoint_rollback_deletes_file_that_did_not_exist(tmp_path: Path) -> None:
    target = tmp_path / "new.txt"
    manager = CheckpointManager(tmp_path)

    record = manager.create([target], reason="before create", tool_name="test")
    target.write_text("created\n", encoding="utf-8")
    manager.rollback(record.checkpoint_id)

    assert not target.exists()


def test_checkpoint_accepts_workspace_relative_paths(tmp_path: Path) -> None:
    target = tmp_path / "relative.txt"
    target.write_text("old\n", encoding="utf-8")
    manager = CheckpointManager(tmp_path)

    record = manager.create(["relative.txt"], reason="before edit", tool_name="test")
    target.write_text("new\n", encoding="utf-8")
    manager.rollback(record.checkpoint_id)

    assert target.read_text(encoding="utf-8") == "old\n"


def test_checkpoint_commands_format_records(tmp_path: Path) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("old\n", encoding="utf-8")
    record = CheckpointManager(tmp_path).create([target], reason="before edit", tool_name="test")

    listing = format_checkpoint_list(tmp_path)
    detail = format_checkpoint_show(tmp_path, record.checkpoint_id)

    assert record.checkpoint_id in listing
    assert "before edit" in listing
    assert "demo.txt" in detail
