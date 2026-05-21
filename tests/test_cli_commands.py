from minicode.cli_commands import find_matching_slash_commands, format_slash_commands, try_handle_local_command
from minicode.local_tool_shortcuts import parse_local_tool_shortcut


def test_find_matching_slash_commands_returns_help_variants() -> None:
    matches = find_matching_slash_commands("/mo")
    assert "/model" in matches
    assert "/model <model-name>" in matches


def test_find_matching_slash_commands_returns_cybernetics() -> None:
    matches = find_matching_slash_commands("/cy")
    assert "/cybernetics" in matches


def test_parse_local_tool_shortcut_parses_cmd() -> None:
    shortcut = parse_local_tool_shortcut("/cmd src::git status")
    assert shortcut == {
        "toolName": "run_command",
        "input": {"command": "git status", "cwd": "src"},
    }


def test_parse_local_tool_shortcut_parses_patch_pairs() -> None:
    shortcut = parse_local_tool_shortcut("/patch demo.txt::hello::hi::world::earth")
    assert shortcut == {
        "toolName": "patch_file",
        "input": {
            "path": "demo.txt",
            "replacements": [
                {"search": "hello", "replace": "hi"},
                {"search": "world", "replace": "earth"},
            ],
        },
    }


def test_format_slash_commands_includes_permissions() -> None:
    assert "/permissions" in format_slash_commands()


def test_format_slash_commands_describes_patch_replacements() -> None:
    commands = format_slash_commands()
    # 检查格式化后的帮助信息包含关键命令
    assert "/patch" in commands
    assert "replacements" in commands or "multiple" in commands


def test_format_slash_commands_includes_history_and_retry() -> None:
    commands = format_slash_commands()
    assert "/history" in commands
    assert "/retry" in commands
    assert "/cybernetics" in commands


def test_memory_command_uses_current_workspace(tmp_path) -> None:
    result = try_handle_local_command("/memory", cwd=str(tmp_path))

    assert result is not None
    assert "Memory System Status" in result


def test_cybernetics_command_shows_controller_inventory() -> None:
    result = try_handle_local_command("/cybernetics")

    assert result is not None
    assert "Cybernetic Control System" in result
    assert "CyberneticSupervisor" in result
    assert "ProgressController" in result


def test_cybernetics_command_uses_persisted_report(tmp_path, monkeypatch) -> None:
    import minicode.cybernetic_supervisor as supervisor_module
    from minicode.cybernetic_supervisor import ControlSnapshot, CyberneticSupervisor, save_supervisor_report

    monkeypatch.setattr(
        supervisor_module,
        "SUPERVISOR_STATE_PATH",
        tmp_path / "cybernetic_supervisor.json",
    )
    report = CyberneticSupervisor().report([
        ControlSnapshot(name="context", health=0.2, risk=0.9, action="compact")
    ])
    save_supervisor_report(report)

    result = try_handle_local_command("/cybernetics")

    assert result is not None
    assert "source: latest agent-loop report" in result
    assert "context: compact" in result
