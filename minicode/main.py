from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

from minicode.agent_loop import run_agent_turn
from minicode.cli_commands import try_handle_local_command
from minicode.config import load_runtime_config
from minicode.history import load_history_entries, save_history_entries
from minicode.local_tool_shortcuts import parse_local_tool_shortcut
from minicode.manage_cli import maybe_handle_management_command
from minicode.model_registry import create_model_adapter
from minicode.permissions import PermissionManager
from minicode.prompt import build_system_prompt
from minicode.tools import create_default_tool_registry
from minicode.tooling import ToolContext
from minicode.tui.transcript import format_transcript_text
from minicode.tui.types import TranscriptEntry
from minicode.tty_app import run_tty_app
from minicode.workspace import resolve_tool_path


def _handle_local_command(user_input: str, tools, permissions=None) -> str | None:
    if user_input == "/tools":
        return "\n".join(f"{tool.name}: {tool.description}" for tool in tools.list())
    local_result = try_handle_local_command(user_input, tools=tools, cwd=str(Path.cwd()), permissions=permissions)
    return local_result


def _render_banner(runtime: dict | None, cwd: str, permission_summary: list[str], counts: dict[str, int]) -> str:
    model = runtime["model"] if runtime else "unconfigured"
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  🤖 MiniCode Python - Your Terminal Coding Assistant    ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║  Model: {model:<46} ║",
        f"║  CWD: {cwd:<50} ║",
    ]
    if permission_summary:
        for perm in permission_summary[:2]:  # 只显示前2个权限摘要
            lines.append(f"║  {perm:<60} ║")
    lines.append("╠══════════════════════════════════════════════════════════╣")
    lines.append(
        f"║  📊 Skills: {counts['skillCount']:>2} | MCP Servers: {counts['mcpCount']:>2} | "
        f"Transcript: {counts['transcriptCount']:>3} ║"
    )
    lines.append("╚══════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def _render_quick_start() -> str:
    """显示快速入门指南"""
    return """
💡 Quick Start Guide:
  📝 Edit files:     edit_file.py or patch_file.py
  🔍 Search code:    /grep <pattern> or grep_files tool
  🏃 Run commands:   /cmd <command> or run_command tool
  🧠 Think deeply:   Use sequential_thinking MCP tool
  📚 View skills:    /skills
  ❓ Get help:       /help

🚀 Try saying:
  "帮我分析这个项目的结构"
  "用 TDD 方式实现 XX 功能"
  "系统性地调试这个 bug"
  "帮我写个技术方案"
"""


def _append_transcript(transcript: list[TranscriptEntry], **kwargs) -> None:
    transcript.append(TranscriptEntry(id=len(transcript) + 1, **kwargs))


def _make_cli_permission_prompt():
    """Create a simple CLI-based permission prompt for non-TTY fallback."""
    def _prompt(request: dict) -> dict:
        print(f"\n{request.get('summary', 'Permission Request')}")
        choices = request.get("choices", [])
        if choices:
            for choice in choices:
                print(f"  [{choice.get('key', '')}] {choice.get('label', '')}")
            answer = input("Choose: ").strip()
            for choice in choices:
                if answer == choice.get("key"):
                    return {"decision": choice.get("decision", "allow_once")}
        answer = input("Allow? (y/n): ").strip().lower()
        return {"decision": "allow_once" if answer in ("y", "yes") else "deny_once"}
    return _prompt


def _configure_stdio_for_unicode() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _save_transcript_file(cwd: str, permissions, transcript: list[TranscriptEntry], output_path: str) -> str:
    target = resolve_tool_path(ToolContext(cwd=cwd, permissions=permissions), output_path, "write")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_transcript_text(transcript), encoding="utf-8")
    return str(target)

def main() -> None:
    # Step 1: 入口先修正终端编码，避免中文输入/输出在 Windows 终端里变成乱码。
    _configure_stdio_for_unicode()

    # Step 2: CLI 参数只负责“怎么启动本次会话”，真正的 Agent 能力稍后再组装。
    parser = argparse.ArgumentParser(
        description="MiniCode Python - A lightweight terminal coding assistant",
        add_help=True,
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="SESSION_ID",
        help="Resume a previous session (use 'latest' or session ID)",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List all saved sessions and exit",
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="SESSION_ID",
        help="Start with a specific session ID",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run the interactive installer",
    )
    parser.add_argument(
        "--validate-config",
        "--valid-config",
        action="store_true",
        help="Validate configuration and exit",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: WARNING)",
    )

    args, remaining_argv = parser.parse_known_args()
    if remaining_argv and not any(not arg.startswith("--") for arg in remaining_argv):
        parser.error(f"unrecognized arguments: {' '.join(remaining_argv)}")

    # Step 3: 先初始化日志；后续配置、模型、工具、上下文出错时都能留下诊断信息。
    from minicode.logging_config import setup_logging
    setup_logging(level=args.log_level)

    # Step 4: 配置校验属于管理命令，完成后直接退出，不进入 Agent 对话循环。
    if args.validate_config:
        from minicode.config import format_config_diagnostic
        print(format_config_diagnostic())
        return
    
    # Step 5: 安装流程也是独立流程；它只负责写配置，不需要加载模型或工具。
    if args.install:
        from minicode.install import main as install_main
        install_main()
        return
    
    cwd = str(Path.cwd())
    argv = remaining_argv
    
    # Step 6: /session、/resume 这类管理命令先处理；命中后不用再启动交互界面。
    management_argv = [a for a in argv if not a.startswith("--")]
    if maybe_handle_management_command(cwd, management_argv):
        return

    runtime = None
    try:
        # Step 7: runtime 是模型名、API key、MCP 配置等运行时信息的统一入口。
        runtime = load_runtime_config(cwd)
    except Exception as e:  # noqa: BLE001
        # Step 8: 配置坏了也不让程序崩掉，而是降级到 mock model，保证初学者还能跑通流程。
        runtime = None
        print(
            f"⚠️  Warning: Failed to load runtime config: {e}\n",
            file=sys.stderr,
        )
        print(
            "🔧 How to fix this:\n"
            "  1. Set your model name: export ANTHROPIC_MODEL=claude-sonnet-4-20250514\n"
            "  2. Set your API key: export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  3. Or edit ~/.mini-code/settings.json:\n"
            '     {"model": "claude-sonnet-4-20250514", "env": {"ANTHROPIC_API_KEY": "sk-ant-..."}}\n'
            "  4. Restart MiniCode\n\n"
            "📖 For more info: https://github.com/QUSETIONS/MiniCode-Python\n"
            "   Falling back to mock model for now...\n",
            file=sys.stderr,
        )

    prompt_handler = _make_cli_permission_prompt() if sys.stdin.isatty() else None
    # Step 9: 工具注册表把本地文件工具、命令工具、MCP 工具统一成同一种 ToolDefinition。
    tools = create_default_tool_registry(cwd, runtime=runtime)
    permissions = PermissionManager(cwd, prompt=prompt_handler)
    
    # Step 10: 模型适配器把不同供应商接口包成统一的 model.next(messages)。
    force_mock = runtime is None
    model = create_model_adapter(
        model=runtime.get("model", "") if runtime else "",
        tools=tools,
        runtime=runtime,
        force_mock=force_mock,
    )
    
    # Step 11: ContextManager 负责观察上下文窗口，接近上限时触发压缩。
    from minicode.context_manager import ContextManager
    from minicode.logging_config import get_logger
    logger = get_logger("main")
    context_mgr = None
    if runtime:
        context_mgr = ContextManager(model=runtime.get("model", "default"))
        logger.info("Context manager initialized for model: %s", runtime.get("model", "unknown"))
    
    # Step 12: MemoryManager 负责跨轮次/跨会话记忆，后面会把相关记忆注入 system prompt。
    from minicode.memory import MemoryManager
    memory_mgr = MemoryManager(project_root=Path(cwd))
    logger.info("Memory manager initialized")
    
    # Step 13: 用户偏好单独加载，避免把“用户习惯”硬编码进主循环。
    from minicode.user_profile import UserProfileManager
    profile_manager = UserProfileManager(cwd=cwd)
    profile_manager.load_merged()
    logger.info("User profile manager initialized (global=%s, project=%s)",
                profile_manager.global_path.exists(),
                profile_manager.project_path.exists())
    
    # Step 14: Store 保存 UI/运行时状态，例如当前 workspace、model、busy/idle。
    from minicode.state import create_app_store
    app_store = create_app_store(
        initial={
            "session_id": args.session or "new",
            "workspace": cwd,
            "model": runtime.get("model", "mock") if runtime else "mock",
        }
    )
    logger.info("Store initialized with session: %s", app_store.get_state().session_id)
    
    messages = [
        {
            "role": "system",
            # Step 15: 第一条消息必须是系统提示；它把工具、权限、MCP、记忆组织成模型的“工作说明书”。
            "content": build_system_prompt(
                cwd,
                permissions.get_summary(),
                {
                    "skills": tools.get_skills(),
                    "mcpServers": tools.get_mcp_servers(),
                    # Step 16: 首轮还没有用户问题，所以先注入项目级通用记忆。
                    "memory_context": memory_mgr.get_relevant_context(),
                },
            ),
        }
    ]
    history = load_history_entries()
    transcript: list[TranscriptEntry] = []

    print(
        _render_banner(
            runtime,
            cwd,
            permissions.get_summary(),
            {
                "transcriptCount": 0,
                "messageCount": len(messages),
                "skillCount": len(tools.get_skills()),
                "mcpCount": len(tools.get_mcp_servers()),
            },
        )
    )
    
    # 显示快速入门指南
    if not sys.stdin.isatty() or os.environ.get("MINI_CODE_SHOW_GUIDE", "1") == "1":
        print(_render_quick_start())
    else:
        print("")

    try:
        if not sys.stdin.isatty():
            for raw_input in sys.stdin:
                user_input = raw_input.strip()
                if not user_input:
                    continue
                if user_input == "/exit":
                    break
                if user_input.startswith("/transcript-save "):
                    # Step 17: transcript 保存是本地命令，不进入模型，避免无意义消耗 token。
                    output_path = user_input[len("/transcript-save ") :].strip()
                    if not output_path:
                        print("Usage: /transcript-save <path>")
                        continue
                    saved_path = _save_transcript_file(cwd, permissions, transcript, output_path)
                    print(f"Saved transcript to {saved_path}")
                    continue
                memory_result = memory_mgr.handle_user_memory_input(user_input)
                if memory_result is not None:
                    # Step 18: 记忆写入命令由本地处理，成功后只回显结果，不让模型再解释一遍。
                    _append_transcript(transcript, kind="user", body=user_input)
                    _append_transcript(transcript, kind="assistant", body=memory_result)
                    print(memory_result)
                    continue
                local_result = _handle_local_command(user_input, tools, permissions=permissions)
                if local_result is not None:
                    # Step 19: /tools、/help 等本地命令直接返回，减少一次模型调用。
                    _append_transcript(transcript, kind="user", body=user_input)
                    _append_transcript(transcript, kind="assistant", body=local_result)
                    print(local_result)
                    continue
                shortcut = parse_local_tool_shortcut(user_input)
                if shortcut is not None:
                    # Step 20: /cmd、/grep 这类快捷工具直接执行；适合用户明确指定工具的场景。
                    _append_transcript(transcript, kind="user", body=user_input)
                    if (
                        permissions.get_mode().value == "plan"
                        and shortcut["toolName"] not in {"read_file", "list_files", "grep_files"}
                    ):
                        blocked = (
                            "Plan mode blocks direct execution and file modification shortcuts. "
                            "Use /execute after you approve the plan."
                        )
                        _append_transcript(transcript, kind="assistant", body=blocked)
                        print(blocked)
                        continue
                    result = tools.execute(
                        shortcut["toolName"],
                        shortcut["input"],
                        context=ToolContext(cwd=cwd, permissions=permissions),
                    )
                    _append_transcript(
                        transcript,
                        kind="tool",
                        body=result.output,
                        toolName=shortcut["toolName"],
                        status="success" if result.ok else "error",
                    )
                    print(result.output)
                    continue
                _append_transcript(transcript, kind="user", body=user_input)
                # Step 21: 普通自然语言输入才进入对话历史，成为模型下一步决策的依据。
                messages.append({"role": "user", "content": user_input})
                history.append(user_input)
                save_history_entries(history)
                messages[0] = {
                    "role": "system",
                    # Step 22: 每轮都重建系统提示，让“当前问题相关的记忆”优先进入上下文。
                    "content": build_system_prompt(
                        cwd,
                        permissions.get_summary(),
                        {
                            "skills": tools.get_skills(),
                            "mcpServers": tools.get_mcp_servers(),
                            "memory_context": memory_mgr.get_relevant_context(query=user_input),
                        },
                    ),
                }
                permissions.begin_turn()
                # Step 23: 真正的 Agent 回合在 run_agent_turn 中完成：模型思考、调用工具、处理结果、继续循环。
                messages = run_agent_turn(
                    model=model,
                    tools=tools,
                    messages=messages,
                    cwd=cwd,
                    permissions=permissions,
                    store=app_store,
                    context_manager=context_mgr,
                    runtime=runtime,
                )
                permissions.end_turn()
                
                # Step 24: 回合结束后记录上下文使用量，方便后续判断是否需要压缩。
                if context_mgr:
                    stats = context_mgr.get_stats()
                    logger.debug("After turn: %d tokens (%.0f%%)", stats.total_tokens, stats.usage_percentage)
                last_assistant = next((message for message in reversed(messages) if message["role"] == "assistant"), None)
                if last_assistant:
                    _append_transcript(transcript, kind="assistant", body=last_assistant["content"])
                    print(last_assistant["content"])
            return

        run_tty_app(
            runtime=runtime,
            tools=tools,
            model=model,
            messages=messages,
            cwd=cwd,
            permissions=permissions,
            resume_session=args.resume,
            list_sessions_only=args.list_sessions,
            memory_manager=memory_mgr,
            context_manager=context_mgr,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Shutting down gracefully...")
    finally:
        # Graceful shutdown: clean up all resources
        from minicode.logging_config import get_logger
        logger = get_logger("main")
        logger.info("Shutting down...")
        
        # Dispose tools (closes MCP connections)
        try:
            tools.dispose()
            logger.info("Tools disposed successfully")
        except Exception as e:
            logger.warning("Error disposing tools: %s", e)
        
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
