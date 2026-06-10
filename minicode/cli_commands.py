from __future__ import annotations

from dataclasses import dataclass

from minicode.config import (
    CLAUDE_SETTINGS_PATH,
    MINI_CODE_MCP_PATH,
    MINI_CODE_PERMISSIONS_PATH,
    MINI_CODE_SETTINGS_PATH,
    load_runtime_config,
    save_mini_code_settings,
)


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    usage: str
    description: str


SLASH_COMMANDS = [
    SlashCommand("/help", "/help", "Show available slash commands."),
    SlashCommand("/tools", "/tools", "List tools available to the coding agent and tool shortcuts."),
    SlashCommand("/state", "/state", "Show detailed application state and Store summary."),
    SlashCommand("/status", "/status", "Show application state summary and current model."),
    SlashCommand("/cost", "/cost [--detailed]", "Show API cost and usage report."),
    SlashCommand("/context", "/context", "Show context window usage."),
    SlashCommand("/cybernetics", "/cybernetics", "Show cybernetic control system status."),
    SlashCommand("/tasks", "/tasks", "Show current task list."),
    SlashCommand("/memory", "/memory", "Show memory system status."),
    SlashCommand("/config", "/config", "Show configuration diagnostics and validation."),
    SlashCommand("/history", "/history", "Show recent prompt history from ~/.mini-code/history.json."),
    SlashCommand("/clear", "/clear", "Clear the current transcript view."),
    SlashCommand("/retry", "/retry", "Retry the last natural-language prompt in this session."),
    SlashCommand("/transcript-save", "/transcript-save <path>", "Save the current session transcript to a text file."),
    SlashCommand("/model", "/model", "Show the current model."),
    SlashCommand("/model", "/model <model-name>", "Persist a model override into ~/.mini-code/settings.json."),
    SlashCommand("/config-paths", "/config-paths", "Show mini-code and Claude fallback settings paths."),
    SlashCommand("/skills", "/skills", "List discovered SKILL.md workflows."),
    SlashCommand("/mcp", "/mcp", "Show configured MCP servers and connection state."),
    SlashCommand("/permissions", "/permissions", "Show mini-code permission storage path."),
    SlashCommand("/plan", "/plan", "Switch to read-only planning mode."),
    SlashCommand("/execute", "/execute", "Switch back to normal execution mode."),
    SlashCommand("/mode", "/mode [default|auto|plan|bypass]", "Show or change the current permission mode."),
    SlashCommand("/checkpoint", "/checkpoint [list|show <id>|rollback <id>]", "List, inspect, or roll back MiniCode file checkpoints."),
    SlashCommand("/exit", "/exit", "Exit mini-code."),
    SlashCommand("/debug", "/debug", "Show scroll and terminal diagnostics."),
    SlashCommand("/user", "/user", "Show or manage user profile (preferences, coding style)."),
    SlashCommand("/ls", "/ls [path]", "List files in a directory."),
    SlashCommand("/grep", "/grep <pattern>::[path]", "Search text in files."),
    SlashCommand("/read", "/read <path>", "Read a file directly."),
    SlashCommand("/write", "/write <path>::<content>", "Write a file directly."),
    SlashCommand("/modify", "/modify <path>::<content>", "Replace a file, showing a reviewable diff before applying it."),
    SlashCommand("/edit", "/edit <path>::<search>::<replace>", "Edit a file by exact replacement."),
    SlashCommand("/patch", "/patch <path>::<search1>::<replace1>::<search2>::<replace2>...", "Apply multiple replacements to one file in one command."),
    SlashCommand("/cmd", "/cmd [cwd::]<command> [args...]", "Run an allowed development command directly."),
]


def format_slash_commands() -> str:
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  📚 Available Commands                                  ║",
        "╠══════════════════════════════════════════════════════════╣",
    ]
    
    command_groups = {
        "🔧 Core Commands": [
            ("/help", "Show this help message"),
            ("/exit", "Exit mini-code"),
            ("/clear", "Clear the current transcript view"),
            ("/history", "Show recent prompt history"),
        ],
        "🛠️ Tool Commands": [
            ("/tools", "List all available tools"),
            ("/skills", "List discovered SKILL.md workflows"),
            ("/mcp", "Show MCP servers and connection state"),
            ("/cmd", "Run development commands directly"),
        ],
        "📊 Status & Info": [
            ("/status", "Show application state summary"),
            ("/model", "Show or change current model"),
            ("/user", "Show or manage user profile"),
            ("/cost", "Show API cost and usage report"),
            ("/context", "Show context window usage"),
            ("/cybernetics", "Show control-system status"),
            ("/tasks", "Show current task list"),
            ("/memory", "Show memory system status"),
        ],
        "🧭 Planning & Recovery": [
            ("/plan", "Switch to read-only planning mode"),
            ("/execute", "Return to reviewed execution mode"),
            ("/mode", "Show or change permission mode"),
            ("/checkpoint", "List, inspect, or roll back checkpoints"),
        ],
        "✏️ File Operations": [
            ("/ls [path]", "List files in directory"),
            ("/grep <pattern>", "Search text in files"),
            ("/read <path>", "Read a file directly"),
            ("/write <path>", "Write content to file"),
            ("/edit <path>", "Edit file by exact replacement"),
            ("/patch <path>", "Apply multiple replacements in one go"),
            ("/modify <path>", "Replace file with reviewable diff"),
        ],
        "💾 Session Management": [
            ("/transcript-save <path>", "Save transcript to text file"),
            ("/retry", "Retry the last prompt"),
            ("/permissions", "Show permission storage path"),
            ("/config-paths", "Show settings file paths"),
        ],
    }
    
    for group_name, commands in command_groups.items():
        lines.append(f"║  {group_name:<54}║")
        for cmd, desc in commands:
            cmd_display = f"    {cmd}"
            lines.append(f"║  {cmd_display:<20} {desc:<33} ║")
        lines.append("╠══════════════════════════════════════════════════════════╣")
    
    lines.extend([
        "║  💡 Tips:                                              ║",
        "║  - Use Tab to autocomplete commands                    ║",
        "║  - Prefix with / to access any command                 ║",
        "║  - Type naturally - I'll understand Chinese & English  ║",
        "╚══════════════════════════════════════════════════════════╝",
    ])
    
    return "\n".join(lines)


def find_matching_slash_commands(user_input: str) -> list[str]:
    """Find slash commands matching user input.

    Tries exact prefix first, falls back to fuzzy subsequence matching.
    """
    commands = [c.usage for c in SLASH_COMMANDS]
    prefix_matches = [c for c in commands if c.startswith(user_input)]
    if prefix_matches:
        return prefix_matches
    # Fuzzy fallback: subsequence match (e.g., "mem" matches "/memory")
    lower = user_input.lower()
    fuzzy = [c for c in commands if all(ch in c.lower() for ch in lower)]
    return fuzzy if fuzzy else commands


def complete_slash_command(line: str) -> tuple[list[str], str]:
    commands = [c.usage for c in SLASH_COMMANDS]
    hits = [c for c in commands if c.startswith(line)]
    if not hits and line:
        lower = line.lower()
        hits = [c for c in commands if all(ch in c.lower() for ch in lower)]
    return (hits if hits else commands, line)


def try_handle_local_command(user_input: str, tools=None, cwd: str | None = None, permissions=None) -> str | None:
    if user_input in {"/", "/help"}:
        return format_slash_commands()

    if user_input == "/config-paths":
        return "\n".join(
            [
                f"mini-code settings: {MINI_CODE_SETTINGS_PATH}",
                f"mini-code permissions: {MINI_CODE_PERMISSIONS_PATH}",
                f"mini-code mcp: {MINI_CODE_MCP_PATH}",
                f"compat fallback: {CLAUDE_SETTINGS_PATH}",
            ]
        )

    if user_input == "/permissions":
        return f"permission store: {MINI_CODE_PERMISSIONS_PATH}"

    if user_input == "/plan":
        from minicode.auto_mode import PermissionMode, set_permission_mode
        if permissions is not None:
            message = permissions.set_mode(PermissionMode.PLAN)
        else:
            message = set_permission_mode(PermissionMode.PLAN)
        return "\n".join([
            message,
            "Use this mode to read files, search code, inspect structure, and produce an implementation plan.",
            "File edits, write tools, and non-read-only commands are blocked until you run /execute.",
        ])

    if user_input == "/execute":
        from minicode.auto_mode import PermissionMode, set_permission_mode
        if permissions is not None:
            message = permissions.set_mode(PermissionMode.DEFAULT)
        else:
            message = set_permission_mode(PermissionMode.DEFAULT)
        return "\n".join([
            message,
            "The agent can now apply reviewed edits and run approved development commands.",
        ])

    if user_input == "/mode":
        if permissions is not None:
            return permissions.format_mode_status()
        from minicode.auto_mode import get_mode_state
        return get_mode_state().format_status()

    if user_input.startswith("/mode "):
        from minicode.auto_mode import set_permission_mode
        mode_name = user_input[len("/mode "):].strip()
        try:
            if permissions is not None:
                message = permissions.set_mode(mode_name)
            else:
                message = set_permission_mode(mode_name)
        except ValueError as error:
            return str(error)
        return message

    if user_input == "/checkpoint" or user_input == "/checkpoint list":
        from pathlib import Path
        from minicode.checkpoints import format_checkpoint_list
        return format_checkpoint_list(Path(cwd) if cwd else Path.cwd())

    if user_input.startswith("/checkpoint show "):
        from pathlib import Path
        from minicode.checkpoints import format_checkpoint_show
        checkpoint_id = user_input[len("/checkpoint show "):].strip()
        try:
            return format_checkpoint_show(Path(cwd) if cwd else Path.cwd(), checkpoint_id)
        except ValueError as error:
            return str(error)

    if user_input.startswith("/checkpoint rollback "):
        from pathlib import Path
        from minicode.checkpoints import CheckpointManager
        checkpoint_id = user_input[len("/checkpoint rollback "):].strip()
        manager = CheckpointManager(Path(cwd) if cwd else Path.cwd())
        try:
            record = manager.rollback(checkpoint_id)
        except ValueError as error:
            return str(error)
        return (
            f"Rolled back checkpoint {record.checkpoint_id}.\n"
            f"Restored {len(record.files)} path(s)."
        )

    if user_input == "/skills":
        skills = tools.get_skills() if tools else []
        if not skills:
            return "No skills discovered. Add skills under ~/.mini-code/skills/<name>/SKILL.md, .mini-code/skills/<name>/SKILL.md, .claude/skills/<name>/SKILL.md, or ~/.claude/skills/<name>/SKILL.md."
        return "\n".join(
            f"{skill['name']}  {skill['description']}  [{skill['source']}]"
            for skill in skills
        )

    if user_input == "/config":
        from minicode.config import format_config_diagnostic
        return format_config_diagnostic()

    if user_input == "/state":
        try:
            from minicode.state import handle_state_command
            return handle_state_command()
        except ImportError:
            return "State system not available. Please ensure state.py exists."

    if user_input == "/memory":
        # Memory system display
        try:
            from minicode.memory import MemoryManager
            from pathlib import Path
            memory_mgr = MemoryManager(project_root=Path(cwd) if cwd else Path.cwd())
            return memory_mgr.format_stats()
        except Exception as e:
            return f"Error loading memory: {e}"

    if user_input == "/context":
        # Context usage display
        try:
            from minicode.context_manager import load_context_state
            from minicode.context_compactor import ContextCompactor
            ctx_mgr = load_context_state()
            compactor = ContextCompactor(
                context_window=ctx_mgr.context_window if ctx_mgr else 200000,
                workspace=cwd,
            )
            if ctx_mgr:
                return "\n\n".join([
                    ctx_mgr.format_context_details(),
                    compactor.explain_layers(),
                ])
            return "\n\n".join([
                "No context state available. Context tracking starts after first turn.",
                compactor.explain_layers(),
            ])
        except Exception as e:
            return f"Error loading context: {e}"

    if user_input == "/cybernetics":
        return format_cybernetics_status()

    if user_input == "/mcp":
        servers = tools.get_mcp_servers() if tools else []
        if not servers:
            return "No MCP servers configured. Add mcpServers to ~/.mini-code/settings.json, ~/.mini-code/mcp.json, or project .mcp.json."
        lines = []
        for server in servers:
            suffix = f"  error={server['error']}" if server.get("error") else ""
            protocol = f"  protocol={server['protocol']}" if server.get("protocol") else ""
            resources = f"  resources={server['resourceCount']}" if server.get("resourceCount") is not None else ""
            prompts = f"  prompts={server['promptCount']}" if server.get("promptCount") is not None else ""
            lines.append(
                f"{server['name']}  status={server['status']}  tools={server['toolCount']}{resources}{prompts}{protocol}{suffix}"
            )
        return "\n".join(lines)

    if user_input == "/status":
        try:
            runtime = load_runtime_config()
        except Exception as error:  # noqa: BLE001
            return f"runtime not configured: {error}"
        from minicode.model_registry import detect_provider
        provider = detect_provider(runtime["model"], runtime)
        auth_methods = []
        if runtime.get("authToken"):
            auth_methods.append("ANTHROPIC_AUTH_TOKEN")
        if runtime.get("apiKey"):
            auth_methods.append("ANTHROPIC_API_KEY")
        if runtime.get("openaiApiKey"):
            auth_methods.append("OPENAI_API_KEY")
        if runtime.get("openrouterApiKey"):
            auth_methods.append("OPENROUTER_API_KEY")
        if runtime.get("customApiKey"):
            auth_methods.append("CUSTOM_API_KEY")
        return "\n".join(
            [
                f"model: {runtime['model']}",
                f"provider: {provider.value}",
                f"baseUrl: {runtime['baseUrl']}",
                f"auth: {', '.join(auth_methods) or 'none'}",
                f"mcp servers: {len(runtime.get('mcpServers', {}))}",
                runtime["sourceSummary"],
            ]
        )

    if user_input == "/model":
        try:
            runtime = load_runtime_config()
            from minicode.model_registry import format_model_status
            return format_model_status(runtime["model"], runtime)
        except Exception as error:  # noqa: BLE001
            return f"runtime not configured: {error}"

    if user_input.startswith("/model "):
        arg = user_input[len("/model "):].strip()
        if not arg:
            from minicode.model_registry import format_model_list
            return format_model_list()
        # Subcommands
        if arg in ("status", "info"):
            try:
                runtime = load_runtime_config()
                from minicode.model_registry import format_model_status
                return format_model_status(runtime["model"], runtime)
            except Exception as error:  # noqa: BLE001
                return f"runtime not configured: {error}"
        if arg in ("list", "ls"):
            from minicode.model_registry import format_model_list
            return format_model_list()
        # Provider filter: /model anthropic, /model openrouter, etc.
        from minicode.model_registry import Provider, format_model_list
        for p in Provider:
            if arg.lower() == p.value:
                return format_model_list(provider=p)
        # Otherwise: set model name
        save_mini_code_settings({"model": arg})
        return f"saved model={arg} to {MINI_CODE_SETTINGS_PATH}\nRestart MiniCode for the change to take effect."

    if user_input == "/user" or user_input.startswith("/user "):
        from minicode.user_profile import handle_user_command
        args = user_input[len("/user"):].strip()
        return handle_user_command(args)

    return None


def format_cybernetics_status() -> str:
    """Format cybernetic controller inventory and persisted state hints."""
    from minicode.cybernetic_supervisor import CyberneticSupervisor, load_supervisor_report
    from minicode.context_manager import load_context_state

    controllers = [
        ("ContextCyberneticsOrchestrator", "context pressure PID + prediction"),
        ("CostControlLoop", "budget PID for tool-result persistence"),
        ("VerificationController", "risk-adaptive verification planning"),
        ("ToolSchedulerController", "error/latency-aware concurrency control"),
        ("MemoryInjectionController", "context-aware memory injection"),
        ("ModelSelectionController", "cost/latency/failure-aware model routing"),
        ("ProgressController", "health/stall task progress control"),
        ("CyberneticSupervisor", "global health and risk aggregation"),
    ]

    ctx = load_context_state()
    snapshots = []
    if ctx:
        stats = ctx.get_stats()
        usage = stats.usage_percentage / 100.0
        snapshots.append(CyberneticSupervisor().snapshot_from_context({
            "sensor": {"current_usage": usage},
            "predictor": {"urgency": 0.0},
        }))
    persisted_report = load_supervisor_report()
    report = persisted_report or CyberneticSupervisor().report(snapshots)

    lines = [
        "Cybernetic Control System",
        "=" * 50,
        f"overall_health: {report.overall_health:.2f}",
        f"risk_level: {report.risk_level.value}",
        f"source: {'latest agent-loop report' if persisted_report else 'current persisted context'}",
        "",
        "Controllers:",
    ]
    for name, desc in controllers:
        lines.append(f"  - {name}: {desc}")
    lines.extend([
        "",
        "Runtime aggregation:",
        "  - pipeline outputs: progress_control + verification_plan + cybernetic_supervisor",
        "  - agent loop logs: context + cost + tool scheduling supervisor report",
    ])
    if report.recommended_actions:
        lines.append("")
        lines.append("Current actions:")
        for action in report.recommended_actions[:5]:
            lines.append(f"  - {action}")
    return "\n".join(lines)
