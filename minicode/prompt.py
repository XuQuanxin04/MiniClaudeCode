from __future__ import annotations

from pathlib import Path

from minicode.prompt_pipeline import PromptPipeline, read_file_cached


def _maybe_read(path: Path) -> str | None:
    """Read file content with caching (reuses pipeline cache)."""
    return read_file_cached(path)


def _engineering_governance_rules() -> str:
    r"""Return engineering governance rules as system prompt section.

    These rules are mandatory and apply to all code generation activities.
    Based on: D:\Desktop\engineering-governance
    """
    return """## Engineering Governance Rules (MANDATORY)

These rules apply to ALL code you write. No exceptions.

### Iron Laws
1. **Theory first**: Read theory before any engineering activity
2. **Requirements first**: No code without design, no design without requirements
3. **1:1 binding**: Requirements and knowledge always appear in pairs
4. **Design-driven**: Code implements design, not independent creation
5. **Audit loop**: Execute audit after each phase, fail → fix → re-audit
6. **Single sink**: business/src/ must have exactly ONE sink file
7. **One-way dependencies**: All dependency flow is unidirectional, zero cycles
8. **No skipping**: Each phase's exit signals must be met before next phase

### Package Structure (Six Areas)
Every package must have:
- `port/port_entry/` — Entry points (can import anything)
- `wrap/src/` — External library adapters (import: port_entry, wrap/config, wrap/src)
- `business/src/` — Business logic (import: wrap sinks, business/config, business/src)
- `test/src/` — Tests (import: business/src, test/config, test/src)
- `business/config/` — Business config (zero dependencies)
- `wrap/config/` — Adapter config (zero dependencies)
- `test/config/` — Test config (zero dependencies)

### Dependency Direction Rules
- `business/src/` → `wrap/src/` sinks → `port/port_entry/` → `vendor/`
- `business/src/` CANNOT import vendor/, external libs directly
- `wrap/src/` CANNOT import business/src/
- Config imports always come LAST in import statements
- Cross-package: port_exit → port_entry (same language to same language)

### Sink Rule
- `business/src/`: EXACTLY ONE sink (file not imported by other business/src/ files)
- `wrap/src/`: Can have multiple sinks (each must be used by business/src/)
- `test/src/`: Can have multiple sinks (all must be used by port_exit)
- Multiple sinks in business/src/ = MUST split package

### Documentation System
- Requirements → Knowledge → Design → Code (strict one-way flow)
- Each requirement scenario has exactly one matching knowledge file (1:1 path mirror)
- Each design file cites: satisfied requirements, depended knowledge
- Code file paths must be isomorphic to design file paths

### Import Sorting Example
```python
# Non-config imports first
from package.wrap/src/adapter import Adapter
from package.business/src/service import Service

# Config imports LAST
from package.business/config import settings
```

### Audit Checklist (Execute After Code Changes)
Audit 0: Knowledge ↔ Requirements 1:1
Audit 1: Design ← Requirements + Knowledge coverage
Audit 2: Code ← Design isomorphism + Dependency compliance
Audit 3: business/src/ single sink + Package DAG

### Boundary Packaging (Legacy Code)
- When introducing legacy code: only through port_entry → wrap/src/ ([LEGACY] tag)
- Each [LEGACY] file must have expected cleanup date
- Legacy code can reference governance area via port_exit directly

### Repository Rules
- ZERO compositional dependencies between repositories
- Cross-repository needs: copy to local vendor/
- Vendor only imported by port_entry/"""


def build_system_prompt(
    cwd: str,
    permission_summary: list[str] | None = None,
    extras: dict | None = None,
) -> str:
    """Build the system prompt using dynamic paragraph assembly.

    Implements cache boundaries:
    - Static prefix (role, governance rules) is cacheable across turns.
    - Dynamic suffix (skills, MCP, CLAUDE.md) is re-evaluated per turn.

    Args:
        cwd: Current working directory
        permission_summary: Permission context for the prompt
        extras: Optional extras dict with skills, mcpServers, etc.
    """
    cwd_path = Path(cwd)
    permission_summary = permission_summary or []
    extras = extras or {}

    # Step 1: PromptPipeline 把 system prompt 拆成多个段落，静态段可缓存，动态段每轮更新。
    pipeline = PromptPipeline()

    # Step 2: 静态前缀定义 Agent 身份、工作习惯、工具使用原则和结构化输出协议。
    pipeline.register_static(
        "role",
        "You are mini-code, a terminal coding assistant.\n"
        "Default behavior: inspect the repository, use tools, make code changes when appropriate, and explain results clearly.\n"
        "Prefer reading files, searching code, editing files, and running verification commands over giving purely theoretical advice.\n"
        f"Current cwd: {cwd}\n"
        "You can inspect or modify paths outside the current cwd when the user asks, but tool permissions may pause for approval first.\n"
        "When making code changes, keep them minimal, practical, and working-oriented.\n"
        "If the user clearly asked you to build, modify, optimize, or generate something, do the work instead of stopping at a plan.\n"
        "If the Permission context says `permission mode: plan`, inspect and plan only: do not edit files or run non-read-only commands until the user switches with /execute.\n"
        "If you need user clarification, call the ask_user tool with one concise question and wait for the user reply. Do not ask clarifying questions as plain assistant text.\n"
        "Do not choose subjective preferences such as colors, visual style, copy tone, or naming unless the user explicitly told you to decide yourself.\n"
        "When using read_file, pay attention to the header fields. If it says TRUNCATED: yes, continue reading with a larger offset before concluding that the file itself is cut off.\n"
        "If the user names a skill or clearly asks for a workflow that matches a listed skill, call load_skill before following it.\n"
        "\n"
        "## Sub-agent (task tool) usage guide\n"
        "You have access to the 'task' tool which can spawn sub-agents for complex work. Use it when:\n"
        "- You need to explore a large codebase without bloating the main context (agent_type='explore')\n"
        "- You need thorough analysis of a codebase area before acting (agent_type='plan')\n"
        "- You need to do multi-step work that benefits from isolation (agent_type='general')\n"
        "Do NOT use the task tool for simple lookups — use read_file/grep_files directly.\n"
        "Do NOT use the task tool just to avoid work — use it when it genuinely improves efficiency.\n"
        "\n"
        "Structured response protocol:\n"
        "- When you are still working and will continue with more tool calls, start your text with <progress>.\n"
        "- Only when the task is actually complete and you are ready to hand control back, start your text with <final>.\n"
        "- Use ask_user when clarification is required; that tool ends the turn and waits for user input.\n"
        "- Do not stop after a progress update. After a <progress> message, continue the task in the next step.\n"
        "- Plain assistant text without <progress> is treated as a completed assistant message for this turn.",
    )

    pipeline.register_static(
        "governance",
        # Step 3: 工程治理规则属于强约束，和角色说明一起作为长期稳定的 system prompt。
        _engineering_governance_rules(),
    )

    # Step 4: 以下是动态后缀；权限、技能、MCP、记忆会随着项目和会话变化。
    if permission_summary:
        perm_text = "Permission context:\n" + "\n".join(permission_summary)
        # Step 5: 权限摘要告诉模型哪些操作可能需要确认，减少模型盲目调用危险工具。
        pipeline.register_dynamic("permissions", lambda: perm_text)

    # Step 6: skills 是可发现能力；存在时注入清单，不存在时也给模型一个明确的“无技能”状态。
    skills = extras.get("skills", [])
    if skills:
        def _build_skills():
            lines = ["Available skills:"]
            lines.extend(
                f"- {skill['name']}: {skill['description']}" for skill in skills
            )
            lines.extend([
                "",
                "SKILL USAGE GUIDE:",
                "- When user asks for creative brainstorming, use 'brainstorming' skill",
                "- When writing implementation plans, use 'writing-plans' skill",
                "- When debugging systematically, use 'systematic-debugging' skill",
                "- When doing TDD, use 'test-driven-development' skill",
                "- When reviewing code in Chinese, use 'chinese-code-review' skill",
                "- When user asks about workflows, check 'using-superpowers' skill first",
                "- For complex multi-step tasks, consider 'subagent-driven-development'",
                "- Before completing, ALWAYS use 'verification-before-completion'",
            ])
            return "\n".join(lines)

        # Step 7: 技能列表动态生成，安装/删除 skill 后下一轮 prompt 可以自动反映新状态。
        pipeline.register_dynamic("skills", _build_skills)
    else:
        pipeline.register_dynamic(
            "no_skills",
            lambda: (
                "Available skills:\n- none discovered\n"
                "Tip: Install skills via `npx superpowers-zh` in your project directory"
            ),
        )

    # Step 8: MCP 区域告诉模型哪些外部服务已连接，以及 MCP 工具的命名规则。
    mcp_servers = extras.get("mcpServers", [])
    if mcp_servers:
        def _build_mcp():
            lines = ["Configured MCP servers:"]
            lines.extend(
                "- "
                + server["name"]
                + f": {server['status']}, tools={server['toolCount']}"
                + (f", resources={server['resourceCount']}" if server.get("resourceCount") is not None else "")
                + (f", prompts={server['promptCount']}" if server.get("promptCount") is not None else "")
                + (f", protocol={server['protocol']}" if server.get("protocol") else "")
                + (f" ({server['error']})" if server.get("error") else "")
                for server in mcp_servers
            )
            if any(server.get("status") == "connected" for server in mcp_servers):
                lines.append(
                    "Connected MCP tools are already exposed in the tool list with names prefixed like mcp__server__tool. "
                    "Use list_mcp_resources/read_mcp_resource and list_mcp_prompts/get_mcp_prompt when a server exposes those capabilities."
                )
            # Step 9: 某些 MCP 工具有特殊使用场景，例如 sequential thinking，单独提示更容易被模型调用。
            sequential_servers = [
                server for server in mcp_servers
                if "sequential" in server.get("name", "").lower()
                or "branch-thinking" in server.get("name", "").lower()
                or "think" in server.get("name", "").lower()
            ]
            if any(server.get("status") == "connected" for server in sequential_servers):
                lines.extend([
                    "",
                    "SEQUENTIAL THINKING MCP SERVER IS CONNECTED!",
                    "When to use sequential_thinking tool:",
                    "- Breaking down complex implementation problems",
                    "- Multi-step debugging or investigation",
                    "- Architectural decisions requiring structured analysis",
                    "- Migration or refactoring planning",
                    "- Any situation requiring step-by-step reasoning",
                    "",
                    "Usage: Call 'sequential_thinking' with structured thoughts before complex tool sequences",
                ])
            return "\n".join(lines)

        pipeline.register_dynamic("mcp", _build_mcp, cache_ttl=60.0)

    memory_context = str(extras.get("memory_context") or "").strip()
    if memory_context:
        # Step 10: 记忆不是普通聊天消息，而是作为“项目背景和历史决策”注入 system prompt。
        pipeline.register_dynamic(
            "memory",
            lambda: (
                "## Project Memory & Context\n\n"
                "The following information has been accumulated from previous sessions. "
                "Use it to preserve project conventions and decisions:\n\n"
                f"{memory_context}"
            ),
            cache_ttl=30.0,
        )

    # Step 11: 全局 CLAUDE.md 保存用户跨项目偏好，读取后加入动态 prompt。
    global_claude_md = _maybe_read(Path.home() / ".claude" / "CLAUDE.md")
    if global_claude_md:
        pipeline.register_dynamic(
            "global_claude_md",
            lambda: f"Global instructions from ~/.claude/CLAUDE.md:\n{global_claude_md}",
            cache_ttl=600.0,
        )

    # Step 12: 项目 CLAUDE.md 保存当前仓库规则，优先级比普通记忆更像“项目说明书”。
    project_claude_md = _maybe_read(cwd_path / "CLAUDE.md")
    if project_claude_md:
        pipeline.register_dynamic(
            "project_claude_md",
            lambda: f"Project instructions from {cwd_path / 'CLAUDE.md'}:\n{project_claude_md}",
            cache_ttl=300.0,
        )

    # Step 13: 最后由 pipeline 统一拼接，调用方只拿到一条完整 system prompt。
    return pipeline.build()
