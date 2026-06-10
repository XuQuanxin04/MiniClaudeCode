from dataclasses import asdict
import os

from minicode.mcp import create_mcp_backed_tools
from minicode.skills import discover_skills
from minicode.tooling import ToolRegistry
from minicode.tools.ask_user import ask_user_tool
from minicode.tools.batch_ops import batch_copy_tool, batch_move_tool, batch_delete_tool
from minicode.tools.code_nav import find_symbols_tool, find_references_tool, get_ast_info_tool
from minicode.tools.code_review import code_review_tool
from minicode.tools.diff_viewer import diff_viewer_tool
from minicode.tools.edit_file import edit_file_tool
from minicode.tools.file_tree import file_tree_tool
from minicode.tools.git import git_tool
from minicode.tools.grep_files import grep_files_tool
from minicode.tools.list_files import list_files_tool
from minicode.tools.load_skill import create_load_skill_tool
from minicode.tools.patch_file import patch_file_tool
from minicode.tools.read_file import read_file_tool
from minicode.tools.run_command import run_command_tool
from minicode.tools.test_runner import test_runner_tool
from minicode.tools.todo_write import todo_write_tool
from minicode.tools.web_fetch import web_fetch_tool
from minicode.tools.web_search import web_search_tool
from minicode.tools.write_file import write_file_tool
from minicode.tools.task import task_tool


_CORE_TOOLS = [
    # Step 1: 核心工具是 Agent 做本地 Coding 任务的最小工具面，默认只暴露这些以降低模型选择成本。
    # User interaction
    ask_user_tool,
    # File operations
    list_files_tool,
    grep_files_tool,
    read_file_tool,
    write_file_tool,
    # modify_file_tool removed: identical to write_file (same _run/_validate)
    edit_file_tool,
    patch_file_tool,
    # Batch operations
    batch_copy_tool,
    batch_move_tool,
    batch_delete_tool,
    # Command execution
    run_command_tool,
    # Web tools
    web_fetch_tool,
    web_search_tool,
    # Task management
    todo_write_tool,
    # Sub-agent
    task_tool,
    # Git workflow
    git_tool,
    # Code intelligence
    find_symbols_tool,
    find_references_tool,
    get_ast_info_tool,
    code_review_tool,
    # Visualization
    file_tree_tool,
    diff_viewer_tool,
    # Testing
    test_runner_tool,
]

def _resolve_tool_profile(runtime: dict | None) -> str:
    # Step 1: 工具档位可由环境变量或 runtime 配置决定，默认 core 保持工具面克制。
    configured = (
        os.environ.get("MINI_CODE_TOOL_PROFILE")
        or (runtime or {}).get("toolProfile")
        or "core"
    )
    return str(configured).strip().lower()


def _is_full_tool_profile(profile: str) -> bool:
    # Step 2: full/utility/all 会额外加载压缩、编码、HTTP、CSV 等通用小工具。
    return profile in {"full", "utility", "utilities", "all"}


def _load_utility_wrapper_tools():
    # Step 3: 稀有工具延迟导入，普通编码会话不为这些包装工具支付启动成本。
    from minicode.tools.archive_utils import (
        gzip_compress_tool, gzip_decompress_tool, tar_create_tool, tar_extract_tool,
        zip_create_tool, zip_extract_tool,
    )
    from minicode.tools.crypto_utils import current_time_tool, timestamp_tool, hash_tool, hmac_tool
    from minicode.tools.csv_utils import csv_parse_tool, csv_create_tool
    from minicode.tools.encoding_utils import base64_encode_tool, base64_decode_tool, url_encode_tool, url_decode_tool
    from minicode.tools.http_utils import http_request_tool
    from minicode.tools.json_utils import json_format_tool, json_parse_tool
    from minicode.tools.regex_utils import regex_test_tool, regex_replace_tool
    from minicode.tools.text_utils import (
        uuid_generate_tool, text_sort_tool, text_dedupe_tool, text_join_tool,
        line_count_tool, random_string_tool,
    )

    return [
        http_request_tool,
        json_format_tool,
        json_parse_tool,
        regex_test_tool,
        regex_replace_tool,
        base64_encode_tool,
        base64_decode_tool,
        url_encode_tool,
        url_decode_tool,
        current_time_tool,
        timestamp_tool,
        hash_tool,
        hmac_tool,
        gzip_compress_tool,
        gzip_decompress_tool,
        tar_create_tool,
        tar_extract_tool,
        zip_create_tool,
        zip_extract_tool,
        csv_parse_tool,
        csv_create_tool,
        uuid_generate_tool,
        text_sort_tool,
        text_dedupe_tool,
        text_join_tool,
        line_count_tool,
        random_string_tool,
    ]


def create_default_tool_registry(cwd: str, runtime: dict | None = None) -> ToolRegistry:
    # Step 1: 先发现 skills；它们会进入 system prompt，指导模型何时调用 load_skill。
    skills = [asdict(skill) for skill in discover_skills(cwd)]
    # Step 2: MCP 工具在这里被包装成普通 ToolDefinition，后面和本地工具统一调度。
    mcp = create_mcp_backed_tools(cwd=cwd, mcp_servers=dict(runtime.get("mcpServers", {})) if runtime else {})
    profile = _resolve_tool_profile(runtime)
    # Step 3: 默认从核心工具开始，保证 Agent 具备读、搜、改、跑、测、git 的基本能力。
    tools = list(_CORE_TOOLS)
    if _is_full_tool_profile(profile):
        # Step 4: 完整档位追加通用工具，但不改变核心工具的行为。
        tools.extend(_load_utility_wrapper_tools())
    tools.extend(
        [
            # Step 5: load_skill 是按需加载技能正文的入口，MCP tools 则来自外部 server。
            create_load_skill_tool(cwd),
            *mcp["tools"],
        ]
    )
    # Step 6: ToolRegistry 是最终交给 AgentLoop 的统一工具入口，负责查找、校验、执行和清理。
    return ToolRegistry(
        tools,
        skills=skills,
        mcp_servers=mcp["servers"],
        disposer=mcp["dispose"],
    )
