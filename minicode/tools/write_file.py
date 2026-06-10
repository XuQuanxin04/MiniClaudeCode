from __future__ import annotations

from minicode.file_review import apply_reviewed_file_change
from minicode.tooling import ToolDefinition
from minicode.workspace import resolve_tool_path


def _validate(input_data: dict) -> dict:
    # Step 1: write_file 是整文件写入，所以必须同时提供 path 和完整 content。
    path = input_data.get("path")
    content = input_data.get("content")
    if not isinstance(path, str) or not path:
        raise ValueError("path is required")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    return {"path": path, "content": content}


def _run(input_data: dict, context):
    # Step 2: 先解析目标路径并触发写权限检查，再把真正写入交给 file_review 统一处理。
    target = resolve_tool_path(context, input_data["path"], "write")
    # Step 3: apply_reviewed_file_change 会生成 diff/走权限审批/落盘，避免写文件绕过审核链路。
    return apply_reviewed_file_change(context, input_data["path"], target, input_data["content"])


write_file_tool = ToolDefinition(
    name="write_file",
    description="Write a UTF-8 text file relative to the workspace root.",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    validator=_validate,
    run=_run,
)

