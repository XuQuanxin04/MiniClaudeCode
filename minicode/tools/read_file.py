from __future__ import annotations

import time
from pathlib import Path

from minicode.tooling import ToolDefinition, ToolResult
from minicode.workspace import resolve_tool_path

DEFAULT_READ_LIMIT = 8000
MAX_READ_LIMIT = 20000

# 文件内容缓存，避免重复读取同一文件
# 缓存键：(文件路径，修改时间) -> (内容, 缓存时间)
_file_cache: dict[tuple[str, float], tuple[str, float]] = {}
_FILE_CACHE_TTL = 2.0  # 缓存有效期 2 秒


def _get_cached_file_content(target: Path) -> str:
    """获取文件内容，使用缓存避免重复读取"""
    try:
        stat = target.stat()
        mtime = stat.st_mtime
        # Step 1: 缓存键包含修改时间；文件被改过后会自动绕开旧缓存。
        cache_key = (str(target), mtime)
        
        if cache_key in _file_cache:
            content, cache_time = _file_cache[cache_key]
            # Step 2: 只缓存很短时间，减少同一轮重复 read_file 的 IO，又不长期持有旧内容。
            now = time.monotonic()
            if now - cache_time <= _FILE_CACHE_TTL:
                return content
        
        # Step 3: 顺手清理过期缓存，避免长会话里缓存字典一直增长。
        now = time.monotonic()
        expired_keys = [k for k, (c, t) in _file_cache.items() if now - t > _FILE_CACHE_TTL]
        for k in expired_keys:
            del _file_cache[k]
        
        # Step 4: 真正读磁盘后写入缓存，后续相同文件的短时间读取会直接复用。
        content = target.read_text(encoding="utf-8")
        _file_cache[cache_key] = (content, time.monotonic())
        return content
    except OSError:
        return ""


def _validate(input_data: dict) -> dict:
    # Step 1: read_file 必须有 path；offset/limit 用于分段读取大文件。
    path = input_data.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("path is required")
    offset = int(input_data.get("offset", 0))
    limit = int(input_data.get("limit", DEFAULT_READ_LIMIT))
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if limit < 1 or limit > MAX_READ_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_READ_LIMIT}")
    return {"path": path, "offset": offset, "limit": limit}


def _run(input_data: dict, context) -> ToolResult:
    # Step 1: 所有路径先经过 workspace/permission 解析，防止模型直接读越界路径。
    target = resolve_tool_path(context, input_data["path"], "read")

    try:
        # Step 2: 读取时走短缓存，同一轮反复查看文件不会重复打磁盘。
        content = _get_cached_file_content(target)
    except UnicodeDecodeError:
        # Step 3: 二进制文件不硬读，直接告诉模型不能按 UTF-8 文本处理。
        return ToolResult(
            ok=False,
            output=f"File {input_data['path']} appears to be binary. Cannot read as text.",
        )
    
    offset = input_data["offset"]
    limit = input_data["limit"]
    end = min(len(content), offset + limit)
    chunk = content[offset:end]
    truncated = end < len(content)
    # Step 4: header 明确告诉模型本次读了哪个区间；TRUNCATED=yes 时模型应继续按 offset 读取。
    header = "\n".join(
        [
            f"FILE: {input_data['path']}",
            f"OFFSET: {offset}",
            f"END: {end}",
            f"TOTAL_CHARS: {len(content)}",
            f"TRUNCATED: {'yes - call read_file again with offset ' + str(end) if truncated else 'no'}",
            "",
        ]
    )
    return ToolResult(ok=True, output=header + chunk)


read_file_tool = ToolDefinition(
    name="read_file",
    description="Read a UTF-8 text file relative to the workspace root.",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "number"}, "limit": {"type": "number"}}, "required": ["path"]},
    validator=_validate,
    run=_run,
)

