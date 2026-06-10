"""Context window management for LLM conversations.

Tracks token usage, estimates context window consumption, and provides
auto-compaction to prevent context overflow in long conversations.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from minicode.config import MINI_CODE_DIR


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default context window sizes (tokens)
DEFAULT_CONTEXT_WINDOWS = {
    # Anthropic
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-haiku-3-20240307": 100_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
    # OpenRouter popular models
    "openrouter/auto": 200_000,
    "anthropic/claude-sonnet-4": 200_000,
    "anthropic/claude-opus-4": 200_000,
    "openai/gpt-4o": 128_000,
    "openai/gpt-4o-mini": 128_000,
    "google/gemini-2.5-pro": 1_000_000,
    "google/gemini-2.5-flash": 1_000_000,
    "meta-llama/llama-4-maverick": 1_000_000,
    "deepseek/deepseek-r1": 128_000,
    "deepseek/deepseek-chat": 128_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "qwen/qwen3-235b-a22b": 128_000,
    "minimax/minimax-m1": 1_000_000,
    "default": 128_000,  # Fallback
}

# Auto-compaction threshold (95% of context window)
AUTOCOMPACT_THRESHOLD = 0.95

# Estimated tokens per character (rough average for English/Code)
CHARS_PER_TOKEN = 4.0

# Minimum messages to keep after compaction
MIN_MESSAGES_TO_KEEP = 10

# System prompt is always kept (counts as 1 message)
SYSTEM_PROMPT_RESERVED = 1


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# 预编译的正则表达式用于快速 CJK 字符检测
_CJK_PATTERN = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]')

# LRU 缓存：token 估算被频繁调用（每条消息、每次上下文检查），
# 相同文本的 token 数是确定性的，缓存可避免重复计算。
_token_cache: dict[str, int] = {}
_TOKEN_CACHE_MAX = 1024


def estimate_tokens(text: str) -> int:
    """改进的 token 估算，支持中英文
    
    - 英文/代码：约 4 字符/token
    - 中文/日文：约 1.5 字符/token
    - 混合文本：使用启发式估算
    
    性能优化：使用正则表达式替代逐字符 ord() 检查，速度快 10-50 倍。
    带 LRU 缓存避免重复计算相同文本。
    """
    if not text:
        return 0
    
    # Step 1: token 估算会被高频调用，先查缓存可以避免同一段文本反复计算。
    cache_key = text if len(text) < 256 else hash(text)  # 长文本用 hash 作为 key
    cached = _token_cache.get(cache_key)
    if cached is not None:
        return cached
    
    # Step 2: 中文/日文/韩文和英文代码 token 密度不同，所以先统计 CJK 字符数量。
    cjk_count = len(_CJK_PATTERN.findall(text))
    
    # Step 3: 用启发式比例合并估算，目标不是精确计费，而是判断上下文压力是否危险。
    ascii_chars = len(text) - cjk_count
    
    result = max(1, int(cjk_count / 1.5 + ascii_chars / 4.0))
    
    # Step 4: 缓存有上限，避免长时间会话把内存越占越大。
    if len(_token_cache) < _TOKEN_CACHE_MAX:
        _token_cache[cache_key] = result
    
    return result


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate tokens for a single message."""
    tokens = 0
    
    # Role overhead
    role = message.get("role", "")
    if role == "system":
        tokens += 3  # System prompt overhead
    elif role == "user":
        tokens += 4  # User message overhead
    elif role == "assistant":
        tokens += 3  # Assistant overhead
    elif role == "assistant_tool_call":
        tokens += 7  # Tool call overhead
    elif role == "tool_result":
        tokens += 6  # Tool result overhead
    elif role == "assistant_progress":
        tokens += 3
    
    # Content tokens
    content = message.get("content", "")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    
    # Tool call input/output
    if "input" in message:
        input_str = json.dumps(message["input"]) if isinstance(message["input"], dict) else str(message["input"])
        tokens += estimate_tokens(input_str)
    
    return tokens


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens for a list of messages."""
    return sum(estimate_message_tokens(msg) for msg in messages)


@dataclass
class _ExtractedInfo:
    """Information extracted from removed messages during summarization."""
    user_intents: list[str] = field(default_factory=list)
    file_paths: set[str] = field(default_factory=set)
    key_tool_results: list[str] = field(default_factory=list)
    assistant_conclusions: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    code_snippets: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)


# Tool categories for classification
_EDIT_TOOLS = frozenset({"edit_file", "write_file", "modify_file", "patch_file", "multi_edit"})
_READ_TOOLS = frozenset({"read_file", "list_files", "grep_files", "file_tree"})
_SEARCH_TOOLS = frozenset({"grep_files", "find_symbols", "find_references", "web_search", "web_fetch"})
_COMMAND_TOOLS = frozenset({"run_command", "execute_command", "bash"})

# Regex for extracting code-like content and decisions
_CODE_FENCE_RE = re.compile(r'```[\w]*\n(.{20,300}?)```', re.DOTALL)
_DECISION_KEYWORDS = re.compile(
    r'(?:decided|decision|chose|chosen|will use|using|switching to|'
    r'implemented|fixed|resolved|refactored|migrated|upgraded|'
    r'recommend|should|must|need to|going to|plan to|'
    r'approach:|strategy:|solution:|conclusion:)',
    re.IGNORECASE,
)


def _extract_from_messages(messages: list[dict[str, Any]]) -> _ExtractedInfo:
    """Extract structured information from messages for layered summarization.
    
    This is the core extraction step that pulls out different categories of
    information at varying levels of detail, enabling the budget-aware builder
    to include the most important information first.
    """
    info = _ExtractedInfo()
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user" and content.strip():
            # Extract user intent — keep more context for short queries,
            # truncate long paste-heavy messages
            preview = content.strip().replace("\n", " ")
            # For short queries (<200 chars), keep them fully
            # For long ones, keep first 200 chars
            if len(preview) > 200:
                preview = preview[:200] + "..."
            info.user_intents.append(preview)
            
        elif role == "assistant" and content.strip():
            text = content.strip()
            
            # Extract decisions/conclusions
            sentences = text.replace("\n", " ").split(". ")
            for sentence in sentences:
                if _DECISION_KEYWORDS.search(sentence):
                    decision = sentence.strip()[:180]
                    if decision and decision not in info.decisions:
                        info.decisions.append(decision)
            
            # Extract code snippets from assistant responses
            for match in _CODE_FENCE_RE.finditer(text):
                snippet = match.group(1).strip()
                if len(snippet) >= 20 and len(info.code_snippets) < 5:
                    info.code_snippets.append(snippet[:300])
            
            # General conclusion preview
            preview = text[:200].replace("\n", " ")
            info.assistant_conclusions.append(preview)
            
        elif role == "assistant_tool_call":
            tool_name = msg.get("toolName", "unknown")
            info.tool_names.append(tool_name)
            
            # Extract file paths from edit/write tools
            if tool_name in _EDIT_TOOLS:
                inp = msg.get("input", {})
                path = inp.get("path") or inp.get("filePath", "")
                if path:
                    info.file_paths.add(path)
            
            # Extract searched patterns from grep/search tools
            if tool_name in _SEARCH_TOOLS:
                inp = msg.get("input", {})
                pattern = inp.get("pattern") or inp.get("query", "")
                if pattern:
                    info.file_paths.add(f"search:{pattern[:80]}")
            
            # Extract command names from run_command
            if tool_name in _COMMAND_TOOLS:
                inp = msg.get("input", {})
                cmd = inp.get("command", "")
                if cmd:
                    cmd_name = cmd.split()[0] if cmd.split() else ""
                    if cmd_name:
                        info.key_tool_results.append(f"ran: {cmd_name}")
            
        elif role == "tool_result":
            tool_name = msg.get("toolName", "")
            is_error = msg.get("isError", False)
            
            # Preserve error results (highest priority tool info)
            if is_error:
                error_preview = content.strip()[:150].replace("\n", " ")
                info.key_tool_results.append(f"ERROR({tool_name}): {error_preview}")
            
            # Preserve edit confirmations with file paths
            elif tool_name in _EDIT_TOOLS and content.strip():
                success_preview = content.strip()[:100].replace("\n", " ")
                info.key_tool_results.append(f"{tool_name} ok: {success_preview}")
            
            # Extract file paths from read_file results
            elif tool_name in _READ_TOOLS and content.strip():
                # Check if content references a file path
                first_line = content.strip().split("\n")[0][:100]
                if "/" in first_line or "\\" in first_line:
                    info.file_paths.add(first_line.strip())
    
    return info


def _build_layered_summary(info: _ExtractedInfo, max_summary_tokens: int = 2000) -> str:
    """Build a budget-aware layered summary from extracted information.
    
    Layers are ordered by importance and each has a token budget allocation:
    - Layer 1: User intents (35% budget) — what the user wanted
    - Layer 2: Decisions & file paths (20% budget) — key choices made
    - Layer 3: Key tool results — errors and important outcomes (15% budget)
    - Layer 4: Assistant conclusions (15% budget) — results reached
    - Layer 5: Code snippets (10% budget) — important code patterns
    - Layer 6: Tool usage summary (5% budget) — compact activity log
    """
    lines: list[str] = []
    
    # Budget allocations per layer (as fraction of total)
    layer_budgets = [0.35, 0.20, 0.15, 0.15, 0.10, 0.05]
    
    def _remaining_budget() -> int:
        return max(0, max_summary_tokens - estimate_tokens("\n".join(lines)))
    
    # Layer 1: User intents (highest priority)
    if info.user_intents:
        budget = int(max_summary_tokens * layer_budgets[0])
        lines.append("## User requests:")
        for intent in info.user_intents[:12]:
            if estimate_tokens("\n".join(lines)) > budget:
                lines.append(f"  ... and {len(info.user_intents) - info.user_intents.index(intent)} more")
                break
            lines.append(f"- {intent}")
    
    # Layer 2: Decisions and file paths
    has_decisions = bool(info.decisions)
    has_files = bool(info.file_paths)
    if has_decisions or has_files:
        budget = int(max_summary_tokens * (layer_budgets[0] + layer_budgets[1]))
        
        if info.decisions:
            lines.append("## Key decisions:")
            for dec in info.decisions[:8]:
                if estimate_tokens("\n".join(lines)) > budget:
                    break
                lines.append(f"- {dec}")
        
        if info.file_paths:
            # Separate real paths from search patterns
            real_paths = sorted(p for p in info.file_paths if not p.startswith("search:"))
            search_patterns = sorted(p[8:] for p in info.file_paths if p.startswith("search:"))
            
            path_line = f"## Files: {', '.join(real_paths[:20])}"
            if len(real_paths) > 20:
                path_line += f" (+{len(real_paths)-20} more)"
            if search_patterns:
                path_line += f"\n## Searched: {', '.join(search_patterns[:5])}"
            
            if estimate_tokens("\n".join(lines) + path_line) <= budget:
                lines.append(path_line)
    
    # Layer 3: Key tool results (errors + edits)
    if info.key_tool_results:
        budget = int(max_summary_tokens * sum(layer_budgets[:3]))
        lines.append("## Key results:")
        for result in info.key_tool_results[:15]:
            if estimate_tokens("\n".join(lines)) > budget:
                break
            lines.append(f"- {result}")
    
    # Layer 4: Assistant conclusions
    if info.assistant_conclusions:
        budget = int(max_summary_tokens * sum(layer_budgets[:4]))
        lines.append("## Conclusions:")
        for conc in info.assistant_conclusions[:8]:
            if estimate_tokens("\n".join(lines)) > budget:
                break
            lines.append(f"- {conc}")
    
    # Layer 5: Code snippets (most selective)
    if info.code_snippets:
        budget = int(max_summary_tokens * sum(layer_budgets[:5]))
        lines.append("## Code patterns:")
        for snippet in info.code_snippets[:3]:
            snippet_line = f"```\n{snippet}\n```"
            if estimate_tokens("\n".join(lines) + snippet_line) > budget:
                break
            lines.append(snippet_line)
    
    # Layer 6: Tool usage summary (most compact)
    if info.tool_names:
        from collections import Counter
        tool_counts = Counter(info.tool_names)
        tool_summary = ", ".join(
            f"{name}×{count}" if count > 1 else name
            for name, count in tool_counts.most_common()
        )
        lines.append(f"## Tools: {tool_summary}")
    
    return "\n".join(lines)


def _summarize_removed_messages(messages: list[dict[str, Any]], max_summary_tokens: int = 2000) -> str:
    """Build a condensed summary of removed messages for context retention.
    
    Uses a two-phase approach:
    1. Extract: Pull structured information from all message types
    2. Build: Assemble layers respecting token budget allocations
    
    This ensures the most important information (user intents, key decisions)
    is always included, while less critical details (tool names, code snippets)
    fill remaining budget.
    """
    if not messages:
        return ""
    
    info = _extract_from_messages(messages)
    return _build_layered_summary(info, max_summary_tokens)


# ---------------------------------------------------------------------------
# Context tracking
# ---------------------------------------------------------------------------

@dataclass
class ContextStats:
    """Current context window statistics."""
    total_tokens: int = 0
    context_window: int = 0
    usage_percentage: float = 0.0
    messages_count: int = 0
    system_tokens: int = 0
    conversation_tokens: int = 0
    tool_calls_count: int = 0
    is_near_limit: bool = False
    should_compact: bool = False


@dataclass
class ContextManager:
    """Manages context window tracking and auto-compaction."""
    model: str = "default"
    context_window: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    compaction_history: list[dict[str, Any]] = field(default_factory=list)
    _token_cache: dict[int, int] = field(default_factory=dict, repr=False)  # id(msg) -> tokens
    
    # 多级压缩支持
    _compaction_level: int = field(default_factory=lambda: 0)  # 0=无压缩, 1=轻微, 2=中等, 3=深度
    
    # 多级压缩目标 (相对于 context window 的百分比)
    _COMPACTION_LEVELS = [0.70, 0.50, 0.30]  # 轻度/中度/深度
    
    def __post_init__(self):
        if self.context_window == 0:
            self.context_window = DEFAULT_CONTEXT_WINDOWS.get(
                self.model, DEFAULT_CONTEXT_WINDOWS["default"]
            )
    
    def update_model(self, model: str) -> None:
        """Update model and adjust context window."""
        self.model = model
        self.context_window = DEFAULT_CONTEXT_WINDOWS.get(
            model, DEFAULT_CONTEXT_WINDOWS["default"]
        )
    
    def add_message(self, message: dict[str, Any]) -> None:
        """Add a message and update tracking."""
        # Step 1: 新消息进入 ContextManager 后立即缓存 token 数，后续 get_stats 就不用重复估算。
        self.messages.append(message)
        self._token_cache[id(message)] = estimate_message_tokens(message)
    
    def get_stats(self) -> ContextStats:
        """Calculate current context statistics.
        
        Uses cached token counts when available (O(1) amortized for
        messages added via add_message).
        """
        if not self.messages:
            return ContextStats(
                context_window=self.context_window,
            )
        
        # Count tokens using cache when available
        system_tokens = 0
        conversation_tokens = 0
        tool_calls = 0
        
        for msg in self.messages:
            # Step 2: 统计时按消息 id 查缓存；缓存缺失再现场估算。
            msg_tokens = self._token_cache.get(id(msg))
            if msg_tokens is None:
                msg_tokens = estimate_message_tokens(msg)
                self._token_cache[id(msg)] = msg_tokens
            if msg.get("role") == "system":
                system_tokens += msg_tokens
            else:
                conversation_tokens += msg_tokens
            
            if msg.get("role") == "assistant_tool_call":
                tool_calls += 1
        
        total_tokens = system_tokens + conversation_tokens
        # Step 3: usage_pct 是压缩触发的核心指标，后续 near_limit/should_compact 都由它派生。
        usage_pct = (total_tokens / self.context_window * 100) if self.context_window > 0 else 0
        
        is_near_limit = usage_pct >= 80  # Warning at 80%
        should_compact = usage_pct >= (AUTOCOMPACT_THRESHOLD * 100)
        
        return ContextStats(
            total_tokens=total_tokens,
            context_window=self.context_window,
            usage_percentage=usage_pct,
            messages_count=len(self.messages),
            system_tokens=system_tokens,
            conversation_tokens=conversation_tokens,
            tool_calls_count=tool_calls,
            is_near_limit=is_near_limit,
            should_compact=should_compact,
        )
    
    def should_auto_compact(self) -> bool:
        """Check if auto-compaction should trigger.
        
        Multi-level trigger:
        - Level 0: Trigger at 95% threshold
        - Level 1: Trigger at 85% threshold  
        - Level 2: Trigger at 75% threshold
        - Level 3: Trigger at 60% threshold (more aggressive)
        """
        stats = self.get_stats()
        # Step 1: 压缩次数越多，阈值越低；说明会话已经偏长，需要更早介入。
        threshold = AUTOCOMPACT_THRESHOLD - (self._compaction_level * 0.10)
        threshold = max(0.60, threshold)  # Minimum 60%
        usage_pct = stats.usage_percentage
        return usage_pct >= (threshold * 100)
    
    def compact_messages(self) -> list[dict[str, Any]]:
        """Compact messages to fit within context window.
        
        Multi-level progressive compression:
        - Level 0 (first compaction): 70% target
        - Level 1 (second compaction): 50% target  
        - Level 2+ (deep compaction): 30% target
        
        Progressive compression strategy with semantic-aware tool pairing:
        1. Keep system prompt (always)
        2. Remove assistant_progress messages (lowest value)
        3. Truncate large tool results in-place (adaptive sizing)
        4. Compress tool_call+result pairs into inline summaries
        5. Remove remaining messages by priority (tool_result > tool_call > assistant > user)
        
        Key improvements over simple priority removal:
        - Tool call+result pairs are compressed (not just deleted), preserving
          the semantic link between what was called and what resulted
        - Tool-specific compression: read-only tools get shorter summaries,
          edit tools preserve file paths, error results preserve error text
        - Recent messages are protected — removal starts from oldest
        - Budget-aware: each phase checks if we've reached the target
        """
        stats = self.get_stats()
        if not stats.should_compact:
            # Step 1: 未到压缩阈值时原样返回，不为了“整洁”牺牲上下文完整性。
            return self.messages
        
        # Step 2: 本次压缩要压到哪个目标比例，由当前压缩等级决定。
        target_pct = self._COMPACTION_LEVELS[min(self._compaction_level, 2)]
        target_tokens = int(self.context_window * target_pct)
        
        # Step 3: system prompt 是模型行为边界，永远保留；只压缩普通对话和工具历史。
        system_messages = [m for m in self.messages if m.get("role") == "system"]
        other_messages = [m for m in self.messages if m.get("role") != "system"]
        
        # Step 4: 第一阶段删 progress；它只是过程播报，对恢复任务目标价值最低。
        filtered = [
            m for m in other_messages
            if m.get("role") != "assistant_progress"
        ]
        
        current_tokens = estimate_messages_tokens(filtered)
        if current_tokens <= target_tokens:
            # Step 5: 如果删 progress 已经够了，就立刻收手，尽量少动历史。
            return self._finalize_compaction(
                system_messages, other_messages, filtered, stats, target_tokens
            )
        
        # Step 6: 第二阶段截断大工具输出；按工具类型给不同预算，因为可重跑性和诊断价值不同。
        _READ_TOOL_TRUNCATE = 1500   # chars to keep for read-only tool results
        _EDIT_TOOL_TRUNCATE = 3000   # chars to keep for edit tool results
        _ERROR_TRUNCATE = 4000       # chars to keep for error results
        _DEFAULT_TRUNCATE = 2000     # default truncation threshold
        
        for i, m in enumerate(filtered):
            if m.get("role") != "tool_result":
                continue
            content = m.get("content", "")
            if not content or len(content) <= _DEFAULT_TRUNCATE:
                continue
            
            tool_name = m.get("toolName", "")
            is_error = m.get("isError", False)
            
            # Step 7: 错误结果最值得保留，读文件结果最容易重跑，所以阈值不同。
            if is_error:
                threshold = _ERROR_TRUNCATE
            elif tool_name in _EDIT_TOOLS:
                threshold = _EDIT_TOOL_TRUNCATE
            elif tool_name in _READ_TOOLS:
                threshold = _READ_TOOL_TRUNCATE
            else:
                threshold = _DEFAULT_TRUNCATE
            
            if len(content) <= threshold:
                continue
            
            # Step 8: 截断保留头尾，中间用提示标记省略，模型能知道信息不完整。
            content_lines = content.split("\n")
            keep_chars = threshold
            head_lines: list[str] = []
            tail_lines: list[str] = []
            head_chars = 0
            
            for line in content_lines:
                if head_chars + len(line) + 1 > keep_chars * 0.7:
                    break
                head_lines.append(line)
                head_chars += len(line) + 1
            
            # Tail: last few lines
            tail_chars = 0
            for line in reversed(content_lines):
                if tail_chars + len(line) + 1 > keep_chars * 0.3:
                    break
                tail_lines.insert(0, line)
                tail_chars += len(line) + 1
            
            omitted = len(content_lines) - len(head_lines) - len(tail_lines)
            truncated_content = "\n".join(head_lines)
            if omitted > 0:
                truncated_content += f"\n... [{omitted} lines truncated for compaction] ...\n"
            truncated_content += "\n".join(tail_lines)
            
            filtered[i] = {**m, "content": truncated_content}
        
        current_tokens = estimate_messages_tokens(filtered)
        if current_tokens <= target_tokens:
            # Step 9: 截断工具输出后如果达标，就不再压缩工具调用语义。
            return self._finalize_compaction(
                system_messages, other_messages, filtered, stats, target_tokens
            )
        
        # Step 10: 第三阶段把 tool_call + tool_result 合成一句摘要，保留“做了什么 -> 结果如何”的因果链。
        compressed: list[dict[str, Any]] = []
        i = 0
        while i < len(filtered):
            msg = filtered[i]
            
            # Look for tool_call + tool_result pairs to compress
            if (msg.get("role") == "assistant_tool_call" and
                    i + 1 < len(filtered) and
                    filtered[i + 1].get("role") == "tool_result"):
                
                call_msg = msg
                result_msg = filtered[i + 1]
                tool_name = call_msg.get("toolName", "unknown")
                result_msg.get("content", "")
                is_error = result_msg.get("isError", False)
                
                # Step 11: 工具对摘要内容不同，例如编辑保留路径，搜索保留关键词和命中数。
                summary = self._compress_tool_pair(call_msg, result_msg)
                
                # Step 12: 用 assistant 摘要替代两条工具消息，让模型仍能读到历史动作。
                compressed.append({
                    "role": "assistant",
                    "content": summary,
                })
                i += 2  # Skip both messages
            else:
                compressed.append(msg)
                i += 1
        
        current_tokens = estimate_messages_tokens(compressed)
        if current_tokens <= target_tokens:
            # Step 13: 工具对压缩后达标就结束，避免删掉用户意图或最终结论。
            return self._finalize_compaction(
                system_messages, other_messages, compressed, stats, target_tokens
            )
        
        # Step 14: 最后一阶段才按优先级删除旧消息；这说明前几种温和压缩都不够。
        PRIORITY = {
            "user": 0,                    # Highest — encode intent
            "assistant": 1,               # High — encode conclusions + compressed tools
            "assistant_tool_call": 2,     # Medium — should have been compressed in Phase 3
            "tool_result": 3,             # Low — should have been compressed in Phase 3
        }
        
        # Step 15: 最近消息保护起来，因为模型下一步最依赖刚发生的工具结果和用户要求。
        PROTECTED_RECENT = 6
        
        while estimate_messages_tokens(compressed) > target_tokens and len(compressed) > MIN_MESSAGES_TO_KEEP:
            # Step 16: 只在可删除范围里找最低价值消息，避免删到系统提示或最近现场。
            removable_end = max(MIN_MESSAGES_TO_KEEP, len(compressed) - PROTECTED_RECENT)
            best_idx = None
            best_priority = -1
            
            for idx in range(removable_end):
                role = compressed[idx].get("role", "")
                priority = PRIORITY.get(role, 1)
                if priority > best_priority:
                    best_priority = priority
                    best_idx = idx
            
            if best_idx is None:
                break
            
            del compressed[best_idx]
        
        return self._finalize_compaction(
            system_messages, other_messages, compressed, stats, target_tokens
        )
    
    @staticmethod
    def _compress_tool_pair(call_msg: dict[str, Any], result_msg: dict[str, Any]) -> str:
        """Compress a tool_call + tool_result pair into a compact inline summary.
        
        Tool-specific compression strategies:
        - Edit tools: preserve file path and success/failure status
        - Read tools: just note the file was read (content can be re-read)
        - Search tools: preserve the pattern and result count
        - Command tools: preserve command name and exit status
        - Error results: preserve error message (critical for debugging)
        """
        tool_name = call_msg.get("toolName", "unknown")
        inp = call_msg.get("input", {})
        result_content = result_msg.get("content", "")
        is_error = result_msg.get("isError", False)
        
        # Step 1: 错误结果保留错误文本，因为错误很难靠重新运行完全复现。
        if is_error:
            error_text = result_content.strip()[:200].replace("\n", " ")
            return f"[Tool {tool_name} ERROR: {error_text}]"
        
        # Step 2: 编辑工具摘要保留文件路径和修改数量，方便模型知道项目哪些文件被动过。
        if tool_name in _EDIT_TOOLS:
            path = inp.get("path") or inp.get("filePath", "unknown")
            if tool_name == "multi_edit":
                edits = inp.get("edits", [])
                return f"[Edited {path}: {len(edits)} changes applied]"
            return f"[Edited {path}: ok]"
        
        if tool_name in _READ_TOOLS:
            # Step 3: 读文件内容可重读，摘要只需要记录读过哪个文件和大概行数。
            path = inp.get("path") or inp.get("filePath", "")
            if path:
                line_count = result_content.count("\n") + 1
                return f"[Read {path}: {line_count} lines]"
            return f"[{tool_name}: completed]"
        
        if tool_name in _SEARCH_TOOLS:
            # Step 4: 搜索摘要保留关键词和结果数量，让模型知道调查方向。
            pattern = inp.get("pattern") or inp.get("query", "")
            match_lines = [l for l in result_content.split("\n") if l.strip() and not l.startswith("#")]
            return f"[Searched '{pattern[:50]}': {len(match_lines)} results]"
        
        if tool_name in _COMMAND_TOOLS:
            # Step 5: 命令摘要保留命令名和 exit 信息，足够判断验证是否通过。
            cmd = inp.get("command", "")
            cmd_name = cmd.split()[0] if cmd.split() else "command"
            exit_info = ""
            if "exit code" in result_content.lower():
                for line in result_content.split("\n"):
                    if "exit code" in line.lower():
                        exit_info = f" ({line.strip()[:50]})"
                        break
            return f"[Ran {cmd_name}{exit_info}]"
        
        # Step 6: 未分类工具退回通用摘要，至少保留工具名和结果开头。
        brief = result_content.strip()[:100].replace("\n", " ")
        if brief:
            return f"[{tool_name}: {brief}]"
        return f"[{tool_name}: completed]"
    
    def _finalize_compaction(
        self,
        system_messages: list[dict[str, Any]],
        original_other: list[dict[str, Any]],
        filtered: list[dict[str, Any]],
        stats: ContextStats,
        target_tokens: int,
    ) -> list[dict[str, Any]]:
        """Build the final compacted message list with summary marker."""
        # Step 1: 找出被移除的消息，再生成分层摘要，防止历史完全丢失。
        removed_set = set(id(m) for m in filtered)
        removed_messages = [m for m in original_other if id(m) not in removed_set]
        summary_text = _summarize_removed_messages(removed_messages)
        
        removed_count = len(original_other) - len(filtered)
        after_pct = estimate_messages_tokens(filtered) / self.context_window * 100 if self.context_window > 0 else 0
        
        # Step 2: 压缩标记作为 system 消息插入，明确告诉模型“之前的部分被摘要化了”。
        compaction_marker = {
            "role": "system",
            "content": (
                f"[Context compacted at {time.strftime('%H:%M:%S')}. "
                f"{removed_count} messages removed. "
                f"Token usage: {stats.usage_percentage:.0f}% → {after_pct:.0f}%]\n"
                + (f"\nSummary of removed conversation:\n{summary_text}" if summary_text else "")
            ),
        }
        
        # Step 3: 最终顺序是原 system prompt、压缩标记、压缩后的普通消息。
        compacted = system_messages + [compaction_marker] + filtered
        
        # Step 4: 保存压缩历史，/context 或日志可以解释“什么时候压缩过、压掉了多少”。
        self.compaction_history.append({
            "timestamp": time.time(),
            "before_tokens": stats.total_tokens,
            "after_tokens": estimate_messages_tokens(compacted),
            "messages_removed": len(self.messages) - len(compacted),
            "compaction_level": self._compaction_level,
        })
        
        # Step 5: 下次压缩更激进，因为同一会话反复接近上限说明需要更强收缩。
        self._compaction_level = min(self._compaction_level + 1, 3)
        
        self.messages = compacted
        # Step 6: 重建缓存，删掉已经不在上下文里的旧消息 token 记录。
        self._token_cache = {
            id(m): self._token_cache.get(id(m), estimate_message_tokens(m))
            for m in compacted
        }
        return compacted
    
    def get_context_summary(self) -> str:
        """Get a human-readable context usage summary."""
        stats = self.get_stats()
        
        if stats.messages_count == 0:
            return "Context: empty"
        
        status = "✓"
        if stats.is_near_limit:
            status = "⚠"
        if stats.should_compact:
            status = "🔴"
        
        return (
            f"Context: {status} {stats.usage_percentage:.0f}% "
            f"({stats.total_tokens:,}/{stats.context_window:,} tokens, "
            f"{stats.messages_count} msgs, {stats.tool_calls_count} tools)"
        )
    
    def format_context_details(self) -> str:
        """Get detailed context information for /context command."""
        stats = self.get_stats()
        
        lines = [
            "Context Window Usage",
            "=" * 50,
            f"Model: {self.model}",
            f"Context window: {stats.context_window:,} tokens",
            "",
            f"Total tokens: {stats.total_tokens:,}",
            f"Usage: {stats.usage_percentage:.1f}%",
            f"Messages: {stats.messages_count}",
            f"Tool calls: {stats.tool_calls_count}",
            "",
        ]
        
        if stats.should_compact:
            lines.append("⚠️  WARNING: Context is near capacity!")
            lines.append("Auto-compaction will trigger soon.")
            lines.append("")
        
        if self.compaction_history:
            lines.append("Compaction History:")
            for comp in self.compaction_history[-3:]:  # Last 3
                ts = time.strftime("%H:%M:%S", time.localtime(comp["timestamp"]))
                lines.append(
                    f"  {ts}: {comp['messages_removed']} messages removed, "
                    f"{comp['before_tokens']:,} → {comp['after_tokens']:,} tokens"
                )
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_context_state(manager: ContextManager) -> None:
    """Save context manager state to disk."""
    state_path = MINI_CODE_DIR / "context_state.json"
    MINI_CODE_DIR.mkdir(parents=True, exist_ok=True)
    
    state = {
        "model": manager.model,
        "context_window": manager.context_window,
        "messages": manager.messages,
        "compaction_history": manager.compaction_history[-10:],  # Keep last 10
        "_compaction_level": manager._compaction_level,  # Save compaction level
    }
    
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_context_state() -> ContextManager | None:
    """Load context manager state from disk."""
    state_path = MINI_CODE_DIR / "context_state.json"
    if not state_path.exists():
        return None
    
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manager = ContextManager(
            model=state.get("model", "default"),
            context_window=state.get("context_window", 0),
            messages=state.get("messages", []),
            compaction_history=state.get("compaction_history", []),
        )
        # Restore compaction level if saved
        if "_compaction_level" in state:
            manager._compaction_level = state["_compaction_level"]
        return manager
    except (json.JSONDecodeError, KeyError):
        return None


def clear_context_state() -> None:
    """Clear saved context state."""
    state_path = MINI_CODE_DIR / "context_state.json"
    if state_path.exists():
        state_path.unlink()
