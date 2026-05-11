"""Cost and usage tracking for API calls.

Tracks token usage, API costs, and code changes across the session.
Inspired by Claude Code's cost-tracker.ts implementation.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

# ---------------------------------------------------------------------------
# Pricing (approximate, per 1M tokens)
# ---------------------------------------------------------------------------

MODEL_PRICING = {
    # Anthropic models (USD per 1M tokens)
    "claude-sonnet-4-20250514": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-opus-4-20250514": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
    "claude-haiku-3-20240307": {
        "input": 0.25,
        "output": 1.25,
        "cache_read": 0.03,
        "cache_write": 0.30,
    },
    # OpenAI models
    "gpt-4o": {
        "input": 2.50,
        "output": 10.0,
        "cache_read": 1.25,
        "cache_write": 2.50,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
        "cache_read": 0.08,
        "cache_write": 0.15,
    },
    "gpt-4-turbo": {
        "input": 10.0,
        "output": 30.0,
        "cache_read": 5.0,
        "cache_write": 10.0,
    },
    "o1": {
        "input": 15.0,
        "output": 60.0,
        "cache_read": 7.50,
        "cache_write": 15.0,
    },
    "o1-mini": {
        "input": 3.0,
        "output": 12.0,
        "cache_read": 1.50,
        "cache_write": 3.0,
    },
    "o3-mini": {
        "input": 1.10,
        "output": 4.40,
        "cache_read": 0.55,
        "cache_write": 1.10,
    },
    # OpenRouter models (pricing via OpenRouter, approximate)
    "openrouter/auto": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "anthropic/claude-sonnet-4": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "anthropic/claude-opus-4": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
    "openai/gpt-4o": {
        "input": 2.50,
        "output": 10.0,
        "cache_read": 1.25,
        "cache_write": 2.50,
    },
    "openai/gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
        "cache_read": 0.08,
        "cache_write": 0.15,
    },
    "google/gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.0,
        "cache_read": 0.63,
        "cache_write": 1.25,
    },
    "google/gemini-2.5-flash": {
        "input": 0.15,
        "output": 0.60,
        "cache_read": 0.08,
        "cache_write": 0.15,
    },
    "meta-llama/llama-4-maverick": {
        "input": 0.20,
        "output": 0.60,
        "cache_read": 0.10,
        "cache_write": 0.20,
    },
    "deepseek/deepseek-r1": {
        "input": 0.55,
        "output": 2.19,
        "cache_read": 0.14,
        "cache_write": 0.55,
    },
    "deepseek/deepseek-chat": {
        "input": 0.14,
        "output": 0.28,
        "cache_read": 0.07,
        "cache_write": 0.14,
    },
    "qwen/qwen3-235b-a22b": {
        "input": 0.22,
        "output": 0.88,
        "cache_read": 0.11,
        "cache_write": 0.22,
    },
    "minimax/minimax-m1": {
        "input": 0.20,
        "output": 0.80,
        "cache_read": 0.10,
        "cache_write": 0.20,
    },
    # Default fallback
    "default": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
}


@functools.lru_cache(maxsize=128)
def _get_pricing(model: str) -> dict[str, float]:
    """Cached pricing lookup to avoid repeated dict.get() calls."""
    return MODEL_PRICING.get(model, MODEL_PRICING["default"])


# ---------------------------------------------------------------------------
# Cost calculation (standalone function for use outside CostTracker)
# ---------------------------------------------------------------------------

def calculate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Calculate cost for a single API call.
    
    Args:
        model: Model name
        input_tokens: Input token count
        output_tokens: Output token count
        cache_read_tokens: Cache read token count
        cache_creation_tokens: Cache write token count
    
    Returns:
        Cost in USD
    """
    pricing = _get_pricing(model)
    return (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
        + (cache_read_tokens / 1_000_000) * pricing["cache_read"]
        + (cache_creation_tokens / 1_000_000) * pricing["cache_write"]
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelUsage:
    """Usage statistics for a specific model."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0
    total_duration_ms: int = 0
    error_count: int = 0
    
    def avg_duration_ms(self) -> float:
        """Average duration per call."""
        if self.call_count == 0:
            return 0.0
        return self.total_duration_ms / self.call_count
    
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens


@dataclass
class CostTracker:
    """Tracks API costs and usage across the session.
    
    Inspired by Claude Code's cost-tracker.ts
    """
    # Global totals
    total_cost_usd: float = 0.0
    total_api_duration_ms: int = 0
    total_lines_added: int = 0
    total_lines_removed: int = 0
    total_lines_modified: int = 0
    
    # Per-model usage
    model_usage: dict[str, ModelUsage] = field(default_factory=dict)
    
    # Session info
    session_start: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Record API usage.
        
        Args:
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            duration_ms: API call duration
            cache_read_tokens: Cache read token count
            cache_write_tokens: Cache write token count
        
        Returns:
            Calculated cost in USD
        """
        # Get pricing (cached for repeated lookups)
        pricing = _get_pricing(model)
        
        # Calculate cost
        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
            + (cache_read_tokens / 1_000_000) * pricing["cache_read"]
            + (cache_write_tokens / 1_000_000) * pricing["cache_write"]
        )
        
        # Update model usage
        if model not in self.model_usage:
            self.model_usage[model] = ModelUsage()
        
        usage = self.model_usage[model]
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.cache_read_tokens += cache_read_tokens
        usage.cache_write_tokens += cache_write_tokens
        usage.cost_usd += cost
        usage.call_count += 1
        usage.total_duration_ms += duration_ms
        
        # Update totals
        self.total_cost_usd += cost
        self.total_api_duration_ms += duration_ms
        self.last_updated = time.time()
        
        return cost
    
    def record_error(self, model: str) -> None:
        """Record an API error.
        
        Args:
            model: Model name
        """
        if model not in self.model_usage:
            self.model_usage[model] = ModelUsage()
        self.model_usage[model].error_count += 1
    
    def record_code_changes(
        self,
        lines_added: int = 0,
        lines_removed: int = 0,
        lines_modified: int = 0,
    ) -> None:
        """Record code changes from edits.
        
        Args:
            lines_added: Lines added
            lines_removed: Lines removed
            lines_modified: Lines modified
        """
        self.total_lines_added += lines_added
        self.total_lines_removed += lines_removed
        self.total_lines_modified += lines_modified
    
    def get_model_usage(self, model: str) -> ModelUsage:
        """Get usage for a specific model."""
        return self.model_usage.get(model, ModelUsage())
    
    def get_total_tokens(self) -> int:
        """Get total tokens across all models."""
        return sum(u.total_tokens() for u in self.model_usage.values())
    
    def get_total_calls(self) -> int:
        """Get total API calls."""
        return sum(u.call_count for u in self.model_usage.values())
    
    def get_total_errors(self) -> int:
        """Get total errors."""
        return sum(u.error_count for u in self.model_usage.values())
    
    def format_cost_report(self, detailed: bool = False) -> str:
        """Format a human-readable cost report.
        
        Args:
            detailed: Include per-model breakdown
        
        Returns:
            Formatted report string
        """
        lines = [
            "Cost Report",
            "===========",
            f"Total cost: ${self.total_cost_usd:.4f}",
            f"Total tokens: {self.get_total_tokens():,}",
            f"Total calls: {self.get_total_calls()}",
            f"Total errors: {self.get_total_errors()}",
            f"Session duration: {int(time.time() - self.session_start)}s",
        ]
        
        if detailed and self.model_usage:
            lines.append("")
            lines.append("Per-model breakdown:")
            for model, usage in sorted(self.model_usage.items()):
                lines.append(
                    f"  {model}: ${usage.cost_usd:.4f} ({usage.total_tokens():,} tokens, "
                    f"{usage.call_count} calls, {usage.error_count} errors)"
                )
        
        return "\n".join(lines)
    
    def format_short_summary(self) -> str:
        """Format a one-line summary.
        
        Returns:
            Short summary string
        """
        total = self.get_total_tokens()
        calls = self.get_total_calls()
        return f"${self.total_cost_usd:.4f} | {total:,} tokens | {calls} calls"
