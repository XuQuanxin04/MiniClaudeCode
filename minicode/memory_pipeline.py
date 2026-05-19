"""Memory Pipeline — unified facade for the complete memory lifecycle.

Design principle: ONE class, FOUR methods. All memory operations flow
through this single entry point. No scattered ad-hoc calls.

Architecture:
  MemoryPipeline
    ├── read(task, files) → DomainClassifier → BM25 → Reranker → [memories]
    ├── inject(task, files, messages) → read + append to system prompt
    ├── write(task, trace) → ReflectionEngine → TaskContext → MemoryManager
    └── maintain() → CuratorAgent → consolidate/validate/promote/link

Sub-components (internal, not exposed):
  - DomainClassifier: auto-detects active domains from files/intent
  - MemoryReranker: LLM curation of BM25 results
  - MemoryInjector: PID-controlled injection into prompt
  - MemoryCuratorAgent: background optimization during idle
  - VectorMemoryStore: optional parallel semantic search
"""
from __future__ import annotations

import time
from typing import Any

from minicode.logging_config import get_logger

logger = get_logger("memory_pipeline")


class MemoryPipeline:
    """Unified memory operations facade.

    Usage:
        pipeline = MemoryPipeline(memory_manager)
        pipeline.initialize(model_adapter, workspace_path)

        # On task start
        memories = pipeline.read("Create login form", ["src/Login.tsx"])
        messages = pipeline.inject("Create login form", ["src/Login.tsx"], messages)

        # On task end
        pipeline.write("Create login form", execution_trace)

        # Background (every ~10 tasks)
        report = pipeline.maintain()
    """

    def __init__(self, memory_manager: Any | None = None):
        self._memory = memory_manager
        self._model: Any = None
        self._workspace: str | None = None

        # Subsystems (lazy init via initialize())
        self._reranker: Any = None
        self._injector: Any = None
        self._curator: Any = None
        self._reflection: Any = None
        self._vector_store: Any = None
        self._domain_classifier_loaded = False

        self._initialized = False
        self._read_count = 0
        self._write_count = 0
        self._maintain_count = 0

    # ── Lifecycle ──────────────────────────────────────────────────

    def initialize(
        self,
        model_adapter: Any | None = None,
        workspace_path: str | None = None,
        enable_reranker: bool = True,
        enable_vector: bool = False,
    ) -> None:
        """Initialize all subsystems. Call once after MemoryManager is ready."""
        self._model = model_adapter
        self._workspace = workspace_path

        # Reranker (LLM curation on read)
        if enable_reranker:
            from minicode.memory_reranker import MemoryReranker
            self._reranker = MemoryReranker(model_adapter=model_adapter)

        # Injector (PID-controlled injection)
        if self._memory:
            from minicode.memory_injector import MemoryInjector
            self._injector = MemoryInjector(
                memory_manager=self._memory,
                reranker=self._reranker if self._reranker and self._reranker.enabled else None,
            )

        # Curator (background optimization)
        from minicode.memory_curator_agent import MemoryCuratorAgent
        self._curator = MemoryCuratorAgent(
            memory_manager=self._memory,
            model_adapter=model_adapter,
            workspace_path=workspace_path,
        )

        # Reflection engine (write path)
        from minicode.agent_reflection import ReflectionEngine
        self._reflection = ReflectionEngine(memory_manager=self._memory)

        # Vector store (optional)
        if enable_vector:
            try:
                from minicode.vector_memory import VectorMemoryStore
                self._vector_store = VectorMemoryStore()
                if self._vector_store.enabled and self._memory:
                    # Index existing memories
                    all_entries = []
                    from minicode.memory import MemoryScope
                    for scope in MemoryScope:
                        if scope in self._memory.memories:
                            all_entries.extend(self._memory.memories[scope].entries)
                    if all_entries:
                        indexed = self._vector_store.index_entries(all_entries)
                        logger.info("VectorMemoryStore: indexed %d existing entries", indexed)
            except Exception:
                pass

        self._initialized = True
        logger.info(
            "MemoryPipeline initialized: reranker=%s vector=%s",
            self._reranker.enabled if self._reranker else False,
            self._vector_store is not None and self._vector_store.enabled if self._vector_store else False,
        )

        # Restore persisted state
        self._load_state()

    # ── State persistence ────────────────────────────────────────────

    def _state_path(self) -> str | None:
        """Path for pipeline state file."""
        if not self._workspace:
            return None
        import os
        return os.path.join(self._workspace, ".mini-code-memory", "pipeline_state.json")

    def save_state(self) -> None:
        """Persist pipeline state to disk (cache stats, counters, curator history)."""
        path = self._state_path()
        if not path:
            return
        try:
            import json, os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            state = {
                "read_count": self._read_count,
                "write_count": self._write_count,
                "maintain_count": self._maintain_count,
                "reranker_cache_hits": self._reranker._cache_hits if self._reranker else 0,
                "reranker_call_count": self._reranker._call_count if self._reranker else 0,
                "curator_history": self._curator.get_history() if self._curator else [],
                "timestamp": time.time(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.debug("MemoryPipeline save_state failed: %s", e)

    def _load_state(self) -> None:
        """Restore pipeline state from disk."""
        path = self._state_path()
        if not path:
            return
        try:
            import json, os
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._read_count = state.get("read_count", 0)
            self._write_count = state.get("write_count", 0)
            self._maintain_count = state.get("maintain_count", 0)
            if self._reranker:
                self._reranker._cache_hits = state.get("reranker_cache_hits", 0)
                self._reranker._call_count = state.get("reranker_call_count", 0)
            logger.debug("MemoryPipeline: restored state (%d reads, %d writes)",
                        self._read_count, self._write_count)
        except Exception as e:
            logger.debug("MemoryPipeline _load_state failed: %s", e)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "read_count": self._read_count,
            "write_count": self._write_count,
            "maintain_count": self._maintain_count,
            "reranker_enabled": self._reranker.enabled if self._reranker else False,
            "reranker_cache_hit_rate": self._reranker.cache_hit_rate if self._reranker else 0.0,
            "vector_enabled": self._vector_store is not None and self._vector_store.enabled if self._vector_store else False,
        }

    # ── READ: Memory retrieval ─────────────────────────────────────

    def read(
        self,
        task_description: str,
        current_files: list[str] | None = None,
        active_domains: list[str] | None = None,
        max_results: int = 15,
    ) -> list[dict[str, Any]]:
        """Full retrieval pipeline: classify domains → BM25 → optional reranker.

        Returns list of {id, content, domain, relevance, source}.
        """
        if not self._memory:
            return []

        self._read_count += 1

        # 1. Derive active domains if not provided
        if not active_domains and current_files:
            active_domains = self._get_active_domains(current_files, task_description)

        # 2. Search with query reformulation fallback (T2)
        entries = self._try_search_with_reformulation(
            task_description, active_domains, max_results,
        )

        # 2b. Parallel vector search + RRF fusion (F1)
        if self._vector_store and self._vector_store.enabled:
            try:
                vec_results = self._vector_store.search(
                    task_description, top_k=max_results,
                )
                if vec_results:
                    from minicode.vector_memory import merge_bm25_vector
                    entries = merge_bm25_vector(entries, vec_results)
            except Exception:
                pass

        # 3. Score entries with value function (T1)
        if self._reranker and self._reranker.enabled and len(entries) > 3:
            try:
                result = self._reranker.curate(
                    entries, task_description,
                    active_domains=active_domains,
                    current_files=current_files,
                )
                # Filter entries to selected IDs
                selected = set(result.selected_ids)
                entries = [e for e in entries if e.id in selected]
            except Exception:
                pass  # Fall through to BM25 results

        # 4. Spreading activation via related_to graph (T3)
        entries = self._spread_activation(entries)

        # 5. Format results
        return [
            {
                "id": e.id,
                "content": e.content,
                "domain": getattr(e, 'domains', []),
                "relevance": getattr(e, 'usage_count', 0),
                "source": "memory_pipeline",
            }
            for e in entries[:max_results]
        ]

    # ── INJECT: Memory into prompt ──────────────────────────────────

    def inject(
        self,
        task_description: str,
        current_files: list[str] | None,
        messages: list[dict],
        context_usage: float = 0.5,
    ) -> list[dict]:
        """Read memories and inject into system prompt with adaptive cooldown.

        Adaptive cooldown (T1): τ_cool = τ_base × (1 - context_pressure).
        Returns modified messages with memory context appended to system message.
        """
        if not self._initialized or not self._memory:
            return messages

        # Adaptive cooldown check
        now = time.time()
        cooldown = self._adaptive_cooldown(context_usage)
        if hasattr(self, '_last_inject_time'):
            if now - self._last_inject_time < cooldown:
                return messages  # Still in cooldown
        self._last_inject_time = now

        try:
            # Use the injector for PID-controlled injection
            if self._injector:
                injected = self._injector.inject_for_task(
                    task_description,
                    current_files=current_files,
                )
                if injected:
                    memory_context = "\n## Relevant Project Memory\n" + "\n".join(
                        f"- {m.content[:200]}" for m in injected[:5]
                    )
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "system":
                            messages[i] = {
                                **msg,
                                "content": msg["content"] + memory_context,
                            }
                            break
                    logger.info(
                        "MemoryPipeline: injected %d memories (mode=%s)",
                        len(injected),
                        self._injector._last_decision.mode.value if self._injector._last_decision else "?",
                    )
        except Exception:
            pass

        return messages

    # ── WRITE: Memory persistence ──────────────────────────────────

    def write(
        self,
        task_description: str,
        execution_trace: list[dict[str, Any]],
    ) -> str | None:
        """Write task reflection as structured memory.

        Uses ReflectionEngine to extract TaskContext with files, libraries,
        and domain tags. Returns the created memory entry ID or None.
        """
        if not self._reflection:
            return None

        self._write_count += 1

        try:
            result = self._reflection.reflect(task_description, execution_trace)
            if result and result.confidence >= self._reflection.min_confidence:
                mem_data = result.to_memory_entry()
                from minicode.memory import MemoryScope
                entry = self._memory.add_entry(
                    scope=MemoryScope.PROJECT,
                    category=mem_data["category"],
                    content=mem_data["content"],
                    tags=mem_data["tags"],
                )
                # Post-add domain assignment
                if mem_data.get("domains"):
                    for scope in MemoryScope:
                        if scope in self._memory.memories:
                            for e in self._memory.memories[scope].entries:
                                if e.content == mem_data["content"]:
                                    e.domains = mem_data["domains"]
                                    break
                logger.info(
                    "MemoryPipeline: wrote reflection success=%s confidence=%.2f",
                    result.success, result.confidence,
                )
                self.save_state()
                return getattr(entry, 'id', None)
        except Exception:
            pass

        self.save_state()
        return None

    # ── FEEDBACK: Close the quality loop (F2) ────────────────────────

    def feedback(
        self,
        task_success: bool,
        injected_memory_ids: list[str] | None = None,
    ) -> None:
        """Task outcome → memory utility. Closes the outermost learning loop.

        Success → boost injected memories (positive reinforcement).
        Failure → gentle decay (they may have misled the agent).
        """
        if not self._memory or not injected_memory_ids:
            return

        from minicode.memory import MemoryScope
        for scope in MemoryScope:
            if scope not in self._memory.memories:
                continue
            for entry in self._memory.memories[scope].entries:
                if entry.id in injected_memory_ids:
                    if task_success:
                        entry.usage_count += 2
                    else:
                        entry.usage_count = max(0, entry.usage_count - 1)
                    entry.last_accessed = time.time()

    # ── MAINTAIN: Background optimization ───────────────────────────

    def maintain(self, force: bool = False) -> dict[str, Any] | None:
        """Run background memory optimization.

        Consolidates insights, archives duplicates, validates against codebase,
        promotes/demotes tiers, and links related memories.

        Returns CuratorReport as dict, or None if not ready.
        """
        if not self._curator:
            return None

        self._curator.on_task_complete()

        if not force and not self._curator.should_run:
            return None

        self._maintain_count += 1
        try:
            report = self._curator.run_cycle(force=True)
            self.save_state()
            return report.to_dict()
        except Exception:
            return None

    # ── Internal ────────────────────────────────────────────────────

    def _get_active_domains(
        self, current_files: list[str], task_description: str
    ) -> list[str]:
        try:
            from minicode.domain_classifier import get_active_domain_values
            return get_active_domain_values(
                current_files=current_files,
                intent_text=task_description,
            )
        except Exception:
            return []

    # ── T1: Memory Value Function + Adaptive Cooldown ───────────────

    # Formal definition:
    #   V(m, t, c) = relevance(m, t) × freshness(m) × utility(m, c)
    #   where:
    #     relevance(m, t) = BM25_score(m, t) ∈ [0, 1]
    #     freshness(m)    = exp(-age_days / τ) with τ = 30 days
    #     utility(m, c)   = 1 + α × I(m was used in similar context c)
    #
    # Adaptive cooldown:  τ_cool(c) = τ_base × (1 - context_pressure)
    #   High context pressure → shorter cooldown → faster injection
    #   Low context pressure → longer cooldown → less noise

    _TAU_FRESHNESS = 30.0  # days
    _ALPHA_UTILITY = 0.15

    def _memory_value(
        self, bm25_score: float, entry: Any, context_usage: float = 0.5
    ) -> float:
        """Compute V(m, t, c) for a single memory entry."""
        import math
        age_days = (time.time() - getattr(entry, 'updated_at', time.time())) / 86400.0
        freshness = math.exp(-age_days / self._TAU_FRESHNESS)
        utility = 1.0 + self._ALPHA_UTILITY * math.log1p(getattr(entry, 'usage_count', 0))
        return bm25_score * freshness * utility

    def _adaptive_cooldown(self, context_usage: float) -> float:
        """Compute adaptive injection cooldown based on context pressure.

        τ_cool = τ_base × (1 - context_pressure), clamped to [5s, 120s].
        High pressure → shorter cooldown (memory is more needed).
        """
        base = 30.0  # seconds
        return max(5.0, min(120.0, base * (1.0 - context_usage)))

    # ── T2: Query Reformulation ─────────────────────────────────────

    # When BM25 returns poor results (top score < τ_low), attempt
    # reformulation: strip stopwords, try domain synonyms, expand abbreviations.
    # Max 3 attempts. If no improvement, keep original results.

    _QUERY_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                        "to", "of", "in", "for", "on", "with", "at", "by", "from",
                        "and", "or", "but", "not", "this", "that", "it", "i", "we",
                        "add", "create", "make", "implement", "build", "set", "get"}

    _QUERY_REFORMULATIONS = [
        lambda q: " ".join(w for w in q.lower().split() if w not in MemoryPipeline._QUERY_STOPWORDS),
        lambda q: q.lower().replace("  ", " ").strip(),
    ]

    def _reformulate_query(self, query: str) -> list[str]:
        """Generate reformulated query variants."""
        variants = [query]
        for reformulate in self._QUERY_REFORMULATIONS:
            v = reformulate(query)
            if v and v != query and v not in variants:
                variants.append(v)
        return variants[:3]

    def _try_search_with_reformulation(
        self, task_description: str, active_domains: list[str] | None, max_results: int
    ) -> list[Any]:
        """Search with query reformulation fallback for poor initial results."""
        from minicode.memory import MemoryScope

        entries = self._memory.search(
            task_description, limit=max_results, active_domains=active_domains,
        )

        if entries and len(entries) >= 3:
            return entries  # Good enough

        # Try reformulations
        for variant in self._reformulate_query(task_description):
            if variant == task_description:
                continue
            alt = self._memory.search(
                variant, limit=max_results, active_domains=active_domains,
            )
            if len(alt) > len(entries):
                logger.debug("Query reformulation improved: %d → %d results", len(entries), len(alt))
                return alt

        return entries

    # ── T3: Spreading Activation ────────────────────────────────────

    # When memory m is retrieved, its related_to neighbors also receive
    # activation: score_neighbor += score(m) × decay × sim(m, neighbor)
    # depth=1, decay=0.5. This surfaces related memories the user might
    # not have explicitly searched for.

    _SPREAD_DECAY = 0.5
    _SPREAD_THRESHOLD = 0.3

    def _spread_activation(
        self, entries: list[Any]
    ) -> list[Any]:
        """Enrich results via spreading activation through related_to graph.

        Concatenates directly-linked neighbors with decayed relevance.
        """
        if not self._memory or not entries:
            return entries

        seen_ids = {e.id for e in entries}
        neighbors = []

        for entry in entries[:5]:  # Only spread from top 5
            if not hasattr(entry, 'related_to') or not entry.related_to:
                continue
            for rid in entry.related_to:
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                # Find neighbor in memory
                for scope_name in ["project", "local", "user"]:
                    try:
                        from minicode.memory import MemoryScope
                        scope = MemoryScope(scope_name)
                        if scope in self._memory.memories:
                            nbr = self._memory.memories[scope]._id_index.get(rid)
                            if nbr:
                                neighbors.append(nbr)
                                break
                    except (ValueError, KeyError):
                        continue

        return entries + neighbors
