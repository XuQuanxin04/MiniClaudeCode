from __future__ import annotations

import concurrent.futures
import inspect
import time
from typing import Any, Callable

from minicode.context_manager import ContextManager, estimate_message_tokens
from minicode.logging_config import get_logger
from minicode.permissions import PermissionManager
from minicode.state import Store, AppState, increment_tool_calls, set_busy, set_idle
from minicode.tooling import ToolContext, ToolRegistry, ToolResult
from minicode.types import AgentStep, ChatMessage, ModelAdapter

# Hooks integration
from minicode.hooks import HookEvent, fire_hook_sync

# Intelligence integration
from minicode.agent_metrics import AgentMetricsCollector
from minicode.agent_intelligence import ErrorClassifier, NudgeGenerator, ToolScheduler
from minicode.working_memory import protect_context

# Work chain integration
from minicode.intent_parser import parse_intent
from minicode.task_object import build_task, TaskObject, TaskState
from minicode.pipeline_engine import get_pipeline_engine
from minicode.capability_registry import get_registry, CapabilityDomain
from minicode.layered_context import ContextBuilder, LayeredContext
from minicode.decision_audit import get_auditor, DecisionOutcome

# 工程控制论集成
from minicode.cybernetic_orchestrator import CyberneticOrchestrator
from minicode.cybernetic_supervisor import CyberneticSupervisor, save_supervisor_report
from minicode.feedforward_controller import FeedforwardController

# 高级控制论模块
from minicode.adaptive_pid_tuner import AdaptivePIDTuner
from minicode.state_observer import StateObserver, MeasurementVector
from minicode.decoupling_controller import DecouplingController
from minicode.predictive_controller import PredictiveController
from minicode.self_healing_engine import SelfHealingEngine

# 任务进度控制
from minicode.progress_controller import ProgressController, ProgressSignal, ProgressAction

# 记忆注入和模型选择控制
from minicode.memory_injector import MemoryInjectionController, MemoryInjectionSignal, MemoryInjector
from minicode.model_registry import ModelSelectionController, ModelSelectionSignal

# 智能路由与自省 (Phase 3 导入)
from minicode.smart_router import SmartRouter, TaskOutcome
from minicode.agent_reflection import ReflectionEngine
from minicode.model_switcher import ModelSwitcher

# 上下文管理集成 (Claude Code-style + Engineering Cybernetics)
from minicode.context_compactor import (
    ContextCompactor,
    AutoCompactConfig,
)
from minicode.context_cybernetics import ContextCyberneticsOrchestrator
from minicode.cost_control import CostControlLoop
from minicode.memory import MemoryManager

logger = get_logger("agent_loop")

# 甯搁噺锛氶伩鍏嶉噸澶嶇殑鎻愮ず鏂囨湰
NUDGE_CONTINUE = (
    "Continue immediately from your <progress> update with concrete tool calls, "
    "code changes, or an explicit <final> answer only if the task is complete. "
    "Prefer taking the next concrete action over explaining what you plan to do."
)

NUDGE_AFTER_TOOL_RESULT = (
    "You have received tool results. Review them briefly, then take the next "
    "concrete action: call another tool, edit code, or give an explicit <final> "
    "answer only if the task is truly complete. Do not restate what you just saw."
)

NUDGE_AFTER_EMPTY_RESPONSE = (
    "Your last response was empty. This often happens after tool errors or when "
    "the model is uncertain. Pick the most likely next action and try it — you can "
    "adjust based on results. Call a tool, edit code, or give <final> if done."
)

NUDGE_AFTER_EMPTY_NO_TOOLS = (
    "Your last response was empty but you have not used any tools yet. Start by "
    "inspecting the relevant files (read_file, grep_files, list_files) to understand "
    "the codebase before making changes."
)

RESUME_AFTER_PAUSE = (
    "Resume from the previous pause. Continue with the next concrete tool call, "
    "code change, or <final> answer."
)

RESUME_AFTER_MAX_TOKENS = (
    "Your previous response was cut short by the token limit. Resume immediately "
    "with the next concrete action — pick up where you left off."
)


def _is_empty_assistant_response(content: str) -> bool:
    return len(content.strip()) == 0


def _extract_task_description(messages: list[ChatMessage]) -> str:
    """Extract the original task description from messages."""
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            content = str(msg["content"])
            if not content.startswith("Continue") and not content.startswith("Your last"):
                return content[:500]
    return "Unknown task"


def _build_work_chain_task(messages: list[ChatMessage]) -> tuple[TaskObject | None, dict]:
    """Build TaskObject from conversation messages and return it with metadata."""
    raw_input = _extract_task_description(messages)
    if raw_input == "Unknown task":
        return None, {}
    intent = parse_intent(raw_input)
    task = build_task(intent, raw_input)
    metadata = {
        "intent_type": intent.intent_type.value,
        "action_type": intent.action_type.value,
        "confidence": intent.confidence,
        "entities": intent.entities,
        "complexity": intent.complexity_hint,
    }
    logger.info(
        "Work chain: intent=%s action=%s confidence=%.2f complexity=%s",
        intent.intent_type.value, intent.action_type.value,
        intent.confidence, intent.complexity_hint,
    )
    return task, metadata


def _build_layered_context(
    messages: list[ChatMessage],
    system_prompt: str = "",
    project_context: str = "",
    task: TaskObject | None = None,
) -> tuple[LayeredContext, ContextBuilder]:
    """Build layered context from conversation and task."""
    context = LayeredContext()
    builder = ContextBuilder(context)
    if system_prompt:
        builder.set_system_prompt(system_prompt)
    if project_context:
        builder.add_project_memory(project_context)
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            builder.add_session_message(role, content)
    if task:
        scratchpad = (
            f"Task: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Constraints: {len(task.constraints)}\n"
            f"Expected outputs: {len(task.expected_outputs)}"
        )
        builder.add_scratchpad(scratchpad)
    return context, builder


def _register_tool_capabilities(tools: ToolRegistry) -> None:
    """Register existing tools as capabilities in the registry."""
    registry = get_registry()
    if registry.list_all():
        return
    for tool_name in tools.list_all():
        try:
            from minicode.capability_registry import CapabilityMetadata, CapabilityScope
            tool_def = tools.find(tool_name)
            if not tool_def:
                continue
            domain = CapabilityDomain.UNKNOWN
            if "file" in tool_name or "write" in tool_name or "read" in tool_name:
                domain = CapabilityDomain.FILE
            elif "search" in tool_name or "grep" in tool_name:
                domain = CapabilityDomain.SEARCH
            elif "web" in tool_name or "http" in tool_name or "fetch" in tool_name:
                domain = CapabilityDomain.WEB
            elif "command" in tool_name or "run" in tool_name or "exec" in tool_name:
                domain = CapabilityDomain.EXECUTION
            elif "code" in tool_name or "diff" in tool_name or "review" in tool_name:
                domain = CapabilityDomain.CODE
            elif "memory" in tool_name:
                domain = CapabilityDomain.MEMORY
            scope = CapabilityScope.READONLY
            if any(k in tool_name for k in ("write", "modify", "edit", "delete", "create")):
                scope = CapabilityScope.WRITE
            if any(k in tool_name for k in ("command", "exec", "run")):
                scope = CapabilityScope.DESTRUCTIVE
            if any(k in tool_name for k in ("web", "fetch", "http")):
                scope = CapabilityScope.EXTERNAL
            metadata = CapabilityMetadata(
                name=tool_name, domain=domain, scope=scope,
                description=tool_def.description or f"Tool: {tool_name}",
                tags=["tool", tool_name],
            )
            registry.register(metadata, lambda **kw: tools.execute(tool_name, kw, ToolContext()), None)
        except Exception as e:
            logger.debug("Failed to register tool %s as capability: %s", tool_name, e)


def _execute_single_tool(
    call: dict,
    tools: ToolRegistry,
    cwd: str,
    permissions: Any | None,
    runtime: dict | None,
    store: Any | None,
    step: int,
    on_tool_start: Callable[[str, dict], None] | None,
    on_tool_result: Callable[[str, str, bool], None] | None,
) -> ToolResult:
    """Execute a single tool call with hooks, state updates, and crash protection.
    
    Used both for serial execution and as a worker function for concurrent execution.
    When running concurrently (store/on_tool_start/on_tool_result are None),
    hooks and UI callbacks are deferred to the result processing phase.
    
    Includes a global exception safety net: any unexpected crash in the tool
    execution pipeline (hooks, state updates, etc.) is caught and converted
    to an error ToolResult, preventing the entire agent loop from crashing.
    """
    tool_name = call["toolName"]
    tool_input = call["input"]
    
    try:
        # Pre-tool hooks and UI (only for serial execution)
        if on_tool_start:
            on_tool_start(tool_name, tool_input)
        
        if store:
            store.set_state(set_busy(tool_name))
        
        # Execute the tool with timeout protection
        import concurrent.futures
        import os
        TOOL_TIMEOUT = int(os.environ.get("MINICODE_TOOL_TIMEOUT", "120"))
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    tools.execute,
                    tool_name, tool_input,
                    ToolContext(cwd=cwd, permissions=permissions, _runtime=runtime),
                )
                result = future.result(timeout=TOOL_TIMEOUT)
        except concurrent.futures.TimeoutError:
            result = ToolResult(
                ok=False,
                output=f"Tool '{tool_name}' timed out after {TOOL_TIMEOUT}s",
            )
        except Exception:
            result = tools.execute(
                tool_name, tool_input,
                ToolContext(cwd=cwd, permissions=permissions, _runtime=runtime),
            )  # Fallback: direct execution
        
        # Post-tool state updates (only for serial execution)
        if store:
            store.set_state(increment_tool_calls())
            store.set_state(set_idle())
        
        if on_tool_result:
            on_tool_result(tool_name, result.output, not result.ok)
        
        return result
    
    except (KeyboardInterrupt, SystemExit):
        # Always propagate these
        raise
    except Exception as exc:  # noqa: BLE001
        # Global safety net: catch ANY unexpected error in the tool execution
        # pipeline (hooks, state updates, permission checks, etc.) and convert
        # it to an error result. This prevents a single tool crash from
        # cascading into a full session failure.
        import traceback
        tb_excerpt = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]).strip()
        error_type = type(exc).__name__
        
        logger.error("Tool execution pipeline crashed (%s): %s", error_type, exc)
        
        # Ensure state is reset even on crash
        if store:
            try:
                store.set_state(set_idle())
            except Exception:
                pass
        
        return ToolResult(
            ok=False,
            output=f"[{error_type}] Tool execution pipeline crashed: {exc}\n"
                   f"Traceback:\n{tb_excerpt}"
        )


def _format_diagnostics(stop_reason: str | None, block_types: list[str] | None, ignored_block_types: list[str] | None) -> str:
    parts: list[str] = []
    if stop_reason:
        parts.append(f"stop_reason={stop_reason}")
    if block_types:
        parts.append(f"blocks={','.join(block_types)}")
    if ignored_block_types:
        parts.append(f"ignored={','.join(ignored_block_types)}")
    return f" Diagnostics: {'; '.join(parts)}." if parts else ""


def _is_recoverable_thinking_stop(*, is_empty: bool, stop_reason: str | None, ignored_block_types: list[str] | None) -> bool:
    if not is_empty:
        return False
    if stop_reason not in {"pause_turn", "max_tokens"}:
        return False
    return "thinking" in (ignored_block_types or [])


def _should_treat_assistant_as_progress(*, kind: str | None, content: str, saw_tool_result: bool) -> bool:
    if kind == "progress":
        return True
    if kind == "final":
        return False
    if not saw_tool_result:
        return False
    return False


def _model_next(
    model: ModelAdapter,
    messages: list[ChatMessage],
    *,
    on_stream_chunk: Callable[[str], None] | None,
    store: Store[AppState] | None,
) -> AgentStep:
    """Call provider adapters with store support while preserving test doubles."""
    if store is None:
        return model.next(messages, on_stream_chunk=on_stream_chunk)

    try:
        signature = inspect.signature(model.next)
    except (TypeError, ValueError):
        logger.warning("_model_next: inspect.signature failed, falling back without store")
        return model.next(messages, on_stream_chunk=on_stream_chunk, store=store)

    supports_store = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == "store"
        for parameter in signature.parameters.values()
    )
    if supports_store:
        return model.next(messages, on_stream_chunk=on_stream_chunk, store=store)
    return model.next(messages, on_stream_chunk=on_stream_chunk)


def _apply_control_signal(
    *,
    control_signal: Any,
    system_state: Any,
    max_steps: int | None,
    tool_scheduler: ToolScheduler,
    context_compactor: ContextCompactor | None,
    model_switcher: Any | None,
) -> int | None:
    """Apply FeedbackController output to live runtime knobs."""
    if not control_signal or control_signal.confidence <= 0.6:
        return max_steps

    if (
        control_signal.limit_max_steps
        and max_steps is not None
        and control_signal.limit_max_steps < max_steps
    ):
        logger.info(
            "FeedbackController: limiting max_steps %d -> %d",
            max_steps, control_signal.limit_max_steps,
        )
        max_steps = control_signal.limit_max_steps

    if control_signal.adjust_token_budget != 1.0:
        if (
            context_compactor
            and hasattr(context_compactor, "_tool_budget")
            and context_compactor._tool_budget
        ):
            new_budget = max(
                1000,
                int(
                    context_compactor._tool_budget.budget_per_message
                    * control_signal.adjust_token_budget
                ),
            )
            context_compactor._tool_budget.budget_per_message = new_budget
            logger.info(
                "FeedbackController: token budget adjusted to %d (mult=%.2f)",
                new_budget, control_signal.adjust_token_budget,
            )

    if control_signal.reduce_parallelism:
        tool_scheduler._force_max_workers = min(
            getattr(tool_scheduler, "_force_max_workers", 2) or 2,
            2,
        )
        logger.info(
            "FeedbackController: reduce_parallelism -> max_workers=2 "
            "(oscillation=%.2f)",
            control_signal.oscillation_index,
        )

    if control_signal.adjust_concurrency != 0:
        cap = max(1, 4 + control_signal.adjust_concurrency)
        tool_scheduler._force_max_workers = cap
        logger.info(
            "FeedbackController: adjust_concurrency=%+d -> max_workers=%d",
            control_signal.adjust_concurrency, cap,
        )

    if control_signal.increase_model_level:
        logger.info(
            "FeedbackController: model upgrade recommended (errors=%.2f perf=%.2f)",
            system_state.error_frequency,
            system_state.performance_score(),
        )
        if model_switcher:
            model_switcher._pending_upgrade = True

    if control_signal.decrease_model_level:
        logger.info(
            "FeedbackController: model downgrade recommended (efficiency=%.2f)",
            system_state.token_efficiency,
        )

    if control_signal.suggest_memory_persistence:
        logger.info("FeedbackController: persisting working memory")
        if context_compactor and hasattr(context_compactor, "_tool_budget"):
            try:
                context_compactor._tool_budget.flush()
            except Exception:
                pass

    if control_signal.recommend_skill_update:
        logger.info(
            "FeedbackController: skill update recommended (pattern=%.2f)",
            system_state.pattern_reuse_rate,
        )

    return max_steps


def run_agent_turn(
    *,
    model: ModelAdapter,
    tools: ToolRegistry,
    messages: list[ChatMessage],
    cwd: str,
    permissions: PermissionManager | None = None,
    store: Store[AppState] | None = None,
    max_steps: int = 50,
    on_tool_start: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str, bool], None] | None = None,
    on_assistant_message: Callable[[str], None] | None = None,
    on_progress_message: Callable[[str], None] | None = None,
    on_assistant_stream_chunk: Callable[[str], None] | None = None,
    context_manager: ContextManager | None = None,
    runtime: dict | None = None,
    metrics_collector: AgentMetricsCollector | None = None,
    system_prompt: str = "",
    project_context: str = "",
    enable_work_chain: bool = True,
) -> list[ChatMessage]:
    current_messages = list(messages)
    saw_tool_result = False
    empty_response_retry_count = 0
    recoverable_thinking_retry_count = 0
    tool_error_count = 0
    step = 0

    tool_scheduler = ToolScheduler(metrics_collector=metrics_collector)

    # Initialize work chain if enabled
    task: TaskObject | None = None
    task_metadata: dict = {}
    layered_context: LayeredContext | None = None
    context_builder: ContextBuilder | None = None
    auditor = get_auditor() if enable_work_chain else None

    # 工程控制论控制器初始化（通过 Orchestrator 统一管理）
    orch: CyberneticOrchestrator | None = None
    feedback_controller: Any = None
    feedforward_controller: Any = None
    stability_monitor: Any = None
    cybernetic_supervisor: Any = None

    adaptive_pid_tuner: Any = None
    state_observer: Any = None
    decoupling_controller: Any = None
    predictive_controller: Any = None
    self_healing_engine: Any = None
    progress_controller: Any = None
    memory_injection_ctrl: Any = None
    model_selection_ctrl: Any = None
    smart_router: Any = None
    reflection_engine: Any = None
    model_switcher: Any = None
    memory_injector: Any = None

    if enable_work_chain:
        task, task_metadata = _build_work_chain_task(current_messages)
        layered_context, context_builder = _build_layered_context(
            current_messages, system_prompt, project_context, task,
        )
        get_pipeline_engine()
        _register_tool_capabilities(tools)

        # 初始化所有工程控制论控制器（通过 Orchestrator 统一管理）
        orch = CyberneticOrchestrator()
        orch.initialize(model, tools, runtime)
        feedback_controller = orch.feedback
        cybernetic_supervisor = orch.cyber_supervisor
        stability_monitor = orch.stability
        adaptive_pid_tuner = orch.adaptive_tuner
        state_observer = orch.state_observer
        decoupling_controller = orch.decoupling
        predictive_controller = orch.predictive
        progress_controller = orch.progress
        memory_injection_ctrl = orch.memory_ctrl
        model_selection_ctrl = orch.model_ctrl
        smart_router = orch.smart_router
        reflection_engine = orch.reflection
        model_switcher = orch.model_switcher
        logger.info("CyberneticOrchestrator: %d controllers initialized", 15)
        if smart_router and task:
            try:
                current_model_id = model.model_id if hasattr(model, 'model_id') else ""
                task_text = task.raw_input if hasattr(task, 'raw_input') else str(current_messages[-1].get('content', ''))
                routing, switch_result = smart_router.route_and_switch(
                    task_text,
                    current_model=current_model_id,
                )
                logger.info(
                    "SmartRouter: model=%s tier=%s cost=$%.4f reason=%s",
                    routing.selected_model, routing.tier_name,
                    routing.estimated_cost, routing.reasoning[:80],
                )
                # 如果路由推荐了不同模型且切换成功，更新 model 引用
                if switch_result and switch_result.success:
                    model = switch_result.adapter
                    logger.info(
                        "SmartRouter: switched model %s -> %s",
                        switch_result.old_model, switch_result.new_model,
                    )
            except Exception:
                pass

        # 初始化前馈控制器（预判式优化）
        if task:
            feedforward_controller = FeedforwardController()
            preemptive_config = feedforward_controller.preconfigure(task.parsed_intent, task.raw_input)
            risk_assessment = feedforward_controller.assess_risks(task.parsed_intent, preemptive_config)
            logger.info(
                "Feedforward control: config=%s risk=%s",
                preemptive_config.recommended_model, risk_assessment.risk_level,
            )
            # Apply feedforward preemptive config to execution parameters
            if preemptive_config.confidence > 0.6:
                max_steps = min(max_steps, preemptive_config.max_turn_steps)
                logger.info(
                    "Feedforward: max_steps=%d model=%s timeout=%.1fs",
                    preemptive_config.max_turn_steps,
                    preemptive_config.recommended_model,
                    preemptive_config.tool_timeout_seconds,
                )
            if risk_assessment.risk_level in ("high", "critical"):
                logger.warning(
                    "Feedforward risk assessment: level=%s probability=%.2f risks=%s",
                    risk_assessment.risk_level,
                    risk_assessment.estimated_failure_probability,
                    ", ".join(risk_assessment.identified_risks[:3]),
                )

        # 模型选择控制器：根据任务特征推荐模型
        if model_selection_ctrl and task:
            try:
                model_signal = ModelSelectionSignal(
                    task_complexity=getattr(task, 'complexity', 'moderate') if hasattr(task, 'complexity') else "moderate",
                    budget_pressure=0.3,
                    latency_pressure=0.3,
                    recent_failures=0,
                    current_model=model.model_id if hasattr(model, 'model_id') else "",
                )
                model_decision = model_selection_ctrl.decide(model_signal)
                logger.info(
                    "ModelSelectionController: model=%s score=%.2f effort=%s reasons=%s",
                    model_decision.model, model_decision.score,
                    model_decision.reasoning_effort.value,
                    ", ".join(model_decision.reasons),
                )
            except Exception:
                pass

        # 初始化上下文管理器 (Claude Code-style + Engineering Cybernetics)
        # 必须在 SelfHealingEngine 之前初始化，因为自愈引擎需要委托压缩操作
        context_compactor: ContextCompactor | None = None
        context_cybernetics: ContextCyberneticsOrchestrator | None = None
        memory_mgr: MemoryManager | None = None
        if context_manager:
            compact_config = AutoCompactConfig(
                threshold_ratio=0.85,
                circuit_breaker_limit=3,
                session_memory_enabled=True,
            )
            memory_mgr = MemoryManager(project_root=cwd)
            # 将 memory_mgr 注入 ReflectionEngine，使自省经验持久化
            if reflection_engine:
                reflection_engine.memory = memory_mgr
            # 初始化 MemoryInjector，将控制论决策落地为实际记忆注入
            # 同时创建 Reranker（使用真实 LLM 做记忆策展）
            memory_reranker = None
            try:
                from minicode.memory_reranker import MemoryReranker
                # Use the agent's model for reranking (lightweight prompt, ~500 tokens)
                memory_reranker = MemoryReranker(model_adapter=model)
            except Exception:
                pass
            memory_injector = MemoryInjector(
                memory_manager=memory_mgr,
                controller=memory_injection_ctrl,
                reranker=memory_reranker,
            )
            if orch:
                orch._last_model = model
                orch._workspace = cwd
                orch.wire_memory(memory_mgr)
                if orch.memory_pipeline is not None:
                    memory_injector = getattr(orch.memory_pipeline, "_injector", memory_injector)
            # 记忆注入控制器：根据上下文压力决定注入策略
            if memory_injection_ctrl:
                try:
                    inj_signal = MemoryInjectionSignal(
                        context_usage=context_manager.get_stats().usage_percentage / 100.0,
                        retrieval_quality=0.5,
                        recent_failure=False,
                    )
                    inj_decision = memory_injection_ctrl.decide(
                        inj_signal,
                        base_max_memories=5,
                        base_min_relevance=0.3,
                        base_max_tokens=200,
                    )
                    logger.info(
                        "MemoryInjectionController: mode=%s max_mem=%d min_rel=%.2f max_tok=%d",
                        inj_decision.mode.value, inj_decision.max_memories,
                        inj_decision.min_relevance, inj_decision.max_tokens_per_memory,
                    )
                except Exception:
                    pass
            # 执行实际记忆注入：将相关记忆注入到系统 prompt 中
            if orch and task:
                try:
                    task_desc = task.raw_input if hasattr(task, 'raw_input') else ""
                    current_messages = orch.inject_memories(task_desc, current_messages)
                except Exception:
                    pass
            elif memory_injector and task:
                try:
                    task_desc = task.raw_input if hasattr(task, 'raw_input') else ""
                    injected = memory_injector.inject_for_task(task_desc)
                    if injected:
                        logger.info(
                            "MemoryInjector: injected %d memories (mode=%s)",
                            len(injected),
                            memory_injector._last_decision.mode.value if memory_injector._last_decision else "?",
                        )
                        # 将注入的记忆追加到系统 prompt
                        memory_context = "\n## Injected Memory\n" + "\n".join(
                            f"- {m.content[:200]}" for m in injected[:5]
                        )
                        for i, msg in enumerate(current_messages):
                            if msg.get("role") == "system":
                                current_messages[i] = {
                                    **msg,
                                    "content": msg["content"] + memory_context,
                                }
                                break
                except Exception:
                    pass
            context_compactor = ContextCompactor(
                context_window=context_manager.context_window,
                workspace=cwd,
                memory_manager=memory_mgr,
                estimate_fn=estimate_message_tokens,
                config=compact_config,
            )
            context_cybernetics = ContextCyberneticsOrchestrator(
                context_compactor,
                kp=2.0, ki=0.15, kd=0.3,
                pid_setpoint=0.70,
                base_threshold=0.85,
                safety_margin_turns=3,
                enabled=True,
            )
            if task and hasattr(task, 'parsed_intent') and task.parsed_intent:
                context_cybernetics.set_intent(str(task.parsed_intent.intent_type))
            logger.info("ContextCybernetics initialized: PID control loop + predictive guard")
            if orch:
                orch.context_compactor = context_compactor
                orch.context_cybernetics = context_cybernetics

        # 初始化自愈引擎（接收 cybernetics 引用用于 CONTEXT_OVERFLOW 委托）
        if orch:
            orch.wire_healing(tool_scheduler, context_compactor)
            self_healing_engine = orch.healing
        else:
            self_healing_engine = SelfHealingEngine(
                orchestrator=context_cybernetics,
                tool_scheduler=tool_scheduler,
                compactor=context_compactor,
            )
        logger.info("Self-healing engine initialized: automated recovery + compaction delegation")

        # 初始化成本控制闭环 (CostTracker → PID → ToolResultBudgetManager)
        cost_control = orch.cost_control if orch else None
        if cost_control is None:
            cost_control = CostControlLoop(
                target_cost_per_min=0.50,
                kp=1.5, ki=0.08, kd=0.2,
                enabled=True,
            )
        if orch:
            orch.cost_control = cost_control
        logger.info("CostControlLoop initialized: BudgetPIDController for cost regulation")

    # 检查上下文状态 + 运行 Claude Code-style 预请求优化管线
    if context_manager:
        context_manager.messages = current_messages
        stats = context_manager.get_stats()
        logger.info("Context: %d tokens (%.0f%%), %d messages",
                   stats.total_tokens, stats.usage_percentage, stats.messages_count)

        # 运行控制论闭环优化管线 (Sense → Predict → Control → Act → Learn)
        if context_cybernetics:
            if cost_control:
                est_cost = stats.total_tokens * 0.000015
                adj = cost_control.run(
                    cost_usd=est_cost,
                    total_tokens=stats.total_tokens,
                    total_calls=max(step, 1),
                )
                if context_compactor and hasattr(context_compactor, '_tool_budget') and context_compactor._tool_budget:
                    cost_control.apply_to_budget_manager(context_compactor._tool_budget)
                elif adj and adj.budget_multiplier < 0.8:
                    logger.warning(
                        "CostControl: budget tightened (mult=%.2f reason=%s) but no compactor active",
                        adj.budget_multiplier, adj.reason,
                    )

            cyber_messages, cyber_result, cyber_action = context_cybernetics.run_cycle(
                current_messages,
                error_rate=float(tool_error_count) / max(step, 1) if step > 0 else 0.0,
                avg_latency=step * 2.0,
                turn_id=step,
            )
            if cyber_result and cyber_result.effective:
                current_messages = cyber_messages
                context_manager.messages = current_messages
                logger.info(
                    "Cybernetics[%s]: %s intensity=%.2f freed=%d tokens [%s]",
                    cyber_action.reason if cyber_action else "unknown",
                    cyber_result.strategy.value,
                    cyber_action.compaction_intensity if cyber_action else 0,
                    cyber_result.tokens_freed,
                    cyber_result.summary_text[:80] if cyber_result.summary_text else "",
                )
        elif context_compactor:
            compaction_result = context_compactor.process_request(current_messages)
            if compaction_result.effective:
                current_messages = compaction_result.messages
                context_manager.messages = current_messages
                logger.info(
                    "ContextCompactor: %s freed %d tokens [%s]",
                    compaction_result.strategy.value,
                    compaction_result.tokens_freed,
                    compaction_result.summary_text[:80],
                )
        elif context_manager.should_auto_compact():
            logger.warning("Context near limit, auto-compacting...")
            current_messages = context_manager.compact_messages()
            if on_assistant_message:
                on_assistant_message(context_manager.get_context_summary())

    try:
        while max_steps is None or step < max_steps:
            step += 1

            # Hook: agent turn started
            fire_hook_sync(HookEvent.AGENT_START, step=step, cwd=cwd)

            # 高级控制论闭环（每个 step 开始时执行）
            if enable_work_chain and orch:
                orch.step_start(
                    context_manager=context_manager,
                    step=step,
                    tool_error_count=tool_error_count,
                    saw_tool_result=saw_tool_result,
                )
            elif enable_work_chain:
                # 状态观测：通过可测量输出估计系统内部状态
                if state_observer:
                    measurement = MeasurementVector(
                        timestamp=time.time(),
                        response_time=step * 2.0,  # 估算响应时间
                        success_rate=1.0 - (tool_error_count / max(step, 1)),
                        context_length=context_manager.get_stats().total_tokens if context_manager else 0,
                        error_count=tool_error_count,
                        tool_calls=0,
                    )
                    observed_state = state_observer.update(measurement)

                    # 将 Kalman 估计值输入到控制器
                    if observed_state.confidence > 0.4:
                        if observed_state.internal_load > 0.8:
                            logger.info(
                                "StateObserver: high internal_load=%.2f, reduce concurrency",
                                observed_state.internal_load,
                            )
                        if observed_state.hidden_errors > 0.5 and self_healing_engine:
                            self_healing_engine.detect_and_heal({
                                "error_rate": observed_state.hidden_errors * 5.0,
                                "context_usage": observed_state.context_pressure,
                            })
                        if observed_state.system_degradation > 0.4:
                            logger.warning(
                                "StateObserver: system degradation=%.2f confidence=%.2f",
                                observed_state.system_degradation,
                                observed_state.confidence,
                            )

                # 预测控制：预测未来趋势并提前调整
                if predictive_controller:
                    if context_manager:
                        stats = context_manager.get_stats()
                        predictive_controller.update("context_usage", stats.usage_percentage / 100.0)
                    predictive_controller.update("error_rate", tool_error_count / max(step, 1))

                    if step > 2:
                        actions = predictive_controller.generate_predictive_actions()
                        if actions and actions[0].urgency > 0.7:
                            action = actions[0]
                            logger.info(
                                "Predictive action: %s urgency=%.2f horizon=%s",
                                action.recommended_action, action.urgency,
                                getattr(action, 'horizon', 'unknown'),
                            )
                            # Execute predictive actions via dispatch
                            dispatch: dict[str, Callable[[], None]] = {
                                "trigger_compaction": lambda: (
                                    context_cybernetics.try_reactive_recover(current_messages, "predictive")
                                    if context_cybernetics else None
                                ),
                                "enable_safe_mode": lambda: logger.info(
                                    "Predictive: safe_mode recommended (reduce concurrency, extend timeouts)"
                                ),
                                "reduce_concurrency": lambda: logger.info(
                                    "Predictive: reduce_concurrency recommended"
                                ),
                            }
                            handler = dispatch.get(action.recommended_action)
                            if handler:
                                try:
                                    handler()
                                except Exception as exc:
                                    logger.warning(
                                        "Predictive action %s failed: %s",
                                        action.recommended_action, exc,
                                    )
                            # Also run self-healing for corroboration
                            if self_healing_engine:
                                healing_actions = self_healing_engine.detect_and_heal({
                                    "context_usage": stats.usage_percentage / 100.0 if context_manager else 0.0,
                                    "error_rate": tool_error_count / max(step, 1),
                                })
                                if healing_actions:
                                    logger.info("Self-healing: %s", healing_actions[0].strategy)

            if metrics_collector:
                metrics_collector.start_turn(step)

            next_step: AgentStep
            try:
                next_step = _model_next(
                    model,
                    current_messages,
                    on_stream_chunk=on_assistant_stream_chunk,
                    store=store,
                )
            except KeyboardInterrupt:
                raise  # Let Ctrl-C propagate
            except ConnectionError as error:
                fallback = f"Network error (connection failed or dropped): {error}"
                logger.error("Model API connection error: %s", error)
                if on_assistant_message:
                    on_assistant_message(fallback)
                current_messages.append({"role": "assistant", "content": fallback})
                if metrics_collector:
                    metrics_collector.end_turn(total_tokens=0)
                return current_messages
            except TimeoutError as error:
                fallback = f"Model API timeout: {error}"
                logger.error("Model API timeout: %s", error)
                if on_assistant_message:
                    on_assistant_message(fallback)
                current_messages.append({"role": "assistant", "content": fallback})
                if metrics_collector:
                    metrics_collector.end_turn(total_tokens=0)
                return current_messages
            except Exception as error:
                # Catch-all for unexpected errors (rate limit, auth, server 5xx, etc.)
                error_type = type(error).__name__
                fallback = f"Model API error ({error_type}): {error}"
                logger.error("Model API error (%s): %s", error_type, error)

                # Reactive Compact: 控制论恢复路径
                error_str = str(error).lower()
                needs_recovery = "prompt" in error_str and ("too long" in error_str or "exceeds" in error_str)
                if context_cybernetics and needs_recovery:
                    recovered_messages, recovery_result = context_cybernetics.try_reactive_recover(current_messages, error_str)
                    if recovery_result and recovery_result.effective:
                        current_messages = recovered_messages
                        if context_manager:
                            context_manager.messages = current_messages
                        logger.info(
                            "Cybernetics Reactive recovered: freed %d tokens",
                            recovery_result.tokens_freed,
                        )
                        continue
                elif context_compactor and needs_recovery:
                    recovery_result = context_compactor.reactive_recover(current_messages, error_str)
                    if recovery_result and recovery_result.effective:
                        current_messages = recovery_result.messages
                        if context_manager:
                            context_manager.messages = current_messages
                        logger.info(
                            "Reactive Compact recovered: freed %d tokens",
                            recovery_result.tokens_freed,
                        )
                        continue

                # ModelSwitcher: 尝试切换到备用模型并重试
                if model_switcher and "rate" not in error_str:
                    try:
                        switch_result = model_switcher.switch_to(
                            "",  # Let switcher pick fallback
                            reason=f"{error_type}: {error_str[:80]}",
                        )
                        if switch_result.success and switch_result.adapter is not None:
                            model = switch_result.adapter
                            logger.info(
                                "ModelSwitcher: switched to %s, retrying with new adapter",
                                switch_result.new_model,
                            )
                            continue
                    except Exception:
                        pass

                if on_assistant_message:
                    on_assistant_message(fallback)
                current_messages.append({"role": "assistant", "content": fallback})
                if metrics_collector:
                    metrics_collector.end_turn(total_tokens=0)
                return current_messages

            if next_step.type == "assistant":
                is_empty = _is_empty_assistant_response(next_step.content)
                if not is_empty and _should_treat_assistant_as_progress(
                    kind=getattr(next_step, 'kind', None),
                    content=next_step.content,
                    saw_tool_result=saw_tool_result,
                ):
                    if on_progress_message:
                        on_progress_message(next_step.content)
                    current_messages.append({"role": "assistant_progress", "content": next_step.content})
                    current_messages.append(
                        {
                            "role": "user",
                            "content": (
                                NUDGE_AFTER_TOOL_RESULT
                                if saw_tool_result and getattr(next_step, 'kind', None) != "progress"
                                else NUDGE_CONTINUE
                            ),
                        }
                    )
                    continue

                diagnostics = next_step.diagnostics

                if _is_recoverable_thinking_stop(
                    is_empty=is_empty,
                    stop_reason=diagnostics.stopReason if diagnostics else None,
                    ignored_block_types=diagnostics.ignoredBlockTypes if diagnostics else None,
                ) and recoverable_thinking_retry_count < 3:
                    recoverable_thinking_retry_count += 1
                    stop_reason = diagnostics.stopReason if diagnostics else None
                    progress_content = (
                        "Model hit max_tokens during thinking; requesting the next step."
                        if stop_reason == "max_tokens"
                        else "Model returned pause_turn; requesting the next step."
                    )
                    if on_progress_message:
                        on_progress_message(progress_content)
                    current_messages.append({"role": "assistant_progress", "content": progress_content})
                    current_messages.append(
                        {
                            "role": "user",
                            "content": (
                                RESUME_AFTER_PAUSE
                                if stop_reason == "pause_turn"
                                else RESUME_AFTER_MAX_TOKENS
                            ),
                        }
                    )
                    continue

                if is_empty and empty_response_retry_count < 2:
                    empty_response_retry_count += 1
                    current_messages.append(
                        {
                            "role": "user",
                            "content": (
                                NUDGE_AFTER_EMPTY_RESPONSE
                                if saw_tool_result
                                else NUDGE_AFTER_EMPTY_NO_TOOLS
                            ),
                        }
                    )
                    continue

                if is_empty:
                    diagnostics_suffix = _format_diagnostics(
                        diagnostics.stopReason if diagnostics else None,
                        diagnostics.blockTypes if diagnostics else None,
                        diagnostics.ignoredBlockTypes if diagnostics else None,
                    )
                    if saw_tool_result:
                        fallback = (
                            f"Model returned an empty response after tool execution and the turn was stopped. There were {tool_error_count} tool error(s); retry, adjust the command, or choose a different approach.{diagnostics_suffix}"
                            if tool_error_count > 0
                            else f"Model returned an empty response after tool execution and the turn was stopped. Retry or ask the model to continue the remaining steps.{diagnostics_suffix}"
                        )
                    else:
                        fallback = f"Model returned an empty response and the turn was stopped.{diagnostics_suffix}"
                    if on_assistant_message:
                        on_assistant_message(fallback)
                    current_messages.append({"role": "assistant", "content": fallback})
                    return current_messages

                if on_assistant_message:
                    on_assistant_message(next_step.content)
                current_messages.append({"role": "assistant", "content": next_step.content})
                # Protect final answer in working memory
                protect_context(
                    content=next_step.content[:500],
                    entry_type="key_decision",
                    ttl_seconds=3600,
                )
                return current_messages

            if next_step.content:
                role = "assistant_progress" if next_step.contentKind == "progress" else "assistant"
                if role == "assistant_progress":
                    if on_progress_message:
                        on_progress_message(next_step.content)
                    current_messages.append({"role": role, "content": next_step.content})
                    current_messages.append(
                        {
                            "role": "user",
                            "content": NUDGE_CONTINUE,
                        }
                    )
                else:
                    if on_assistant_message:
                        on_assistant_message(next_step.content)
                    current_messages.append({"role": role, "content": next_step.content})

            if not next_step.calls and next_step.content and next_step.contentKind != "progress":
                return current_messages

            # --- Concurrent tool execution ---
            # Classify calls into concurrent-safe (read-only) vs serial (writes/commands)
            calls = next_step.calls
            _results: list[tuple[dict, ToolResult]] = []

            if len(calls) <= 1:
                # Single call — no benefit from concurrency, run directly
                call = calls[0]
                if metrics_collector:
                    metrics_collector.start_tool(call["toolName"])
                result = _execute_single_tool(
                    call, tools, cwd, permissions, runtime, store, step,
                    on_tool_start, on_tool_result,
                )
                if metrics_collector:
                    metrics_collector.end_tool(
                        success=result.ok,
                        error=result.output if not result.ok else "",
                    )
                _results.append((call, result))
            else:
                # Multiple calls — use ToolScheduler for intelligent partitioning
                concurrent_calls, serial_calls = tool_scheduler.schedule_calls(calls, tools)

                _results.clear()  # Reuse outer declaration

                # Phase 1: Run all concurrent-safe tools in parallel
                if concurrent_calls:
                    max_workers = tool_scheduler.get_recommended_max_workers(
                        concurrent_calls,
                        error_rate=tool_error_count / max(step, 1),
                        avg_latency=step * 2.0,
                        recent_failures=tool_error_count,
                    )
                    # Apply cybernetic concurrency cap if FeedbackController reduced parallelism
                    force_cap = getattr(tool_scheduler, '_force_max_workers', None)
                    if force_cap:
                        max_workers = min(max_workers, force_cap)
                    if tool_scheduler.last_decision:
                        logger.info(
                            "ToolSchedulerController: workers=%d multiplier=%.2f cooldown=%.2fs [%s]",
                            max_workers,
                            tool_scheduler.last_decision.concurrency_multiplier,
                            tool_scheduler.last_decision.cooldown_seconds,
                            ", ".join(tool_scheduler.last_decision.reasons or []),
                        )
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_workers,
                        thread_name_prefix="mc-tool",
                    ) as pool:
                        future_to_call = {
                            pool.submit(
                                _execute_single_tool,
                                call, tools, cwd, permissions, runtime, None, step,
                                None, None,  # No UI callbacks during concurrent phase
                            ): call
                            for call in concurrent_calls
                        }
                        for future in concurrent.futures.as_completed(future_to_call):
                            call = future_to_call[future]
                            try:
                                result = future.result()
                            except Exception as exc:
                                result = ToolResult(ok=False, output=f"Concurrent execution error: {exc}")
                            _results.append((call, result))

                # Phase 2: Run serial tools sequentially (in original order)
                if serial_calls:
                    for call in serial_calls:
                        if metrics_collector:
                            metrics_collector.start_tool(call["toolName"])
                        result = _execute_single_tool(
                            call, tools, cwd, permissions, runtime, store, step,
                            on_tool_start, on_tool_result,
                        )
                        if metrics_collector:
                            metrics_collector.end_tool(
                                success=result.ok,
                                error=result.output if not result.ok else "",
                            )
                        _results.append((call, result))
                        # If a serial tool awaits user, return immediately
                        if result.awaitUser:
                            # Still need to process remaining results for messages
                            break
            
            # Process all results and build messages (preserve original call order)
            call_order = {call["id"]: idx for idx, call in enumerate(calls)}
            _results.sort(key=lambda pair: call_order.get(pair[0]["id"], 999))
            
            for call, result in _results:
                # Fire hooks and UI callbacks for concurrent calls (deferred)
                tool_def = tools.find(call["toolName"])
                is_concurrent = tool_def and tool_def.is_concurrency_safe and len(calls) > 1
                
                if is_concurrent:
                    # Deferred UI callbacks for concurrent tools
                    if on_tool_start:
                        on_tool_start(call["toolName"], call["input"])
                    if store:
                        store.set_state(set_busy(call["toolName"]))
                        store.set_state(increment_tool_calls())
                        store.set_state(set_idle())
                    # Hook: pre-tool-use (fire after the fact for concurrent tools)
                    fire_hook_sync(
                        HookEvent.PRE_TOOL_USE,
                        tool_name=call["toolName"],
                        tool_input=call["input"],
                        step=step,
                    )
                
                # Hook: post-tool-use
                fire_hook_sync(
                    HookEvent.POST_TOOL_USE,
                    tool_name=call["toolName"],
                    tool_output=result.output,
                    is_error=not result.ok,
                    step=step,
                )
                
                if is_concurrent:
                    if on_tool_result:
                        on_tool_result(call["toolName"], result.output, not result.ok)
                
                saw_tool_result = True
                if not result.ok:
                    tool_error_count += 1
                    # Use ErrorClassifier for intelligent error handling
                    classified = ErrorClassifier.classify(result.output, tool_name=call["toolName"])
                    nudge = NudgeGenerator.generate(classified, retry_count=tool_error_count)
                    # Append nudge to tool result content for model context
                    result_output = result.output + "\n\n[System note: " + nudge + "]"
                else:
                    result_output = result.output

                # Record conflicts between concurrent tools if both failed
                if not result.ok and len(calls) > 1:
                    for other_call, other_result in _results:
                        if other_call["id"] == call["id"]:
                            continue
                        if not other_result.ok:
                            tool_scheduler.record_conflict(call["toolName"], other_call["toolName"])

                # ReadDedup: 去重相同文件的重复读取，节省上下文空间
                if (
                    context_compactor
                    and result.ok
                    and call.get("toolName") == "read_file"
                ):
                    file_path = call.get("input", {}).get("path", "")
                    if file_path:
                        dedup_mgr = context_compactor.read_dedup
                        if dedup_mgr.should_dedup(file_path, result_output):
                            result_output = dedup_mgr.get_stub(file_path)
                            logger.debug("ReadDedup replaced content for %s (stub)", file_path)
                        dedup_mgr.register_read(file_path, result_output, len(current_messages))

                current_messages.append(
                    {
                        "role": "assistant_tool_call",
                        "toolUseId": call["id"],
                        "toolName": call["toolName"],
                        "input": call["input"],
                    }
                )
                current_messages.append(
                    {
                        "role": "tool_result",
                        "toolUseId": call["id"],
                        "toolName": call["toolName"],
                        "content": result_output,
                        "isError": not result.ok,
                    }
                )
                if result.awaitUser:
                    if on_assistant_message:
                        on_assistant_message(result_output)
                    current_messages.append({"role": "assistant", "content": result_output})
                    if metrics_collector:
                        metrics_collector.end_turn(total_tokens=0)
                    return current_messages

            # 工具执行完成后的控制论反馈
            if enable_work_chain:
                # 多变量解耦：消除工具间的耦合影响
                if decoupling_controller:
                    decoupling_controller.record_measurement({
                        "token_usage_to_latency": (
                            context_manager.get_stats().usage_percentage / 100.0 if context_manager else 0.0,
                            step * 2.0 / 60.0,
                        ),
                        "context_pressure_to_errors": (
                            context_manager.get_stats().usage_percentage / 100.0 if context_manager else 0.0,
                            tool_error_count / max(step, 1),
                        ),
                    })
                    decoupling_controller.compute_decoupling_matrix()

                if orch:
                    step_summary = orch.step_end(
                        tool_scheduler=tool_scheduler,
                        context_manager=context_manager,
                        step=step,
                        tool_error_count=tool_error_count,
                        saw_tool_result=saw_tool_result,
                        max_steps=max_steps,
                    )
                    max_steps = _apply_control_signal(
                        control_signal=step_summary.get("control_signal"),
                        system_state=step_summary.get("system_state"),
                        max_steps=max_steps,
                        tool_scheduler=tool_scheduler,
                        context_compactor=context_compactor,
                        model_switcher=model_switcher,
                    )
                else:
                    # 自愈检测：检测并修复故障
                    if self_healing_engine:
                        metrics_for_healing = {
                            "error_rate": tool_error_count / max(step, 1),
                            "context_usage": context_manager.get_stats().usage_percentage / 100.0 if context_manager else 0.0,
                            "oscillation_index": feedback_controller._compute_oscillation() if feedback_controller else 0.0,
                        }
                        healing_actions = self_healing_engine.detect_and_heal(metrics_for_healing)
                        if healing_actions:
                            logger.info("Self-healing triggered: %s", healing_actions[0].strategy)

                    # 进度控制：检测任务是否卡住或完成
                    if progress_controller:
                        progress_signal = ProgressSignal(
                            total_steps=max_steps,
                            completed_steps=step - tool_error_count,
                            failed_steps=tool_error_count,
                            tool_calls=step,
                            tool_errors=tool_error_count,
                            output_changed=saw_tool_result,
                            elapsed_seconds=step * 2.0,
                            max_steps=max_steps,
                        )
                        progress_decision = progress_controller.decide(progress_signal)
                        if progress_decision.action in (ProgressAction.STOP, ProgressAction.REQUEST_CONFIRMATION):
                            logger.warning(
                                "ProgressController: action=%s health=%.2f stall=%.2f reasons=%s",
                                progress_decision.action.value,
                                progress_decision.health_score,
                                progress_decision.stall_score,
                                ", ".join(progress_decision.reasons),
                            )

            # Tool execution completed for this step; ask the model for the next turn
            # instead of falling through to the max-step fallback.
            if metrics_collector:
                total_tokens = sum(
                    estimate_message_tokens(m) for m in current_messages
                ) if context_manager else 0
                metrics_collector.end_turn(total_tokens=total_tokens)
            continue

        fallback = "Reached the maximum tool step limit for this turn."
        if on_assistant_message:
            on_assistant_message(fallback)
        current_messages.append({"role": "assistant", "content": fallback})
        return current_messages
    finally:
        fire_hook_sync(HookEvent.AGENT_STOP, step=step, tool_errors=tool_error_count)

        if metrics_collector and metrics_collector._current_turn is not None:
            total_tokens = sum(
                estimate_message_tokens(m) for m in current_messages
            ) if context_manager else 0
            metrics_collector.end_turn(total_tokens=total_tokens)

        if enable_work_chain and task:
            final_state = TaskState.COMPLETED if tool_error_count == 0 else TaskState.FAILED
            task.set_state(final_state)
            task.result_summary = f"Turn completed: {step} steps, {tool_error_count} errors"

            if auditor:
                outcome = DecisionOutcome.SUCCESS if tool_error_count == 0 else DecisionOutcome.FAILURE
                auditor.complete_decision(
                    outcome,
                    step * 100.0,
                    task.result_summary,
                    task.error_message if tool_error_count > 0 else "",
                )

            logger.info(
                "Work chain completed: task=%s state=%s steps=%d errors=%d",
                task.id, task.state.value, step, tool_error_count,
            )

            # 任务后自省：提取经验教训
            if orch and task:
                try:
                    execution_trace: list[dict[str, Any]] = [
                        {"type": "tool_call", "count": step},
                        {"type": "error", "count": tool_error_count, "content": f"{tool_error_count} errors"} if tool_error_count > 0 else {},
                        {"type": "assistant", "steps": step},
                    ]
                    orch.reflect_on_task(
                        task_description=task.raw_input if hasattr(task, 'raw_input') else str(task.id),
                        step=step,
                        tool_error_count=tool_error_count,
                        execution_trace=execution_trace,
                    )
                except Exception:
                    pass
            elif reflection_engine and task:
                try:
                    execution_trace: list[dict[str, Any]] = [
                        {"type": "tool_call", "count": step},
                        {"type": "error", "count": tool_error_count, "content": f"{tool_error_count} errors"} if tool_error_count > 0 else {},
                        {"type": "assistant", "steps": step},
                    ]
                    reflection = reflection_engine.reflect(
                        task_description=task.raw_input if hasattr(task, 'raw_input') else str(task.id),
                        execution_trace=execution_trace,
                    )
                    logger.info(
                        "AgentReflection: success=%s confidence=%.2f lessons=%d improvements=%d",
                        reflection.success, reflection.confidence,
                        len(reflection.lessons_learned), len(reflection.suggested_improvements),
                    )
                except Exception:
                    pass

            # 记忆质量反馈：任务成功→注入的记忆 usage_count+1
            if memory_injector and hasattr(memory_injector, '_cached_result'):
                try:
                    from minicode.memory import MemoryScope
                    for mem in memory_injector._cached_result:
                        if not hasattr(mem, 'id'):
                            continue
                        try:
                            _mgr = memory_mgr
                        except NameError:
                            continue
                        for scope_name in ['project', 'local', 'user']:
                            try:
                                scope = MemoryScope(scope_name)
                                if scope in _mgr.memories:
                                    entry = _mgr.memories[scope]._id_index.get(mem.id)
                                    if entry:
                                        entry.usage_count += (2 if tool_error_count == 0 else -1)
                                        entry.last_accessed = time.time()
                                        break
                                        entry.last_accessed = time.time()
                                        break
                            except (ValueError, KeyError):
                                continue
                except Exception:
                    pass

            # 路由反馈学习：记录任务结果以优化未来路由
            if smart_router and task:
                try:
                    outcome = TaskOutcome(
                        task_text=task.raw_input if hasattr(task, 'raw_input') else str(task.id),
                        assigned_model=model.model_id if hasattr(model, 'model_id') else "unknown",
                        success=(tool_error_count == 0),
                        duration_ms=step * 2000.0,
                        cost_usd=0.0,
                        tool_errors=tool_error_count,
                        model_switches=model_switcher.switch_count() if model_switcher else 0,
                    )
                    smart_router.learner().record_outcome(outcome)
                except Exception:
                    pass

        # 控制论反馈：记录模式有效性
        if enable_work_chain and feedback_controller and task:
            pattern_id = f"{task_metadata.get('intent_type', 'unknown')}_{task.id}"
            feedback_controller.record_pattern_effectiveness(
                pattern_id, tool_error_count == 0
            )

        # 稳定性监测：记录快照
        if stability_monitor:
            from minicode.stability_monitor import MetricSnapshot
            snapshot = MetricSnapshot(
                timestamp=time.time(),
                error_rate=float(tool_error_count) / max(step, 1),
                avg_latency=step * 2.0,  # 简化估算
                context_usage=context_manager.get_stats().usage_percentage if context_manager else 0.0,
                active_tasks=1,
            )
            stability_monitor.record_snapshot(snapshot)
            if context_cybernetics:
                stability_monitor.feed_orchestrator(context_cybernetics)

        # 高级控制论：最终状态报告
        if enable_work_chain:
            # 状态观测器报告
            if state_observer:
                state_summary = state_observer.get_state_summary()
                logger.info("State observer summary: %s", state_summary)

            # 预测控制器报告
            if predictive_controller:
                pred_summary = predictive_controller.get_prediction_summary()
                logger.info("Prediction summary: accuracy=%s", pred_summary.get("accuracy", {}))

            # 自愈引擎统计
            if self_healing_engine:
                healing_stats = self_healing_engine.get_healing_statistics()
                logger.info("Self-healing stats: %s", healing_stats)

            # 多变量解耦状态
            if decoupling_controller:
                coupling_status = decoupling_controller.get_coupling_status()
                logger.info("Coupling status: strong=%s", coupling_status.get("strong_couplings", []))

        # 上下文管理管线统计 (Claude Code-style + Cybernetics)
        if context_compactor:
            compactor_stats = context_compactor.get_stats()
            logger.info(
                "ContextCompactor: passes=%d persisted=%d dedup=%d "
                "microcompact=%d boundaries=%d circuit=%s",
                compactor_stats["total_passes"],
                compactor_stats["tool_results_persisted"],
                compactor_stats["read_dedup_entries"],
                compactor_stats["microcompact_tokens_cleared"],
                compactor_stats["auto_compact_boundaries"],
                "TRIPPED" if compactor_stats["circuit_breaker_tripped"] else "OK",
            )
        # 控制论闭环统计 (Engineering Cybernetics)
        if context_cybernetics:
            cyber_stats = context_cybernetics.get_stats()
            logger.info(
                "Cybernetics: cycles=%d usage=%.1f%% pid_out=%.2f "
                "predict_overflow=%s urgency=%.2f threshold=%.2f feedback_eff=%.0f%%",
                cyber_stats["cycles_executed"],
                (cyber_stats["sensor"]["current_usage"] or 0) * 100,
                cyber_stats["pid"]["last_output"] or 0,
                cyber_stats["predictor"]["turns_until_overflow"],
                cyber_stats["predictor"]["urgency"] or 0,
                cyber_stats["threshold"]["effective_threshold"] or 0,
                (cyber_stats["feedback"]["effectiveness_rate"] or 0) * 100,
            )
        # 成本控制闭环统计 (BudgetPIDController)
        if cost_control:
            cc_stats = cost_control.get_stats()
            adj = cc_stats.get("adjustment")
            logger.info(
                "CostControl: cycles=%d cost/min=$%.4f pid_out=%.2f "
                "budget_mult=%.2f threshold_mult=%.2f [%s]",
                cc_stats["cycles_executed"],
                cc_stats["sensor"]["cost_per_min"],
                cc_stats["pid"]["last_output"] or 1.0,
                adj["budget_mult"] if adj else 1.0,
                adj["threshold_mult"] if adj else 1.0,
                adj["reason"] if adj else "none",
            )
        # 双层 PID 闭环: Cybernetics → FeedbackController
        if context_cybernetics and feedback_controller:
            system_state = context_cybernetics.to_system_state()
            control_signal = feedback_controller.observe(system_state)
            if control_signal.force_compaction and context_cybernetics.enabled:
                logger.info(
                    "Dual-PID: FeedbackController force_compaction=True, "
                    "stability=%.2f performance=%.2f",
                    system_state.stability_score(),
                    system_state.performance_score(),
                )
            # Apply outer-loop ControlSignal to runtime parameters
            if control_signal.confidence > 0.6:
                if control_signal.limit_max_steps and control_signal.limit_max_steps < max_steps:
                    logger.info(
                        "FeedbackController: limiting max_steps %d → %d",
                        max_steps, control_signal.limit_max_steps,
                    )
                    max_steps = control_signal.limit_max_steps
                if control_signal.adjust_token_budget != 1.0:
                    if context_compactor and hasattr(context_compactor, '_tool_budget') and context_compactor._tool_budget:
                        new_budget = max(
                            1000,
                            int(context_compactor._tool_budget.budget_per_message * control_signal.adjust_token_budget),
                        )
                        context_compactor._tool_budget.budget_per_message = new_budget
                        logger.info(
                            "FeedbackController: token budget adjusted to %d (mult=%.2f)",
                            new_budget, control_signal.adjust_token_budget,
                        )
                if control_signal.reduce_parallelism:
                    # Cap tool concurrency at 2
                    if not hasattr(tool_scheduler, '_force_max_workers'):
                        tool_scheduler._force_max_workers = 2
                    logger.info(
                        "FeedbackController: reduce_parallelism → max_workers=2 "
                        "(oscillation=%.2f)", control_signal.oscillation_index,
                    )
                if control_signal.adjust_concurrency != 0:
                    cap = max(1, 4 + control_signal.adjust_concurrency)
                    tool_scheduler._force_max_workers = cap
                    logger.info(
                        "FeedbackController: adjust_concurrency=%+d → max_workers=%d",
                        control_signal.adjust_concurrency, cap,
                    )
                if control_signal.increase_model_level:
                    logger.info(
                        "FeedbackController: model upgrade recommended (errors=%.2f perf=%.2f)",
                        system_state.error_frequency, system_state.performance_score(),
                    )
                    if model_switcher:
                        model_switcher._pending_upgrade = True
                if control_signal.decrease_model_level:
                    logger.info(
                        "FeedbackController: model downgrade recommended (efficiency=%.2f)",
                        system_state.token_efficiency,
                    )
                if control_signal.suggest_memory_persistence:
                    logger.info("FeedbackController: persisting working memory")
                    if context_compactor and hasattr(context_compactor, '_tool_budget'):
                        try:
                            context_compactor._tool_budget.flush()
                        except Exception:
                            pass
                if control_signal.recommend_skill_update:
                    logger.info("FeedbackController: skill update recommended (pattern=%.2f)",
                               system_state.pattern_reuse_rate)

            # 自适应PID调参：每20轮自动调节内外环PID参数
            if adaptive_pid_tuner and step > 0 and step % 20 == 0 and feedback_controller:
                try:
                    stability_error = 1.0 - system_state.stability_score()
                    perf_score = system_state.performance_score()
                    tuned = adaptive_pid_tuner.tune(
                        stability_error, dt=1.0, performance_score=perf_score
                    )
                    if tuned and adaptive_pid_tuner._performance_history:
                        recent_perf = adaptive_pid_tuner._performance_history[-5:]
                        avg_perf = sum(recent_perf) / len(recent_perf)
                        if context_cybernetics:
                            cp = context_cybernetics.pid
                            cp.kp = tuned.kp
                            cp.ki = tuned.ki
                            cp.kd = tuned.kd
                            logger.info(
                                "AdaptivePIDTuner: context PID tuned kp=%.3f ki=%.3f kd=%.3f "
                                "method=%s perf=%.2f",
                                tuned.kp, tuned.ki, tuned.kd,
                                adaptive_pid_tuner._active_method.value if hasattr(adaptive_pid_tuner, '_active_method') else 'unknown',
                                avg_perf,
                            )
                except Exception:
                    pass  # 调参失败不能拖垮主循环

        # 总监督层: 汇总局部控制器输出为统一风险视图
        if cybernetic_supervisor:
            supervisor_snapshots = []
            if context_cybernetics:
                supervisor_snapshots.append(
                    cybernetic_supervisor.snapshot_from_context(context_cybernetics.get_stats())
                )
            if cost_control:
                supervisor_snapshots.append(
                    cybernetic_supervisor.snapshot_from_cost(cost_control.get_stats())
                )
            if tool_scheduler.last_decision:
                supervisor_snapshots.append(
                    cybernetic_supervisor.snapshot_from_tool_decision(
                        tool_scheduler.last_decision.to_dict()
                    )
                )
            supervisor_report = cybernetic_supervisor.report(supervisor_snapshots)
            save_supervisor_report(supervisor_report)
            logger.info(
                "CyberneticSupervisor: health=%.2f risk=%s actions=%s",
                supervisor_report.overall_health,
                supervisor_report.risk_level.value,
                "; ".join(supervisor_report.recommended_actions[:3]),
            )

