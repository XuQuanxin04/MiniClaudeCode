# MiniCode Python

MiniCode Python is the Python implementation in the MiniCode family. It is
developed in this repository and linked from the main MiniCode repository:

- Main MiniCode repository: [LiuMengxuan04/MiniCode](https://github.com/LiuMengxuan04/MiniCode)
- Python repository: [QUSETIONS/MiniCode-Python](https://github.com/QUSETIONS/MiniCode-Python)
- Rust repository: [harkerhand/MiniCode-rs](https://github.com/harkerhand/MiniCode-rs/tree/master)
- Java repository: [hobbescalvin414-tech/minicode4j](https://github.com/hobbescalvin414-tech/minicode4j/tree/feat/default-ts-ui)

This Python version focuses on a local terminal coding agent with an explicit
control layer around the model loop. Instead of treating context pressure, tool
failures, noisy memory, and cost drift as prompt-only problems, the agent
measures them every turn and feeds those signals back into runtime decisions.

The current repository version is centered on the root package configured in
`pyproject.toml`:

- active package: `minicode/`
- active tests: `tests/`
- active console entrypoint: `minicode-py = minicode.main:main`
- compatibility/staging mirror: `py-src/minicode/`

The README describes the active root package. The `py-src/` tree is kept aligned
for reference and migration work, but the installed package and test suite use
`minicode/`.

## What Changed

This version turns MiniCode from a mostly linear agent loop into a closed-loop
agent runtime:

```text
user task
   |
   v
agent loop -----> tools / files / terminal
   |
   v
sensors: context, cost, tool errors, progress, memory quality
   |
   v
controllers: PID, Kalman state observer, prediction, self-healing, routing
   |
   v
actions: compact context, change budget, cap concurrency, inject memory,
         route models, record reflection, recover from faults
```

The important engineering result is not just that these controllers exist. The
main agent loop now calls the unified `CyberneticOrchestrator` lifecycle:

- `wire_memory()`
- `wire_healing()`
- `inject_memories()`
- `step_start()`
- `step_end()`
- `reflect_on_task()`

That keeps controller initialization, memory injection, per-step observation,
feedback, self-healing, and post-task reflection tied to the same runtime
surface.

## Core Capabilities

### Cybernetic Control

MiniCode uses a multi-controller runtime:

- `FeedbackController` produces control signals for compaction, concurrency,
  step limits, token budget, model-level hints, and memory persistence.
- `ContextCyberneticsOrchestrator` runs context sensing, PID control,
  prediction, threshold adaptation, and compaction.
- `AdaptivePIDTuner` retunes PID parameters during longer runs.
- `StateObserver` estimates hidden runtime state with Kalman filters.
- `PredictiveController` raises proactive actions before pressure becomes a
  hard failure.
- `SelfHealingEngine` detects recoverable faults and delegates recovery.
- `CostControlLoop` adjusts token-budget behavior from spend signals.
- `ProgressController` detects stalled or unhealthy task progress.
- `CyberneticSupervisor` aggregates controller state into a single health view.

### Memory Pipeline

Memory is handled as an adaptive pipeline rather than raw keyword search:

```text
task + files
  -> DomainClassifier
  -> BM25 / indexed retrieval
  -> optional vector fusion
  -> LLM reranking
  -> memory injection
  -> reflection write-back
  -> maintenance / decay / conflict handling
```

The active facade is `MemoryPipeline`, wired through `CyberneticOrchestrator`.
It supports project memory retrieval, prompt injection, task reflection, and
background maintenance from one interface.

### Agent Loop Integration

The main loop in `minicode/agent_loop.py` now applies controller output to live
runtime knobs:

- `limit_max_steps` can reduce the remaining loop budget.
- `adjust_token_budget` changes the compactor tool-result budget.
- `reduce_parallelism` caps concurrent tool workers.
- `adjust_concurrency` updates the scheduler worker cap.
- `suggest_memory_persistence` flushes working memory.
- model upgrade/downgrade hints are surfaced to the model switcher layer.

The goal is conservative autonomy: controllers can intervene in measurable
runtime behavior, but the agent still remains inspectable and testable.

## Install

```bash
git clone https://github.com/QUSETIONS/MiniCode-Python.git
cd MiniCode-Python
python -m pip install -e .[dev]
```

Run the CLI:

```bash
minicode-py
```

Or run directly:

```bash
python -m minicode.main
```

For local smoke tests without a real model provider, use the mock model paths
covered by the test suite.

## Verify

The current root package was verified with:

```bash
python -m compileall -q minicode py-src\minicode tests
pytest -q
```

Latest local result:

```text
738 passed, 2 skipped, 3 warnings
```

The remaining warnings are unregistered `pytest.mark.benchmark` markers in the
memory benchmark tests. They do not indicate failing behavior.

## Repository Map

```text
minicode/
  agent_loop.py                 main agent runtime
  cybernetic_orchestrator.py    controller lifecycle facade
  context_cybernetics.py        context PID and compaction loop
  feedback_controller.py        outer-loop control signals
  self_healing_engine.py        fault detection and recovery
  memory_pipeline.py            unified memory facade
  memory_reranker.py            LLM-backed memory curation
  domain_classifier.py          task/file domain inference
  model_registry.py             model selection controller
  progress_controller.py        task health and stall detection

tests/
  test_agent_flow.py            end-to-end agent loop coverage
  test_cybernetics_*.py         control-stack tests
  test_memory_*.py              memory retrieval/injection tests

py-src/
  minicode/                     staging mirror kept aligned with root package

docs/
  OPTIMIZATION_SUMMARY.md       full optimization record
  memory_theory.md              memory/control theory notes
```

## Version Alignment

This branch is the active GitHub repository version for
`QUSETIONS/MiniCode-Python`. The root package (`minicode/`) is the canonical
runtime used by installation and tests. `py-src/minicode/` is retained as a
secondary source tree for compatibility with earlier project layout, and obvious
behavioral fixes are mirrored there when needed.

The main TypeScript repository can include this repository as
`external/MiniCode-Python`, but the Python package itself is installed and tested
from this repository root.

## Design Notes

MiniCode is intentionally small enough to inspect, but the runtime is no longer
a bare model wrapper. The design direction is:

- observe the agent while it works;
- convert observations into structured control signals;
- apply only bounded runtime actions;
- preserve logs, tests, and explicit artifacts so claims can be checked.

For the detailed optimization history, see
[`docs/OPTIMIZATION_SUMMARY.md`](docs/OPTIMIZATION_SUMMARY.md).
