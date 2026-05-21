# MiniCode — Cybernetic AI Coding Agent

> **Terminal-first AI coding assistant with closed-loop self-regulation**
>
> 钱学森工程控制论驱动 · 15+ 自适应控制器 · 737 tests

MiniCode is a terminal AI coding agent. It reads your codebase, executes tools, and writes code — like Claude Code or Aider. **What makes it different: it regulates itself.**

Every coding agent hits the same walls — context overflow, runaway costs, tool errors, irrelevant memory. MiniCode uses **engineering cybernetics** (PID loops, Kalman filters, feedback control) to detect these problems and auto-correct in real time. No human intervention needed.

---

## Quick Start

```bash
git clone https://github.com/QUSETIONS/MiniCode-Python.git
cd MiniCode-Python
pip install -e .
python -m minicode.main
```

Mock mode (no API key):
```bash
MINI_CODE_MODEL_MODE=mock python -m minicode.main
```

---

## What It Can Do

MiniCode handles real coding tasks end-to-end. **10/10 real API tests pass with zero errors:**

| Task | Time | Result |
|------|------|--------|
| `Create hello.py with a hello() function` | 33s | ✓ |
| `Find all files containing "cybernetic"` | 25s | ✓ |
| `Read README and summarize` | 15s | ✓ |
| `Edit: rename function + change return value` | 39s | ✓ |
| `Multi-step: grep → read → analyze` | 68s | ✓ |
| `Create utils with ISO 8601 timestamp` | 42s | ✓ |

---

## Architecture: The Cybernetic Loop

MiniCode wraps the LLM in a **Sense → Predict → Control → Act** feedback loop:

```
User Input → Agent Loop → Response
                │
    ┌───────────┼───────────┐
    │           │           │
  SENSE      CONTROL       ACT
  sensors    PID ×4       tools
  Kalman×5   feedback     budget
  metrics    adaptive     compact
```

### What Gets Auto-Regulated

| Problem | Controller | Action |
|---------|-----------|--------|
| Context near limit | ContextPIDController | Auto-compaction, strategy selection |
| Tool errors spiking | SelfHealingEngine | Safe mode, reduce concurrency |
| Cost exceeding budget | BudgetPIDController | Tighten token budget |
| Agent oscillating | FeedbackController | Reduce parallelism, dampen PID |
| Task stalling | ProgressController | Switch strategy, narrow scope |
| Memory irrelevant | DomainClassifier + Reranker | Domain-filter, LLM-curate top-3 |

### Controller Matrix

| Controller | Type | What It Does |
|-----------|------|-------------|
| ContextPIDController | PID | Usage → compaction strength |
| BudgetPIDController | PID | Cost → token budget |
| FeedbackController | Dual-PID | System state → 13-dim control signal |
| AdaptivePIDTuner | Self-tuning | Auto-tunes every 20 turns |
| StateObserver | Kalman ×5 | Hidden state from observables |
| FeedforwardController | Preemptive | Intent → tool config |
| PredictiveController | Forecast | Time series → proactive actions |
| DecouplingController | RGA | Multi-variable coupling |
| SelfHealingEngine | Recovery | 8 fault types auto-heal |
| StabilityMonitor | Health | 6-dim health scoring |
| ProgressController | Stall | Strategy suggestions |
| MemoryInjectionController | PID | Controls injection rate |
| ModelSelectionController | Router | Risk/cost model selection |
| DomainClassifier | Classifier | 9 domains from file extensions |

---

## Smart Memory

Remembers your project conventions across sessions — not just keyword search, but a full adaptive pipeline:

```
Task + Files → DomainClassifier → BM25 + SparseVector(RRF)
  → Value(rel×fresh×util) → LLM Reranker → Spreading Activation → Inject
```

**Ablation study: 80 memories × 20 queries × 5 domains**

| Configuration | P@3 | Noise |
|-------------|-----|-------|
| BM25 (baseline) | 0.350 | 65% |
| + Domain + Expansion | 0.450 | 38% |
| + LLM Reranker (Full) | **0.717** | **6.7%** |

**2.05× precision improvement, 58% noise reduction.**

---

## Terminal Experience

| Feature | How |
|---------|-----|
| Colored diffs | `edit_file` output: +green/-red/@@cyan with word emphasis |
| Multi-line input | `Ctrl+J` inserts newline, multi-line rendering |
| Word-level editing | `Ctrl+←→` jump words, `Ctrl+W` delete word, `Ctrl+K` to end |
| Visual scrollbar | █ thumb, ▲▼ hints, ░ track |
| Bracketed paste | Batch insertion, control character stripping |
| Animated spinner | `⠋⠙⠹` 8fps during tool execution |
| Focus tracking | Auto-refresh on terminal tab switch |
| Fuzzy autocomplete | Prefix → subsequence fallback |

---

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/memory` | Memory system status |
| `/context` | Context window usage |
| `/cybernetics` | Controller health dashboard |
| `/skills` | List discoverable skills |
| `/exit` | Save session and exit |

---

## Configuration

`~/.mini-code/settings.json`:
```json
{
  "model": "deepseek-v4-pro[1m]",
  "env": {
    "ANTHROPIC_BASE_URL": "https://your-endpoint",
    "ANTHROPIC_AUTH_TOKEN": "your-token"
  }
}
```

Environment variables:
- `ANTHROPIC_BASE_URL` — API endpoint
- `ANTHROPIC_AUTH_TOKEN` — API key
- `MINICODE_MODEL_TIMEOUT` — API timeout in seconds
- `MINICODE_TOOL_TIMEOUT` — Tool execution timeout

---

## Testing

```bash
pytest  # 737 passed, 2 skipped
```

---

## MiniCode Ecosystem

| Repo | Role |
|------|------|
| [MiniCode](https://github.com/LiuMengxuan04/MiniCode) | Main project |
| [MiniCode-Python](https://github.com/QUSETIONS/MiniCode-Python) | Python (this repo) |
| [MiniCode-rs](https://github.com/harkerhand/MiniCode-rs) | Rust |

---

## Theory

MiniCode's control loop is mathematically grounded:

- **Lyapunov stability**: V̇ = -(kp/m)·e² < 0, proving PID convergence
- **Memory value function**: V(m,t,c) = relevance × freshness × utility
- **Kalman optimality**: minimum-variance unbiased state estimation
- **RRF fusion**: reciprocal rank fusion of BM25 + vector results

See [`docs/memory_theory.md`](docs/memory_theory.md) for the full formal treatment.

---

## Acknowledgments

- 钱学森《工程控制论》(Engineering Cybernetics, 1954)
- Wiener, *Cybernetics* (1948)
- Mem0, Letta/MemGPT, True Memory
