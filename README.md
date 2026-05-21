# MiniCode

> **The coding agent that regulates itself.**
>
> 15 cybernetic controllers watch your agent in real time —
> auto-fixing context overflow, tool errors, cost spikes, and irrelevant memory.
> No other coding agent does this.

[![Tests](https://img.shields.io/badge/tests-737%20passed-brightgreen)]()

---

## Why MiniCode

Every AI coding agent hits the same problems: context fills up, tools fail, costs spiral, memory is noise. The industry answer is "better prompts" or "bigger models." We took a different path.

MiniCode wraps the LLM in a **closed-loop cybernetic control system** — PID controllers, Kalman filters, feedback loops — that watches the agent in real time and auto-corrects before you even notice.

```
Your prompt → Agent Loop → Response
                  │
     ┌────────────┼────────────┐
     │            │            │
   SENSE       CONTROL        ACT
   Kalman×5    PID ×4        tools
   metrics     feedback      budget
```

**It's like ABS for your AI agent.**

| Problem | Traditional Fix | MiniCode |
|---------|----------------|----------|
| Context overflow | Retry with bigger window | PID auto-compacts |
| Tool errors pile up | Restart the session | Self-heals in real time |
| Cost runs away | Manual budget check | Budget PID throttles |
| Memory returns noise | Skip memory entirely | Domain filter + LLM curation |

---

## What It Can Do

Verified with DeepSeek V4 Pro. **10 real coding tasks, zero errors.**

```
Create hello.py with hello()        33s  ✓
Find all files with "cybernetic"    25s  ✓
Read + summarize README             15s  ✓
Edit: rename function + change val  39s  ✓
Multi-step: grep → read → analyze   68s  ✓
```
Full results: [`docs/test_results.md`](docs/test_results.md)

---

## Quick Start

```bash
git clone https://github.com/QUSETIONS/MiniCode-Python.git
cd MiniCode-Python
pip install -e .
python -m minicode.main
```

No API key? Mock mode works out of the box:
```bash
MINI_CODE_MODEL_MODE=mock python -m minicode.main
```

---

## Configuration

```json
{
  "model": "your-model",
  "env": {
    "ANTHROPIC_BASE_URL": "https://your-endpoint",
    "ANTHROPIC_AUTH_TOKEN": "your-token"
  }
}
```

---

## The 15 Controllers

Every turn, MiniCode's controllers measure, decide, and act:

| Controller | Job |
|-----------|-----|
| **ContextPIDController** | Usage → compaction strength |
| **BudgetPIDController** | Spending → token budget |
| **FeedbackController** | System health → 13-dim control signal |
| **AdaptivePIDTuner** | Auto-tunes PID gains every 20 turns |
| **StateObserver** | Kalman estimates of 5 hidden states |
| **SelfHealingEngine** | Detects and recovers 8 fault types |
| **FeedforwardController** | Pre-configures from task intent |
| **PredictiveController** | Forecasts and acts before problems hit |
| **DecouplingController** | Untangles multi-variable interactions |
| **StabilityMonitor** | Multi-dimensional health scoring |
| **ProgressController** | Detects task stalling |
| **DomainClassifier** | Auto-detects frontend/backend/db/devops |
| **MemoryInjectionController** | PID-controls memory injection rate |
| **ModelSelectionController** | Risk/cost-driven model selection |
| **CyberneticSupervisor** | Aggregates all controller states |

---

## Memory That Actually Works

Not keyword search. A full adaptive pipeline:

```
Task → DomainClassifier → BM25 + Vector(RRF) → Value Scoring
  → LLM Reranker (curates top-3 from 15) → Spreading Activation → Inject
```

**80 memories × 20 queries: P@3 0.35 → 0.72, noise 65% → 7%.**

The LLM Reranker uses the same model you're coding with to pick which memories actually matter.

---

## Terminal Polish

| Feature | Key |
|---------|-----|
| Colored diffs | `edit_file` output: +green/-red with word highlights |
| Multi-line input | `Ctrl+J` — paste code blocks directly |
| Word editing | `Ctrl+←→` `Ctrl+W` `Ctrl+K` |
| Visual scrollbar | █ • ▲ ▼ |
| Bracketed paste | Batch insert, strip control chars |
| Spinner | `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` at 8fps |
| Focus tracking | Auto-refresh on tab switch |
| Fuzzy complete | `/mem` matches `/memory` |

---

## Testing

```bash
pytest
# 737 passed, 2 skipped
```

---

## Theory

The control loop isn't heuristic — it's mathematically grounded:

| Concept | Formalization |
|---------|--------------|
| PID Stability | V̇ = -(kp/m)·e² < 0 (Lyapunov) |
| Memory Value | V(m,t,c) = relevance × freshness × utility |
| State Estimation | 5 Kalman filters, minimum-variance unbiased |
| Retrieval Fusion | RRF: BM25 + SparseVector cosine |

[`docs/memory_theory.md`](docs/memory_theory.md)

---

## Acknowledgments

钱学森《工程控制论》(1954) · Wiener *Cybernetics* (1948) · Mem0 / Letta / True Memory
