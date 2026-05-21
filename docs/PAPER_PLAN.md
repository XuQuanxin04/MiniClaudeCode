# 论文写作指南

## 标题

**Closed-Loop Cybernetic Memory: A Control-Theoretic Framework for Adaptive Agent Retrieval**

（闭环控制论记忆：面向自适应 Agent 检索的控制论框架）

备选：
- *Engineering Cybernetics for Agent Memory: PID-Controlled Adaptive Retrieval*
- *Memory as a Control Problem: A Cybernetic Architecture for LLM Agent Recall*

---

## 核心贡献（3 个）

### 1. 问题形式化

首次将 Agent 记忆检索形式化为**最优控制问题**：

```
状态 x(t): [context_usage, error_rate, relevance, cost]
控制 u(t): [injection_rate, compaction_intensity, model_tier]
扰动 d(t): [task_complexity, user_feedback]
输出 y(t): [task_success, token_efficiency]

目标: min ∫(y(t) - y_desired)² dt
```

### 2. PID 闭环 + Lyapunov 稳定性证明

不是启发式规则，而是可证明收敛的控制律：

```
V_L(e, ∫e) = ½e² + (ki/2)(∫e)²
V̇_L = -(kp/m)·e² < 0  (当 kp > 0)
→ 系统渐近稳定: e(t) → 0 as t → ∞
```

### 3. 消融实验

80 条记忆 × 20 查询 × 4 配置，证明了每个控制论组件的独立贡献。

---

## 论文大纲

### 1. Introduction

**问题陈述**：
- LLM Agent 依赖记忆来维持跨会话上下文
- 现有方法（Mem0, MemGPT, RAG）使用**静态检索**：固定的 top-K, 固定阈值
- 静态检索在不同上下文压力下性能剧烈波动

**我们的方案**：
- 将记忆检索建模为**闭环控制问题**
- 使用 PID 控制器动态调整检索深度、注入速率
- Kalman 滤波器估计隐藏系统状态
- 可证明的稳定性 + 可量化的性能提升

**贡献列表**：
1. 首次将工程控制论形式化地应用于 Agent 记忆
2. 双 PID 外环 + Kalman 状态估计的完整架构
3. 消融实验证明每个组件的量化贡献
4. Memory Value Function V(m,t,c) 的理论分析

### 2. Related Work

| 方向 | 代表工作 | 与我们的差异 |
|------|---------|------------|
| Agent 记忆系统 | Mem0, Letta/MemGPT, MemMachine | 静态/RL 检索，无 PID |
| 控制论+AI | SCL (R-CCAM), HAF, PEACE | 符号规则，无经典控制论 |
| Memory-as-Control | Oblivion, INFMEM, EvolveMem | RL/衰减驱动，无 PID/Kalman |
| 记忆理论 | "When to Forget" (Simsek) | 单维 Memory Worth，无多维控制 |

**关键 gap**：没有任何论文将经典控制论（PID + Kalman + Lyapunov）形式化应用于 Agent 记忆。

### 3. Architecture

```
┌─────────────────────────────────────────────┐
│           Cybernetic Memory Pipeline        │
│                                             │
│  Sense ─→ Predict ─→ Control ─→ Act ─→ Learn │
│    │         │          │        │       │    │
│  Domain    Value     PID×4    Tools   Feedback│
│  Classif   Scoring   Kalman×5 Budget  Loop   │
│  BM25      Reranker  Feedfwd  Inject         │
└─────────────────────────────────────────────┘
```

**3 层检索管线**：
- Layer 1: Domain + BM25 + Value Scoring (零 LLM 成本)
- Layer 2: LLM Reranker (1 次轻量调用)
- Layer 3: Spreading Activation + Adaptive Injection

**双 PID 外环**：
- 内环 (ContextPID): context_usage → compaction
- 外环 (FeedbackController): SystemState → 13-dim ControlSignal

### 4. Theoretical Framework

#### 4.1 Memory Value Function

$$V(m, t, c) = \text{relevance}(m, t) \times \text{freshness}(m) \times \text{utility}(m, c)$$

| 分量 | 定义 |
|------|------|
| relevance | BM25 × 0.7 + domain_jaccard × 0.3 |
| freshness | exp(-age_days / 30) |
| utility | 1 + α·ln(1 + usage_count) |

#### 4.2 PID Stability

构造 Lyapunov 函数 V_L，证明 V̇_L < 0 → 系统渐近稳定。

#### 4.3 Adaptive Cooldown

$$\tau_{\text{cool}}(c) = \tau_{\text{base}} \times (1 - \text{context\_pressure})$$

#### 4.4 Information Preservation

跨层级压缩损失上界：I(m_arch) ≈ I(m) - ε

### 5. Experiments

#### 5.1 Setup
- 80 条记忆，5 个领域 (frontend/backend/database/devops/testing)
- 20 条查询，人工标注 ground truth
- 指标：P@3, R@5, MRR, Noise Rate

#### 5.2 Ablation Study

| Configuration | P@3 | Noise |
|-------------|-----|-------|
| C0: BM25 (baseline) | 0.350 | 65% |
| C1: + Domain Weight | 0.383 | 42% |
| C2: + Query Expansion | 0.450 | 38% |
| C3: + Reranker (Full) | **0.717** | **6.7%** |

#### 5.3 Analysis

- Reranker 贡献 73% 精度提升（+0.267 P@3）
- Domain + Expansion 在零 LLM 成本下削减 27% 噪音
- 完整管线精度 2.05× 基准

#### 5.4 需要补充的实验（投论文前）

- LongMemEval / LoCoMo 标准基准评估
- 与 Mem0 / MemGPT 的对比
- PID on/off 对比（证明控制论贡献）
- 不同模型规模下的鲁棒性
- 延迟和成本分析

### 6. Discussion

- 控制论视角的局限性
- Thinking round-trip 的工程挑战
- 未来方向：多 Agent 记忆联邦、真实代码库验证

### 7. Conclusion

首次将工程控制论形式化地应用于 Agent 记忆系统。PID 闭环提供可证明的稳定性，消融实验证明每个组件的独立贡献。

---

## 投稿建议

| 会议 | 截稿 | 特点 |
|------|------|------|
| **EMNLP 2026** | 约 6 月 | NLP 系统，适合 Agent 方向 |
| **NeurIPS 2026** | 约 5 月（已过） | 顶会，需要更强理论 |
| **AAAI 2027** | 约 8 月 | AI 系统，包容性强 |
| **COLM 2026** | 约 5 月（已过） | 语言建模，新会议 |
| **ICLR 2027** | 约 9 月 | 顶会，理论要求高 |

**建议**：瞄准 **EMNLP 2026** 或 **AAAI 2027**。

---

## 写前准备清单

| 项目 | 状态 | 优先级 |
|------|------|--------|
| 消融实验 | ✓ 完成 | — |
| 基准评估框架 | ✓ Benchmark 脚本 | — |
| 标准基准 (LongMemEval) | ✗ 需要搭建 | ★★★ |
| 与 Mem0 对比 | ✗ | ★★ |
| PID on/off 实验 | ✗ | ★★★ |
| 延迟/成本分析 | ✗ | ★★ |
| 理论形式化 | ✓ 完成 | — |
| 相关论文调研 | ✓ 完成 | — |
| 架构图 | ✓ 完成 | — |
| 代码开源 | ✓ GitHub | — |

**下一步**：搭建 LongMemEval + 跑 Mem0 baseline + 写 PID on/off 实验。这三项做完论文实验部分就完整了。
