# AI Assurance Study Sheet — Failure Taxonomy + Pyramid Layers 2–3

*Distilled for MCP-agent / DeepEval testing. Source: "AI Assurance: A Comprehensive Testing Strategy for Enterprise AI Systems" (Badagi et al., 2026).*

---

## Part 1 — The Failure Taxonomy (why it exists)

**The principle:** A test strategy without a failure taxonomy is operationally weak. Tests exist to *detect failure modes*. If the failure modes aren't classified, coverage becomes "whatever the engineer thought of." The taxonomy is a **design input** — each category demands a *distinct* test mechanism.

### The 15 modes, 5 categories

| # | Category | Failure mode | Manifestation | Test mechanism that catches it |
|---|----------|--------------|---------------|-------------------------------|
| 1 | **Grounding** | Hallucination | Fabricated facts stated confidently | Faithfulness eval vs. context |
| 2 | | Grounding failure | Response contradicts retrieved context | Semantic eval vs. retrieved docs |
| 3 | | Retrieval failure | Wrong documents surfaced | Context precision/recall (isolated) |
| 4 | **Reasoning** | Reasoning failure | Wrong conclusion from correct premises | Reasoning-quality rubric |
| 5 | | Instruction drift | Constraints abandoned mid-session | Stateful / instruction-adherence eval |
| 6 | | Trajectory collapse | Locally valid steps → invalid outcome | Trajectory eval (Layer 2) |
| 7 | **Safety** | Prompt injection | Malicious input overrides instructions | Adversarial test sets |
| 8 | | Unsafe compliance | Policy-violating request fulfilled | Guardrail eval + red-team |
| 9 | | Reward hacking | Metric satisfied, intent violated | Goal-completion eval (Layer 4) |
| 10 | **Coordination** | Context loss | Earlier state forgotten | Handoff / state-retention eval (Layer 3) |
| 11 | | Tool misuse | Wrong tool or wrong parameters | Tool-call accuracy (Layer 2) |
| 12 | | Over-delegation | Sub-agents exceed scope | Orchestration eval (Layer 3) |
| 13 | | Infinite loops | Agent retries without terminating | Failure-recovery eval (Layer 3) |
| 14 | **Stochastic** | Stochastic inconsistency | Unacceptable pass-rate variance | Repeated runs + consistency analysis |
| 15 | | Latent emergence | New capability appears after model update | Cross-version emergent-behavior monitoring |

### The "unknown unknowns" problem
The failure space is **not enumerable in advance** — the model is sensitive to inputs no one anticipated. Conventional coverage can't address this. Three complementary practices:
- **Adversarial discovery** — red-teaming to find breaking inputs on purpose.
- **Exploratory evaluation** — open-ended runs to *surface* novel patterns, not confirm known ones.
- **Emergent capability monitoring** — track behavior across model versions, not just regression on known cases.

> **Takeaway:** If your test strategy has no explicit coverage for one of these five categories, that failure mode is invisible until it hits production.

---

## Part 2 — Layer 2: Agent Behavioural Evaluations

**The question shifts** from *"did this component produce the right output?"* (Layer 1) to **"did this agent take the right *path*?"**

### The core insight (memorize this)
> A correct final answer via an incorrect trajectory is a **false positive and a production risk.** The system got lucky once; the fragility surfaces under different conditions. Layer 2 exists to catch this *before* production.

### What gets evaluated

**1. Trajectory evaluation** — the central technique.
A trajectory = the full sequence of reasoning steps, tool invocations, and state transitions. Two agents can reach identical answers via very different paths (one reliable, one fragile). Evaluating only the final output misses this entirely.

- *Reference trajectory* — exact acceptable step sequence. Precise but **brittle** (a valid alternative path fails). Use for **constrained tasks with one known optimal path.**
- *Rubric* — defines acceptable *properties* of any path (mandatory steps + prohibited patterns). Flexible but must be designed carefully so it doesn't pass clearly wrong paths. Use for **tasks with multiple valid approaches.**

**2. Tool call accuracy** — two *independent* failure dimensions:
- **Tool selection** — right tool chosen?
- **Parameter construction** — right inputs passed?

These fail independently → your dataset **must** include cases that isolate each: correct tool + wrong params, AND wrong tool + valid params. A dataset where both always succeed or both always fail **cannot localise regressions** after a model update.

**3. Reasoning quality** — is the inference chain logically consistent and instruction-adherent? Assessed with rubrics *at each key decision point*. Evaluating this *separately from output correctness* is what distinguishes "right answer, flawed reasoning" from "sound reasoning, data error."

**4. State retention** — does the agent carry context across steps without re-asking for info already given, dropping earlier constraints, or contradicting prior reasoning?
- **Design rule:** a stateful scenario is only valid if context loss produces a *verifiably wrong* output. The early-turn info must be *necessary* for the correct final answer. A state-keeping agent and a context-dropping agent must produce **detectably different outputs** on the same scenario — otherwise the scenario doesn't test state.

### Layer 2 KPIs
| KPI | Definition |
|-----|------------|
| Trajectory correctness rate | % tasks where agent followed an acceptable trajectory (not just correct output) |
| Tool call accuracy | % calls with correct tool **AND** correct parameters |
| Unnecessary call rate | % calls that were redundant/avoidable |
| Instruction adherence rate | % tasks maintaining all constraints through completion |
| Single-agent task success rate | % tasks completed correctly end-to-end |

---

## Part 3 — Layer 3: Multi-Agent & Workflow Evaluations

**Where agentic systems get genuinely complex.** Different agents handle planning, retrieval, synthesis, formatting. Unlike microservices with **explicit typed contracts**, agents communicate in **natural language** — so contracts are implicit and subject to ambiguity, context loss, misinterpretation. A planning agent's ambiguous brief can make a downstream agent proceed wrongly **without raising any error.**

### What gets evaluated

**1. Agent handoff accuracy** — the primary concept. *Distinct from* tool-call accuracy:
- Tool-call accuracy (L2) = how correctly an agent invokes a **tool**.
- Handoff accuracy (L3) = how correctly one agent transfers **task context** to another: right scope, right constraints, right prior context.
- A handoff can have perfect tool-call mechanics and still transfer an incomplete/incorrect task context.

- **Design rule:** construct scenarios where the downstream agent's correct output *depends explicitly* on a constraint the upstream agent must transfer. Run a **baseline** (complete brief) vs. a **variation** (one element omitted/corrupted). Outputs must differ detectably. Otherwise you can't tell a handoff failure from a downstream reasoning failure.

**2. Orchestration correctness** — does the orchestrator route subtasks to the right sub-agents with the right scope and sequence? Failures here are invisible in individual tests: each sub-agent handles its task fine, but the *wrong task was assigned* or the *sequence was wrong*.

**3. Failure recovery** — when a sub-agent errors / returns empty / returns malformed output, does the orchestrator correctly **retry, fall back, request clarification, or terminate cleanly?**
- **Design rule (failure injection):** deliberately break components — API errors, empty retrieval, malformed tool responses. Each failure mode *and each recovery path* should be its **own distinct scenario.** Recovery seen only as a side effect of general quality testing is **not reliably evaluated** — no test scenario = not evaluated.

**4. Emergent behaviour detection** — run full workflows against scenarios engineered to *force* coordination failures: ambiguous handoff instructions, state conflicts between agents, an agent needing to interpret another's incomplete output without wrongly flagging it as an error. **Build these from real production incidents** — production is currently the main source of emergent-failure discovery.

### Layer 3 KPIs
| KPI | Definition |
|-----|------------|
| Agent handoff accuracy | % handoffs where receiving agent has correct context/scope/constraints |
| State retention across agents | % multi-agent tasks preserving context across all boundaries |
| Recovery success rate | % sub-agent failures where orchestrator recovers and completes |
| Trajectory efficiency score | Actual steps ÷ optimal steps |
| Multi-agent task success rate | % tasks correct across the full workflow |

---

## Part 4 — Why this maps directly onto MCP-agent testing

MCP interactions *are* the L2/L3 surface, made concrete:

- **Tool-call accuracy (L2)** ≈ correctness of MCP `tools/call` — right tool selected from the server's exposed list, schema-valid arguments constructed. The two-dimension rule applies directly: test "right tool / bad args" and "valid args / wrong tool" separately.
- **State retention (L2)** ≈ context carried across multiple MCP round-trips within one task without re-fetching resources already retrieved.
- **Handoff accuracy (L3)** ≈ context fidelity when one agent's MCP result becomes another agent's input — the implicit "contract" is exactly the language-based handoff the paper warns about.
- **Failure recovery (L3)** ≈ how the orchestrator handles MCP errors, empty `resources/read`, malformed tool results, or a server timing out. Each needs its own injected-failure scenario.
- **Emergent behaviour (L3)** ≈ multi-server / multi-agent coordination where no single MCP call is wrong but the composition fails.

### Diagnostic shortcut (catch failures low)
A symptom at Layer 4 usually has a root cause at Layer 1 or 2. Debug **bottom-up.** Example from the paper: goal-completion drops 3 weeks after a model update → L3 passes → L2 flags tool-call failures on *date params* → fix is a targeted **Layer 1 prompt adjustment**. Days of investigation reduced to hours.

---

## One-line anchors to memorize
1. Testing AI = **continuous risk reduction**, not verification.
2. **Catch failures as low in the pyramid as possible.**
3. Correct output ≠ correct trajectory (the false positive that bites later).
4. **Tool selection and parameter construction fail independently** — design datasets to isolate them.
5. A scenario only tests what it can make **verifiably differ** (state retention, handoffs).
6. If a recovery path has no test scenario, **it hasn't been evaluated.**
