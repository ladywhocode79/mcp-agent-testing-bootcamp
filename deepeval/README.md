# DeepEval Test Scaffolds — MCP Agent Testing Bootcamp

*Companion test suite for the 21-day bootcamp.*
*Source paper: "AI Assurance: A Comprehensive Testing Strategy for Enterprise AI Systems" (Badagi et al., 2026).*

---

## Where this fits in the bootcamp

```
mcp-agent-testing-bootcamp/
  deepeval/
    layer1_component/        ← Phase 2, Days 8–10
    layer2_agent_trajectory/ ← Phase 2, Days 11–12
    layer3_multiagent/       ← Phase 2, Days 13–14
    layer4_rag/              ← Phase 3, Days 15–17
    README.md                ← this file
```

---

## The pyramid at a glance

| Layer | Name | Bootcamp days | Failure modes caught |
|-------|------|---------------|----------------------|
| 1 | Component Evaluations | 8–10 | Hallucination, grounding failure, guardrail bypass, wrong tool routing |
| 2 | Agent Behavioural | 11–12 | Trajectory collapse, tool selection error, param construction error, state retention failure |
| 3 | Multi-Agent & Workflow | 13–14 | Context loss at handoff, orchestration error, infinite loop, emergent coordination failure |
| 4 / RAG | Business Outcome + RAG | 15–17 | Incomplete retrieval, retrieval noise, hallucination (end-to-end), goal completion failure, red-team |

**Key rule (memorise this):** Catch failures as low as possible. A Layer 0/1 fix costs near zero and points to the exact bug. A Layer 4 failure is expensive, vague, and may already be in production.

---

## Install

```bash
pip install deepeval ragas pytest
deepeval login   # creates ~/.deepeval config with your API key
```

---

## Run a layer

```bash
# Layer 1 only
pytest deepeval/layer1_component/ -v

# Layer 2 only
pytest deepeval/layer2_agent_trajectory/ -v

# All layers
pytest deepeval/ -v

# With DeepEval's dashboard report
deepeval test run deepeval/layer1_component/test_layer1_component.py
```

---

## How to wire up your MCP agent

Each test file has clearly marked stubs (`raise NotImplementedError`).
Replace them with your actual agent calls. Example pattern:

```python
# In test_layer1_component.py
def get_actual_output(input_text: str, context: list[str]) -> str:
    # BEFORE (stub):
    raise NotImplementedError("Wire up your MCP agent here.")

    # AFTER (your real agent):
    from your_agent import MCPAgent
    agent = MCPAgent(server="filesystem")
    return agent.run(input_text, context=context).text
```

Same pattern applies in every layer file.

---

## The diagnostic profile (RAG / Layer 4)

When scores come back, read them as a profile:

| Precision | Recall | Faithfulness | Relevancy | What to fix |
|-----------|--------|--------------|-----------|-------------|
| Low | Low | Any | Any | Retrieval pipeline — check chunking + embeddings |
| High | Low | Any | Any | Index coverage — recall issue |
| High | High | Low | Any | Generation grounding — hallucination |
| High | High | High | Low | Prompt shaping — response clarity |
| High | High | High | High | All good ✅ |

**Non-negotiable gates — block deployment if either fails:**
- `faithfulness >= 0.90`
- `context_recall >= 0.85`

---

## One-line anchors from the paper

1. Testing AI = **continuous risk reduction**, not verification.
2. **Correct output ≠ correct trajectory** — the false positive that bites later.
3. Tool selection and parameter construction **fail independently** — design datasets to isolate them.
4. A scenario only tests what it can make **verifiably different** (state retention, handoffs).
5. If a recovery path has **no test scenario**, it hasn't been evaluated.
6. **Catch failures as low in the pyramid as possible.**
