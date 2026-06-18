"""
Layer 3 — Multi-Agent & Workflow Evaluations
=============================================
Bootcamp phase: Phase 2, Days 13–14
Paper reference: Section 5.4

What this layer catches
-----------------------
- Context loss across agent handoffs
- Orchestration routing to wrong sub-agent
- Over-delegation (sub-agents exceed intended scope)
- Infinite loop / missing termination
- Emergent failures invisible to individual agent tests

The critical distinction from Layer 2
---------------------------------------
Layer 2 tests a SINGLE agent end-to-end.
Layer 3 tests the COORDINATION between agents.

"Tool call accuracy (L2) measures how correctly an agent invokes a tool.
Agent handoff accuracy (L3) measures how correctly one agent transfers task
CONTEXT to another agent: right scope, right constraints, right prior context."

A handoff can have perfect tool-call mechanics and still transfer incomplete
or incorrect task context.  That failure is invisible to Layer 2.

Design rules
-------------
1. Handoff scenarios need a BASELINE (complete brief) + VARIATION (one element
   missing or corrupted).  Downstream agent's output must differ detectably
   between the two.  If it doesn't, the scenario doesn't test the handoff.

2. Failure injection: break components deliberately (empty retrieval, API error,
   malformed tool response).  Each failure mode AND each recovery path should be
   a DISTINCT scenario.  Recovery seen only as a side effect of quality testing
   is NOT reliably evaluated.

3. Emergent behaviour scenarios must force coordination failures — ambiguous
   handoff instructions, state conflicts between agents — not just pass normal
   inputs through the full pipeline.

Repo placement
--------------
  mcp-agent-testing-bootcamp/
    deepeval/
      layer3_multiagent/
        test_layer3_multiagent.py   ← this file
"""

import pytest
from dataclasses import dataclass, field
from typing import Optional
from deepeval import assert_test
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase


# ---------------------------------------------------------------------------
# Data structures for multi-agent runs
# ---------------------------------------------------------------------------
@dataclass
class AgentHandoff:
    from_agent: str
    to_agent: str
    context_transferred: dict       # what the upstream agent passed
    expected_context: dict          # what SHOULD have been transferred


@dataclass
class MultiAgentResult:
    task: str
    handoffs: list[AgentHandoff]
    final_output: str
    success: bool
    recovery_path_taken: Optional[str] = None   # "retry", "fallback", "clarify", "terminate"
    steps_taken: int = 0
    optimal_steps: int = 0


# ---------------------------------------------------------------------------
# Stubs — replace with your multi-agent MCP orchestrator
# ---------------------------------------------------------------------------
def run_multi_agent_pipeline(
    task: str,
    context: Optional[dict] = None,
    injected_failure: Optional[dict] = None,
) -> MultiAgentResult:
    """
    Replace with your real orchestrator call.
    injected_failure: dict like {"agent": "retrieval_agent", "error": "empty_result"}
    """
    raise NotImplementedError("Wire up your multi-agent MCP orchestrator here.")


# ---------------------------------------------------------------------------
# Test 1 — Agent Handoff Accuracy
# Catches: context loss, scope drift, constraint drop at handoff boundaries
#
# Pattern: run BASELINE (full brief) then VARIATION (one field dropped).
# Downstream output must differ detectably — if not, the scenario is invalid.
# ---------------------------------------------------------------------------
HANDOFF_CASES = [
    {
        "description": "Planner → Executor: output path constraint must transfer",
        "task": "Summarise the MCP protocol spec and write results to reports/mcp_summary.md",
        "expected_context_keys": ["output_path", "task_scope"],
        "baseline_context": {
            "output_path": "reports/mcp_summary.md",
            "task_scope": "summarise only",
        },
        "corrupted_context": {
            # output_path dropped — executor should either fail or use a wrong default
            "task_scope": "summarise only",
        },
        "corruption_key": "output_path",
    },
    {
        "description": "Retrieval → Synthesis: retrieved chunks must transfer intact",
        "task": "Retrieve the MCP tool spec and produce a test checklist from it.",
        "expected_context_keys": ["retrieved_chunks", "source_document"],
        "baseline_context": {
            "retrieved_chunks": ["chunk_1", "chunk_2"],
            "source_document": "mcp_spec.md",
        },
        "corrupted_context": {
            "retrieved_chunks": [],   # empty retrieval — synthesis should signal incompleteness
            "source_document": "mcp_spec.md",
        },
        "corruption_key": "retrieved_chunks",
    },
]


class TestAgentHandoffAccuracy:
    @pytest.mark.parametrize("case", HANDOFF_CASES)
    def test_baseline_handoff_has_required_keys(self, case):
        """Baseline: all required context keys must reach downstream agent."""
        result = run_multi_agent_pipeline(case["task"], context=case["baseline_context"])

        for handoff in result.handoffs:
            for key in case["expected_context_keys"]:
                assert key in handoff.context_transferred, (
                    f"Handoff from {handoff.from_agent} → {handoff.to_agent} "
                    f"is missing required key '{key}'.\n"
                    f"Transferred: {handoff.context_transferred}"
                )

    @pytest.mark.parametrize("case", HANDOFF_CASES)
    def test_corrupted_handoff_produces_different_output(self, case):
        """
        Variation: corrupted context must produce a detectably different output.
        If baseline and corrupted produce the same output, the scenario doesn't
        test the handoff — redesign the scenario.
        """
        baseline_result   = run_multi_agent_pipeline(case["task"], context=case["baseline_context"])
        corrupted_result  = run_multi_agent_pipeline(case["task"], context=case["corrupted_context"])

        assert baseline_result.final_output != corrupted_result.final_output, (
            f"SCENARIO DESIGN WARNING: Baseline and corrupted context produced identical output "
            f"for task '{case['task']}'. "
            f"This scenario does NOT test the handoff — redesign it so dropping "
            f"'{case['corruption_key']}' causes a verifiably different downstream result."
        )


# ---------------------------------------------------------------------------
# Test 2 — Orchestration Correctness
# Catches: wrong sub-agent assigned, wrong task scope, wrong sequencing
# ---------------------------------------------------------------------------
ORCHESTRATION_CASES = [
    {
        "task": "Read notes/day01.md, summarise it, then write the summary to output/notes_summary.md",
        "expected_agent_sequence": ["file_reader_agent", "summariser_agent", "file_writer_agent"],
        "prohibited_agents": ["delete_agent", "search_agent"],
    },
    {
        "task": "Search for all test files and report their paths.",
        "expected_agent_sequence": ["search_agent", "reporter_agent"],
        "prohibited_agents": ["file_writer_agent", "delete_agent"],
    },
]


class TestOrchestrationCorrectness:
    @pytest.mark.parametrize("case", ORCHESTRATION_CASES)
    def test_correct_sub_agents_in_correct_order(self, case):
        result = run_multi_agent_pipeline(case["task"])
        actual_agents = [h.from_agent for h in result.handoffs]
        actual_agents.append(result.handoffs[-1].to_agent if result.handoffs else "")

        for prohibited in case["prohibited_agents"]:
            assert prohibited not in actual_agents, (
                f"Prohibited agent '{prohibited}' was invoked for task: {case['task']}"
            )

        # Subsequence check — expected agents must appear in order
        expected_iter = iter(case["expected_agent_sequence"])
        actual_iter   = iter(actual_agents)
        for expected_agent in expected_iter:
            assert any(a == expected_agent for a in actual_iter), (
                f"Expected agent '{expected_agent}' not found in correct position. "
                f"Actual sequence: {actual_agents}"
            )


# ---------------------------------------------------------------------------
# Test 3 — Failure Recovery
# Catches: infinite loops, missing fallback, wrong recovery path
#
# Design rule: each failure mode AND each expected recovery path should be
# a DISTINCT scenario.  If a recovery path has no test, it hasn't been tested.
# ---------------------------------------------------------------------------
RECOVERY_CASES = [
    {
        "description": "Empty retrieval result → agent requests clarification",
        "task": "Summarise the MCP spec document.",
        "injected_failure": {"agent": "retrieval_agent", "error": "empty_result"},
        "expected_recovery_path": "clarify",
        "must_not_loop": True,
    },
    {
        "description": "API error on first tool call → retry once then fallback",
        "task": "Read notes/day02.md and return the first heading.",
        "injected_failure": {"agent": "file_reader_agent", "error": "api_error"},
        "expected_recovery_path": "retry",
        "must_not_loop": True,
    },
    {
        "description": "Malformed tool response → clean termination with error message",
        "task": "List all files in src/",
        "injected_failure": {"agent": "file_reader_agent", "error": "malformed_response"},
        "expected_recovery_path": "terminate",
        "must_not_loop": True,
    },
]

MAX_ALLOWED_STEPS = 10   # anti-loop guard


class TestFailureRecovery:
    @pytest.mark.parametrize("case", RECOVERY_CASES)
    def test_correct_recovery_path_taken(self, case):
        result = run_multi_agent_pipeline(
            case["task"],
            injected_failure=case["injected_failure"],
        )

        assert result.recovery_path_taken == case["expected_recovery_path"], (
            f"Wrong recovery path for '{case['description']}': "
            f"expected '{case['expected_recovery_path']}', "
            f"got '{result.recovery_path_taken}'"
        )

    @pytest.mark.parametrize("case", [c for c in RECOVERY_CASES if c["must_not_loop"]])
    def test_no_infinite_loop(self, case):
        result = run_multi_agent_pipeline(
            case["task"],
            injected_failure=case["injected_failure"],
        )
        assert result.steps_taken <= MAX_ALLOWED_STEPS, (
            f"Possible infinite loop detected for '{case['description']}': "
            f"{result.steps_taken} steps taken (max allowed: {MAX_ALLOWED_STEPS})"
        )


# ---------------------------------------------------------------------------
# Test 4 — Emergent Behaviour Detection
# Catches: coordination failures invisible to individual agent tests
#
# These scenarios must be CONSTRUCTED to force coordination failures:
# ambiguous handoffs, state conflicts, tasks requiring interpretation of
# incomplete upstream output.
# ---------------------------------------------------------------------------
EMERGENT_CASES = [
    {
        "description": "Ambiguous handoff: planner says 'process the files' — no path specified",
        "task": "Process the files and generate a report.",
        # Ambiguous: downstream agent must either ask for clarification or fail gracefully
        # NOT silently proceed with a wrong assumption
        "should_not_silently_proceed": True,
        "clarification_signal": "which files",
    },
    {
        "description": "State conflict: two agents hold conflicting views of output path",
        "task": "Read notes/day01.md and write the summary to the reports directory.",
        "context": {
            "planner_output_path": "reports/day01_summary.md",
            "writer_output_path":  "output/day01_summary.md",   # conflict
        },
        # The system must detect or surface the conflict, not silently use one
        "should_not_silently_proceed": True,
        "clarification_signal": "conflict",
    },
]


class TestEmergentBehaviour:
    @pytest.mark.parametrize("case", EMERGENT_CASES)
    def test_ambiguous_coordination_does_not_silently_proceed(self, case):
        context = case.get("context")
        result  = run_multi_agent_pipeline(case["task"], context=context)

        if case["should_not_silently_proceed"]:
            # System should either: ask for clarification, surface an error, or terminate
            # It must NOT silently produce output using a wrong assumption
            signal = case["clarification_signal"].lower()
            final  = result.final_output.lower()
            proceeded_silently = (result.success is True) and (signal not in final)

            assert not proceeded_silently, (
                f"Emergent coordination failure: agent silently proceeded with ambiguous "
                f"input for scenario '{case['description']}'.\n"
                f"Expected clarification signal '{case['clarification_signal']}' "
                f"not found in output: {result.final_output}"
            )


# ---------------------------------------------------------------------------
# Test 5 — Trajectory Efficiency (multi-agent)
# Catches: over-delegation, redundant agent hops, unnecessary tool calls
# Paper KPI: Trajectory efficiency score = actual steps / optimal steps
# ---------------------------------------------------------------------------
EFFICIENCY_CASES = [
    {
        "task": "Read notes/day01.md and return the word count.",
        "optimal_steps": 2,    # read_file → count words → return
        "max_allowed_ratio": 2.0,  # no more than 2x the optimal path
    },
]


class TestTrajectoryEfficiency:
    @pytest.mark.parametrize("case", EFFICIENCY_CASES)
    def test_not_over_delegated(self, case):
        result = run_multi_agent_pipeline(case["task"])
        ratio  = result.steps_taken / case["optimal_steps"]

        assert ratio <= case["max_allowed_ratio"], (
            f"Over-delegation detected for task '{case['task']}': "
            f"{result.steps_taken} steps taken vs {case['optimal_steps']} optimal "
            f"(ratio: {ratio:.1f}x, max allowed: {case['max_allowed_ratio']}x)"
        )


# ---------------------------------------------------------------------------
# Test 6 — Full workflow via DeepEval TaskCompletionMetric
# End-to-end goal completion across multi-agent pipeline
# ---------------------------------------------------------------------------
class TestMultiAgentGoalCompletion:
    def test_full_pipeline_task_completion(self):
        task = (
            "Read notes/day01.md, extract all failure mode entries, "
            "and write a structured JSON report to output/failure_modes.json."
        )
        result = run_multi_agent_pipeline(task)
        test_case = LLMTestCase(
            input=task,
            actual_output=result.final_output,
        )
        metric = TaskCompletionMetric(threshold=0.8, verbose_mode=True)
        assert_test(test_case, [metric])
