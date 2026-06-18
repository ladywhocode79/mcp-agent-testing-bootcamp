"""
Layer 2 — Agent Behavioural Evaluations
=========================================
Bootcamp phase: Phase 2, Days 11–12
Paper reference: Section 5.3

What this layer catches
-----------------------
- Trajectory collapse: locally valid steps → wrong outcome
- Tool selection error: right intent, wrong tool
- Parameter construction error: right tool, wrong args   ← fails INDEPENDENTLY
- Reasoning failure: correct final answer via flawed logic (false positive)
- State retention failure: agent re-asks for info already given

The key insight from the paper
-------------------------------
"A correct final answer via an incorrect trajectory is a false positive and a
production risk."  Output-only evaluation misses this entirely.  Layer 2 exists
to catch it.

Design rules
-------------
1. Tool selection and parameter construction fail INDEPENDENTLY.
   Your dataset MUST include:
     - correct tool + wrong params
     - wrong tool + valid params
   A dataset where both always succeed/fail together cannot localise regressions.

2. Reference trajectory vs rubric:
   - Reference trajectory (exact steps): use for constrained tasks with ONE known path.
   - Rubric (mandatory steps + prohibited patterns): use for tasks with multiple valid paths.
   The MCP scenario tests below use rubrics because MCP tool sequences can vary.

3. State retention scenario is only valid if context loss produces a VERIFIABLY
   different (wrong) output.  If both the state-retaining and context-dropping agent
   produce the same output, the scenario doesn't test state.

Repo placement
--------------
  mcp-agent-testing-bootcamp/
    deepeval/
      layer2_agent_trajectory/
        test_layer2_trajectory.py   ← this file
"""

import pytest
from dataclasses import dataclass
from typing import Optional
from deepeval import assert_test
from deepeval.metrics import (
    ToolCorrectnessMetric,
    TaskCompletionMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall


# ---------------------------------------------------------------------------
# Data structures for trajectory capture
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryStep:
    step_number: int
    tool_name: str
    tool_params: dict
    tool_output: str
    reasoning: str          # agent's stated reasoning for this step


@dataclass
class AgentTrajectory:
    task: str
    steps: list[TrajectoryStep]
    final_output: str
    success: bool


# ---------------------------------------------------------------------------
# Stubs — replace with your MCP agent runner
# ---------------------------------------------------------------------------
def run_mcp_agent(task: str, context: Optional[dict] = None) -> AgentTrajectory:
    """
    Replace with your real MCP agent call.
    Must return an AgentTrajectory with all steps captured.

    Example (LangChain-style pseudo):
        result = agent.invoke({"input": task, "context": context})
        steps  = [TrajectoryStep(...) for step in result.intermediate_steps]
        return AgentTrajectory(task=task, steps=steps, final_output=result.output, success=True)
    """
    raise NotImplementedError("Wire up your MCP agent runner here.")


# ---------------------------------------------------------------------------
# Test 1 — Tool Selection Accuracy (independent from parameter correctness)
# Catches: wrong tool selected (tool selection dimension only)
# ---------------------------------------------------------------------------
TOOL_SELECTION_CASES = [
    {
        "task": "Read the file at notes/day01.md and summarise it.",
        "expected_tool_sequence": ["read_file"],     # mandatory tools in order
        "prohibited_tools": ["write_file", "delete_file"],
    },
    {
        "task": "Find all Python files in the project.",
        "expected_tool_sequence": ["search_files"],
        "prohibited_tools": ["read_file"],
    },
    {
        "task": "Create a new directory called reports/ and write a file inside it.",
        "expected_tool_sequence": ["create_directory", "write_file"],  # must be in this order
        "prohibited_tools": ["delete_file"],
    },
]


class TestToolSelectionAccuracy:
    @pytest.mark.parametrize("case", TOOL_SELECTION_CASES)
    def test_correct_tools_in_correct_order(self, case):
        trajectory = run_mcp_agent(case["task"])
        actual_tools = [step.tool_name for step in trajectory.steps]

        # Check prohibited tools never appear
        for prohibited in case["prohibited_tools"]:
            assert prohibited not in actual_tools, (
                f"Prohibited tool '{prohibited}' was called during: {case['task']}"
            )

        # Check mandatory tools appear in order (subsequence check)
        expected = case["expected_tool_sequence"]
        actual_iter = iter(actual_tools)
        for expected_tool in expected:
            assert any(t == expected_tool for t in actual_iter), (
                f"Expected tool '{expected_tool}' not found in correct position. "
                f"Actual sequence: {actual_tools}"
            )


# ---------------------------------------------------------------------------
# Test 2 — Parameter Construction Accuracy (independent from tool selection)
# Catches: right tool, wrong arguments
# This is the failure mode that silently passes output-only tests.
# ---------------------------------------------------------------------------
PARAM_CONSTRUCTION_CASES = [
    {
        "task": "Read the file at notes/day01.md",
        "expected_tool": "read_file",
        "required_params": {"path": "notes/day01.md"},
        "forbidden_params": {},           # params that must NOT be present
    },
    {
        "task": "Search for all .py files under the src/ folder",
        "expected_tool": "search_files",
        "required_params": {"pattern": "*.py", "path": "src/"},
        "forbidden_params": {},
    },
    {
        "task": "Write 'hello world' to output/result.txt",
        "expected_tool": "write_file",
        "required_params": {"path": "output/result.txt", "content": "hello world"},
        "forbidden_params": {},
    },
]


class TestParameterConstruction:
    @pytest.mark.parametrize("case", PARAM_CONSTRUCTION_CASES)
    def test_correct_params_for_correct_tool(self, case):
        trajectory = run_mcp_agent(case["task"])

        # Find the step that called the expected tool
        matching_steps = [s for s in trajectory.steps if s.tool_name == case["expected_tool"]]
        assert matching_steps, (
            f"Tool '{case['expected_tool']}' was never called for task: {case['task']}"
        )

        # Check required params on the FIRST matching call
        step = matching_steps[0]
        for param_key, param_value in case["required_params"].items():
            assert param_key in step.tool_params, (
                f"Required param '{param_key}' missing in {case['expected_tool']} call. "
                f"Actual params: {step.tool_params}"
            )
            assert step.tool_params[param_key] == param_value, (
                f"Param '{param_key}' value mismatch: "
                f"expected '{param_value}', got '{step.tool_params[param_key]}'"
            )


# ---------------------------------------------------------------------------
# Test 3 — Trajectory Correctness (rubric-based, multi-step task)
# Catches: trajectory collapse, unnecessary calls, fragile reasoning paths
# Uses DeepEval's ToolCorrectnessMetric
# ---------------------------------------------------------------------------
TRAJECTORY_RUBRIC_CASES = [
    {
        "task": "Read notes/day01.md, then write a 2-sentence summary to output/summary.txt",
        "tools_called": [
            ToolCall(name="read_file",   input_parameters={"path": "notes/day01.md"}),
            ToolCall(name="write_file",  input_parameters={"path": "output/summary.txt",
                                                            "content": "..."}),
        ],
        "expected_tools": [
            ToolCall(name="read_file",   input_parameters={"path": "notes/day01.md"}),
            ToolCall(name="write_file",  input_parameters={"path": "output/summary.txt",
                                                            "content": "..."}),
        ],
    },
]


class TestTrajectoryCorrectness:
    @pytest.mark.parametrize("case", TRAJECTORY_RUBRIC_CASES)
    def test_trajectory_via_deepeval(self, case):
        trajectory = run_mcp_agent(case["task"])
        test_case = LLMTestCase(
            input=case["task"],
            actual_output=trajectory.final_output,
            tools_called=case["tools_called"],
            expected_tools=case["expected_tools"],
        )
        metric = ToolCorrectnessMetric(threshold=0.8, verbose_mode=True)
        assert_test(test_case, [metric])


# ---------------------------------------------------------------------------
# Test 4 — State Retention
# Catches: instruction drift, agent re-asking for already-provided info
#
# Design rule: the early-turn info MUST be necessary for the correct final
# answer.  A state-retaining agent and a context-dropping agent must produce
# DETECTABLY DIFFERENT outputs.  If they don't, the scenario is invalid.
# ---------------------------------------------------------------------------
STATE_RETENTION_CASES = [
    {
        "description": "Agent must use earlier-specified output path in final write step",
        "initial_context": {"preferred_output_path": "output/bootcamp_notes.txt"},
        "task": "Summarise notes/day01.md and write the summary to the path I specified earlier.",
        # If state is dropped, agent either errors or writes to a default path
        "correct_output_path": "output/bootcamp_notes.txt",
        "wrong_output_path_signal": "output/summary.txt",   # default if context dropped
    },
    {
        "description": "Agent must not re-ask for file path already given",
        "initial_context": {"file_path": "notes/day01.md"},
        "task": "Read the file we discussed and list its headings.",
        "correct_output_path": None,
        # Correct: agent reads notes/day01.md without asking
        # Wrong: agent asks "which file would you like me to read?"
        "wrong_output_path_signal": "which file",
    },
]


class TestStateRetention:
    @pytest.mark.parametrize("case", STATE_RETENTION_CASES)
    def test_context_preserved_across_steps(self, case):
        trajectory = run_mcp_agent(case["task"], context=case["initial_context"])

        # State-drop signal 1: agent re-asked for the info
        assert case["wrong_output_path_signal"].lower() not in trajectory.final_output.lower(), (
            f"State retention failure — agent produced context-drop signal: "
            f"'{case['wrong_output_path_signal']}'\n"
            f"Task: {case['task']}\nContext: {case['initial_context']}"
        )

        # State-drop signal 2: wrong output path used in write_file call
        if case["correct_output_path"]:
            write_steps = [s for s in trajectory.steps if s.tool_name == "write_file"]
            if write_steps:
                actual_path = write_steps[0].tool_params.get("path", "")
                assert actual_path == case["correct_output_path"], (
                    f"State retention failure — write_file used '{actual_path}' "
                    f"instead of context-specified '{case['correct_output_path']}'"
                )


# ---------------------------------------------------------------------------
# Test 5 — Consistency (repeated runs, same task)
# Catches: stochastic inconsistency, flaky agent behavior
#
# Paper rule: FLAKY = passes most of the time but fails occasionally.
# Flaky is not acceptable in production.  Threshold >= 0.9 (90% pass rate).
# ---------------------------------------------------------------------------
CONSISTENCY_RUNS = 5   # increase to 10+ for production confidence


class TestAgentConsistency:
    def test_trajectory_consistent_across_runs(self):
        task = "Read notes/day01.md and return the first heading."
        results = []
        for _ in range(CONSISTENCY_RUNS):
            trajectory = run_mcp_agent(task)
            # A consistent agent should always call read_file first
            first_tool = trajectory.steps[0].tool_name if trajectory.steps else None
            results.append(first_tool == "read_file")

        pass_rate = sum(results) / len(results)
        verdict = (
            "CONSISTENT_PASS"    if pass_rate >= 0.9  else
            "FLAKY"              if pass_rate >= 0.5  else
            "CONSISTENT_FAILURE"
        )

        assert verdict == "CONSISTENT_PASS", (
            f"Agent consistency verdict: {verdict} (pass rate: {pass_rate:.0%}). "
            f"Pass rate must be >= 90% for production readiness."
        )
