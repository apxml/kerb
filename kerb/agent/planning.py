"""Agent planning and plan execution.

This module provides planning abstractions for agents including plan creation,
execution, validation, and replanning.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ============================================================================
# Planning Classes
# ============================================================================


@dataclass
class Plan:
    """Structured plan with steps.

    Attributes:
        goal: The goal to achieve
        steps: List of planned steps
        current_step: Index of current step
        completed: Whether plan is completed
        metadata: Additional plan metadata
    """

    goal: str
    steps: List[str]
    current_step: int = 0
    completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def next_step(self) -> Optional[str]:
        """Get next step in the plan."""
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            self.current_step += 1
            return step
        return None

    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return self.current_step >= len(self.steps)


class Planner:
    """Generate plans from goals."""

    def __init__(self, llm_func: Optional[Callable] = None):
        """Initialize planner.

        Args:
            llm_func: Function to call LLM for planning. Should accept a prompt
                and return a string response.
        """
        self.llm_func = llm_func

    def create_plan(self, goal: str, context: Dict[str, Any] = None) -> Plan:
        """Create a plan for achieving a goal.

        Args:
            goal: The goal to achieve
            context: Additional context for planning

        Returns:
            Plan with steps to achieve the goal
        """
        if self.llm_func:
            # Use LLM to generate plan
            prompt = self._create_planning_prompt(goal, context)
            response = self.llm_func(prompt)
            steps = self._parse_plan_steps(response)
        else:
            # Simple default planning
            steps = [f"Step towards: {goal}"]

        return Plan(goal=goal, steps=steps, metadata={"context": context or {}})

    def _create_planning_prompt(self, goal: str, context: Dict[str, Any] = None) -> str:
        """Create prompt for LLM-based planning."""
        prompt = f"Create a step-by-step plan to achieve the following goal:\n\nGoal: {goal}\n\n"

        if context:
            prompt += f"Context: {json.dumps(context, indent=2)}\n\n"

        prompt += "Provide a numbered list of concrete steps to achieve this goal."
        return prompt

    def _parse_plan_steps(self, response: str) -> List[str]:
        """Parse steps from LLM response."""
        steps = []
        # Look for numbered steps
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Match patterns like "1.", "1)", "Step 1:", etc.
            match = re.match(r"^(?:\d+[.):]|\*|\-)\s*(.+)$", line)
            if match:
                steps.append(match.group(1).strip())
            elif line and not steps:
                # If no numbering found, treat non-empty lines as steps
                steps.append(line)

        return steps if steps else [response.strip()]


# ============================================================================
# Planning Functions
# ============================================================================


def create_plan(
    goal: str, context: Dict[str, Any] = None, llm_func: Optional[Callable] = None
) -> Plan:
    """Create a plan for achieving a goal.

    Args:
        goal: Goal to achieve
        context: Additional context
        llm_func: LLM function for planning

    Returns:
        Plan with steps
    """
    planner = Planner(llm_func)
    return planner.create_plan(goal, context)


def execute_plan(
    plan: Plan, executor_func: Optional[Callable[[str], str]] = None
) -> List[str]:
    """Execute a plan step by step.

    Args:
        plan: Plan to execute
        executor_func: Function to execute each step

    Returns:
        List of results from each step
    """
    results = []
    while not plan.is_complete():
        step = plan.next_step()
        if step:
            if executor_func:
                result = executor_func(step)
            else:
                result = f"Executed: {step}"
            results.append(result)

    return results


def replan(
    original_plan: Plan, feedback: str, llm_func: Optional[Callable] = None
) -> Plan:
    """Update plan based on feedback.

    Args:
        original_plan: Original plan
        feedback: Feedback on execution
        llm_func: LLM function for replanning

    Returns:
        Updated plan
    """
    context = {
        "original_goal": original_plan.goal,
        "completed_steps": original_plan.steps[: original_plan.current_step],
        "feedback": feedback,
    }

    planner = Planner(llm_func)
    return planner.create_plan(original_plan.goal, context)


def validate_plan(plan: Plan) -> Tuple[bool, List[str]]:
    """Validate plan feasibility.

    Args:
        plan: Plan to validate

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    if not plan.steps:
        issues.append("Plan has no steps")

    if len(plan.steps) > 50:
        issues.append("Plan has too many steps (>50)")

    # Check for circular dependencies (simple check)
    step_texts = [s.lower() for s in plan.steps]
    if len(step_texts) != len(set(step_texts)):
        issues.append("Plan has duplicate steps")

    return len(issues) == 0, issues
