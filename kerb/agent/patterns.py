"""Agent pattern implementations.

This module provides various agent patterns like ReAct, Plan-and-Execute,
Reflex, Chain-of-Thought, and Tree-of-Thought agents.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .core import Agent, AgentResult, AgentState, AgentStatus, AgentStep
from .planning import Planner

# ============================================================================
# Agent Pattern Implementations
# ============================================================================


class ReActAgent(Agent):
    """Reasoning and Acting (ReAct) agent.

    Follows the pattern: Thought -> Action -> Observation -> Thought -> ...
    """

    def run(self, goal: str, context: Dict[str, Any] = None) -> AgentResult:
        """Run ReAct loop.

        Args:
            goal: The goal to achieve
            context: Additional context

        Returns:
            AgentResult with final output
        """
        start_time = time.time()
        state = AgentState(
            goal=goal, max_iterations=self.max_iterations, context=context or {}
        )
        state.status = AgentStatus.THINKING

        self._log(f"Starting ReAct agent with goal: {goal}")

        for i in range(self.max_iterations):
            step = AgentStep(step_number=i + 1)

            # Thought phase
            state.status = AgentStatus.THINKING
            step.thought = self._generate_thought(state)
            self._log(f"Thought: {step.thought}")

            # Check if we're done
            if self._is_complete(step.thought):
                step.observation = "Task completed"
                state.add_step(step)
                state.status = AgentStatus.COMPLETED
                break

            # Action phase
            state.status = AgentStatus.ACTING
            step.action, step.action_input = self._select_action(state, step.thought)
            self._log(f"Action: {step.action}({step.action_input})")

            # Observation phase
            state.status = AgentStatus.OBSERVING
            step.observation = self._execute_action(step.action, step.action_input)
            self._log(f"Observation: {step.observation}")

            state.add_step(step)

        else:
            state.status = AgentStatus.COMPLETED
            self._log("Max iterations reached")

        # Extract final answer
        output = self._extract_final_answer(state)

        total_time = time.time() - start_time
        return AgentResult(
            output=output, steps=state.steps, status=state.status, total_time=total_time
        )

    def _generate_thought(self, state: AgentState) -> str:
        """Generate reasoning thought."""
        if not self.llm_func:
            return "Thinking about the next step..."

        prompt = self._create_react_prompt(state)
        response = self.llm_func(prompt)

        # Extract thought from response
        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if thought_match:
            return thought_match.group(1).strip()

        return response.strip()

    def _select_action(self, state: AgentState, thought: str) -> Tuple[str, Any]:
        """Select action based on thought."""
        if not self.llm_func:
            return "default_action", {}

        prompt = f"{thought}\n\nBased on this thought, what action should be taken?"
        if self.tools:
            prompt += f"\n\nAvailable tools: {', '.join([str(t) for t in self.tools])}"

        response = self.llm_func(prompt)

        # Parse action
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        input_match = re.search(
            r"Action Input:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL
        )

        action = action_match.group(1).strip() if action_match else "proceed"
        action_input = input_match.group(1).strip() if input_match else {}

        return action, action_input

    def _execute_action(self, action: str, action_input: Any) -> str:
        """Execute the selected action."""
        # Try to find and execute matching tool
        for tool in self.tools:
            if hasattr(tool, "name") and tool.name.lower() == action.lower():
                try:
                    if hasattr(tool, "execute"):
                        return str(tool.execute(action_input))
                    elif callable(tool):
                        return str(tool(action_input))
                except Exception as e:
                    return f"Error executing tool: {str(e)}"

        # No matching tool found
        return f"Executed {action} with input {action_input}"

    def _is_complete(self, thought: str) -> bool:
        """Check if the task is complete based on thought."""
        complete_indicators = ["final answer", "task completed", "done", "finished"]
        return any(indicator in thought.lower() for indicator in complete_indicators)

    def _extract_final_answer(self, state: AgentState) -> str:
        """Extract final answer from state."""
        if state.steps:
            last_step = state.steps[-1]
            # Look for final answer pattern
            for step in reversed(state.steps):
                if "final answer" in step.thought.lower():
                    # Extract answer after "Final Answer:"
                    match = re.search(
                        r"Final Answer:\s*(.+)", step.thought, re.IGNORECASE | re.DOTALL
                    )
                    if match:
                        return match.group(1).strip()
                    return step.observation

            return last_step.observation

        return "No answer generated"

    def _create_react_prompt(self, state: AgentState) -> str:
        """Create ReAct-style prompt."""
        prompt = f"Goal: {state.goal}\n\n"

        if state.steps:
            prompt += "Previous steps:\n"
            for step in state.steps[-3:]:  # Last 3 steps
                prompt += f"Thought: {step.thought}\n"
                prompt += f"Action: {step.action}\n"
                prompt += f"Observation: {step.observation}\n\n"

        prompt += (
            "What should you think about next? Respond with: Thought: [your reasoning]"
        )
        return prompt


class PlanAndExecuteAgent(Agent):
    """Plan-and-Execute agent pattern.

    First creates a plan, then executes each step sequentially.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner = Planner(self.llm_func)

    def run(self, goal: str, context: Dict[str, Any] = None) -> AgentResult:
        """Run plan-and-execute loop.

        Args:
            goal: The goal to achieve
            context: Additional context

        Returns:
            AgentResult with final output
        """
        start_time = time.time()
        state = AgentState(
            goal=goal, max_iterations=self.max_iterations, context=context or {}
        )

        self._log(f"Starting Plan-and-Execute agent with goal: {goal}")

        # Planning phase
        state.status = AgentStatus.PLANNING
        plan = self.planner.create_plan(goal, context)
        self._log(f"Created plan with {len(plan.steps)} steps")

        # Execution phase
        for i, planned_step in enumerate(plan.steps):
            if i >= self.max_iterations:
                break

            step = AgentStep(step_number=i + 1)
            step.thought = f"Executing planned step: {planned_step}"
            step.action = planned_step

            self._log(f"Step {i+1}: {planned_step}")

            # Execute the planned step
            state.status = AgentStatus.ACTING
            step.observation = self._execute_planned_step(planned_step, state)
            self._log(f"Result: {step.observation}")

            state.add_step(step)

            # Check if we need to replan
            if self._should_replan(step.observation):
                self._log("Replanning based on observation...")
                plan = self.planner.create_plan(goal, {"previous_steps": state.steps})

        state.status = AgentStatus.COMPLETED
        output = self._synthesize_output(state)

        total_time = time.time() - start_time
        return AgentResult(
            output=output, steps=state.steps, status=state.status, total_time=total_time
        )

    def _execute_planned_step(self, planned_step: str, state: AgentState) -> str:
        """Execute a planned step."""
        # Try to match with tools
        for tool in self.tools:
            if hasattr(tool, "name") and tool.name.lower() in planned_step.lower():
                try:
                    if hasattr(tool, "execute"):
                        return str(tool.execute(planned_step))
                    elif callable(tool):
                        return str(tool(planned_step))
                except Exception as e:
                    return f"Error: {str(e)}"

        # Use LLM if no tool matches
        if self.llm_func:
            prompt = f"Execute this step: {planned_step}\n\nContext: {state.context}"
            return self.llm_func(prompt)

        return f"Executed: {planned_step}"

    def _should_replan(self, observation: str) -> bool:
        """Check if replanning is needed."""
        replan_indicators = ["error", "failed", "cannot", "impossible"]
        return any(indicator in observation.lower() for indicator in replan_indicators)

    def _synthesize_output(self, state: AgentState) -> str:
        """Synthesize final output from all steps."""
        if not state.steps:
            return "No steps executed"

        # Combine observations
        results = [step.observation for step in state.steps]
        return "\n".join(results)


class ReflexAgent(Agent):
    """Simple reflex agent that maps observations to actions."""

    def __init__(self, *args, rules: Dict[str, Callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules = rules or {}

    def run(self, goal: str, context: Dict[str, Any] = None) -> AgentResult:
        """Run reflex agent.

        Args:
            goal: The goal/observation to react to
            context: Additional context

        Returns:
            AgentResult with reaction
        """
        start_time = time.time()
        state = AgentState(goal=goal, context=context or {})

        step = AgentStep(step_number=1)
        step.thought = "Matching observation to rules"

        # Find matching rule
        action_taken = False
        for pattern, action_func in self.rules.items():
            if re.search(pattern, goal, re.IGNORECASE):
                step.action = f"Execute rule: {pattern}"
                try:
                    step.observation = str(action_func(goal, context))
                    action_taken = True
                    break
                except Exception as e:
                    step.observation = f"Error: {str(e)}"
                    break

        if not action_taken:
            step.observation = "No matching rule found"

        state.add_step(step)
        state.status = AgentStatus.COMPLETED

        total_time = time.time() - start_time
        return AgentResult(
            output=step.observation,
            steps=state.steps,
            status=state.status,
            total_time=total_time,
        )


class ChainOfThoughtAgent(Agent):
    """Chain-of-Thought agent with explicit reasoning steps."""

    def run(self, goal: str, context: Dict[str, Any] = None) -> AgentResult:
        """Run chain-of-thought reasoning.

        Args:
            goal: The problem to solve
            context: Additional context

        Returns:
            AgentResult with solution
        """
        start_time = time.time()
        state = AgentState(goal=goal, context=context or {})

        self._log(f"Starting Chain-of-Thought reasoning for: {goal}")

        # Generate reasoning chain
        if self.llm_func:
            prompt = self._create_cot_prompt(goal, context)
            response = self.llm_func(prompt)

            # Parse reasoning steps
            steps_text = self._parse_reasoning_steps(response)

            for i, step_text in enumerate(steps_text):
                step = AgentStep(step_number=i + 1)
                step.thought = step_text
                step.observation = "Reasoning step completed"
                state.add_step(step)
                self._log(f"Step {i+1}: {step_text}")

            # Extract final answer
            output = self._extract_answer(response)
        else:
            output = "No LLM function provided for reasoning"

        state.status = AgentStatus.COMPLETED
        total_time = time.time() - start_time

        return AgentResult(
            output=output, steps=state.steps, status=state.status, total_time=total_time
        )

    def _create_cot_prompt(self, goal: str, context: Dict[str, Any] = None) -> str:
        """Create chain-of-thought prompt."""
        prompt = f"Let's solve this step by step.\n\nProblem: {goal}\n\n"
        if context:
            prompt += f"Context: {json.dumps(context, indent=2)}\n\n"
        prompt += (
            "Please think through this carefully, showing your reasoning at each step."
        )
        return prompt

    def _parse_reasoning_steps(self, response: str) -> List[str]:
        """Parse individual reasoning steps."""
        # Split by common step markers
        lines = response.split("\n")
        steps = []
        current_step = []

        for line in lines:
            line = line.strip()
            # Check if line starts with step marker
            if re.match(
                r"^(Step \d+|First|Second|Third|Next|Finally|Therefore)",
                line,
                re.IGNORECASE,
            ):
                if current_step:
                    steps.append(" ".join(current_step))
                current_step = [line]
            elif line:
                current_step.append(line)

        if current_step:
            steps.append(" ".join(current_step))

        return steps if steps else [response]

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response."""
        # Look for answer patterns
        patterns = [
            r"(?:Final Answer|Answer|Therefore|Conclusion):\s*(.+?)(?:\n\n|$)",
            r"(?:The answer is|We get|This gives us)\s+(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Return last non-empty line as answer
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        return lines[-1] if lines else response


class TreeOfThoughtAgent(Agent):
    """Tree-of-Thought agent that explores multiple reasoning paths."""

    def __init__(self, *args, breadth: int = 3, depth: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.breadth = breadth  # Number of thoughts to explore at each step
        self.depth = depth  # Depth of exploration

    def run(self, goal: str, context: Dict[str, Any] = None) -> AgentResult:
        """Run tree-of-thought exploration.

        Args:
            goal: The problem to solve
            context: Additional context

        Returns:
            AgentResult with best solution found
        """
        start_time = time.time()
        state = AgentState(goal=goal, context=context or {})

        self._log(f"Starting Tree-of-Thought exploration for: {goal}")

        # Generate multiple reasoning paths
        paths = self._explore_paths(goal, context)

        # Evaluate and select best path
        best_path = self._select_best_path(paths)

        # Record steps from best path
        for i, thought in enumerate(best_path["thoughts"]):
            step = AgentStep(step_number=i + 1)
            step.thought = thought
            step.observation = "Path exploration"
            state.add_step(step)

        state.status = AgentStatus.COMPLETED
        total_time = time.time() - start_time

        return AgentResult(
            output=best_path.get("answer", "No answer found"),
            steps=state.steps,
            status=state.status,
            total_time=total_time,
            metadata={"all_paths": len(paths), "best_score": best_path.get("score", 0)},
        )

    def _explore_paths(
        self, goal: str, context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Explore multiple reasoning paths."""
        paths = []

        if not self.llm_func:
            return [{"thoughts": ["No LLM provided"], "answer": "N/A", "score": 0}]

        # Generate initial thoughts
        initial_thoughts = self._generate_thoughts(goal, context, count=self.breadth)

        for thought in initial_thoughts:
            path = self._explore_path([thought], goal, context, depth=self.depth - 1)
            paths.append(path)

        return paths

    def _explore_path(
        self, current_path: List[str], goal: str, context: Dict[str, Any], depth: int
    ) -> Dict[str, Any]:
        """Recursively explore a reasoning path."""
        if depth == 0:
            # Reach leaf, evaluate path
            answer = self._generate_answer(current_path, goal)
            score = self._evaluate_path(current_path, answer, goal)
            return {"thoughts": current_path, "answer": answer, "score": score}

        # Generate next thoughts
        next_thoughts = self._generate_thoughts(
            goal, context, previous_thoughts=current_path, count=self.breadth
        )

        # Explore each branch (simplified: take first)
        best_branch = None
        best_score = -float("inf")

        for thought in next_thoughts[:1]:  # Simplified to avoid explosion
            new_path = current_path + [thought]
            branch = self._explore_path(new_path, goal, context, depth - 1)
            if branch["score"] > best_score:
                best_score = branch["score"]
                best_branch = branch

        return best_branch or {"thoughts": current_path, "answer": "N/A", "score": 0}

    def _generate_thoughts(
        self,
        goal: str,
        context: Dict[str, Any] = None,
        previous_thoughts: List[str] = None,
        count: int = 3,
    ) -> List[str]:
        """Generate multiple possible thoughts."""
        if not self.llm_func:
            return [f"Thought {i+1}" for i in range(count)]

        prompt = f"Problem: {goal}\n\n"
        if previous_thoughts:
            prompt += "Previous reasoning:\n"
            for i, t in enumerate(previous_thoughts):
                prompt += f"{i+1}. {t}\n"
            prompt += "\n"

        prompt += (
            f"Generate {count} different next reasoning steps to solve this problem."
        )

        response = self.llm_func(prompt)

        # Parse thoughts
        thoughts = []
        for line in response.split("\n"):
            line = line.strip()
            if line and len(thoughts) < count:
                # Remove numbering
                cleaned = re.sub(r"^\d+[.)]\s*", "", line)
                if cleaned:
                    thoughts.append(cleaned)

        return thoughts[:count] if thoughts else ["Continue reasoning"]

    def _generate_answer(self, path: List[str], goal: str) -> str:
        """Generate answer from reasoning path."""
        if not self.llm_func:
            return "Answer based on reasoning path"

        prompt = f"Problem: {goal}\n\nReasoning path:\n"
        for i, thought in enumerate(path):
            prompt += f"{i+1}. {thought}\n"
        prompt += "\nBased on this reasoning, what is the final answer?"

        return self.llm_func(prompt)

    def _evaluate_path(self, path: List[str], answer: str, goal: str) -> float:
        """Evaluate quality of reasoning path."""
        # Simple heuristic: longer paths with more detail score higher
        score = len(path) * 0.3
        score += len(answer) * 0.001

        # Bonus for coherence (simplistic check)
        if answer and answer.lower() not in ["n/a", "none", "unknown"]:
            score += 1.0

        return score

    def _select_best_path(self, paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best path from explored paths."""
        if not paths:
            return {"thoughts": [], "answer": "No paths explored", "score": 0}

        return max(paths, key=lambda p: p["score"])
