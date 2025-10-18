"""Multi-agent coordination and teams.

This module provides abstractions for multi-agent systems including agent teams,
conversations, task delegation, and result aggregation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .core import Agent, AgentResult, AgentStatus

# ============================================================================
# Multi-Agent Classes
# ============================================================================


@dataclass
class AgentTeam:
    """Coordinate multiple agents."""

    agents: List[Agent] = field(default_factory=list)
    coordinator: Optional[Agent] = None

    def add_agent(self, agent: Agent) -> None:
        """Add agent to team."""
        self.agents.append(agent)

    def run_parallel(self, goal: str) -> List[AgentResult]:
        """Run all agents in parallel on same goal."""
        results = []
        for agent in self.agents:
            result = agent.run(goal)
            results.append(result)
        return results

    def run_sequential(self, goal: str) -> List[AgentResult]:
        """Run agents sequentially, passing output forward."""
        results = []
        current_goal = goal

        for agent in self.agents:
            result = agent.run(current_goal)
            results.append(result)
            current_goal = result.output

        return results


@dataclass
class Conversation:
    """Agent-to-agent conversation."""

    messages: List[Dict[str, str]] = field(default_factory=list)

    def add_message(self, agent_name: str, content: str) -> None:
        """Add message to conversation."""
        self.messages.append(
            {
                "agent": agent_name,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_history(self, n: int = None) -> List[Dict[str, str]]:
        """Get conversation history."""
        if n is None:
            return self.messages
        return self.messages[-n:]


# ============================================================================
# Multi-Agent Functions
# ============================================================================


def delegate_task(
    task: str, from_agent: Agent, to_agent: Agent, context: Dict[str, Any] = None
) -> AgentResult:
    """Delegate task to another agent.

    Args:
        task: Task to delegate
        from_agent: Agent delegating
        to_agent: Agent receiving task
        context: Task context

    Returns:
        Result from delegated agent
    """
    delegation_context = context or {}
    delegation_context["delegated_by"] = from_agent.name

    return to_agent.run(task, delegation_context)


def aggregate_results(results: List[AgentResult]) -> AgentResult:
    """Combine results from multiple agents.

    Args:
        results: List of agent results

    Returns:
        Aggregated result
    """
    all_steps = []
    for result in results:
        all_steps.extend(result.steps)

    combined_output = "\n\n".join([r.output for r in results])
    total_time = sum(r.total_time for r in results)

    return AgentResult(
        output=combined_output,
        steps=all_steps,
        status=AgentStatus.COMPLETED,
        total_time=total_time,
        metadata={"num_agents": len(results)},
    )
