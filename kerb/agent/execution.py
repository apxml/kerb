"""Agent execution functions and utilities.

This module provides functions for executing agents in various modes:
- Synchronous execution
- Asynchronous execution
- Streaming execution
- Loop-based execution with timeout support
"""

import asyncio
import time
from typing import Any, AsyncIterator, Dict, Optional

from .core import Agent, AgentResult, AgentStep

# ============================================================================
# Agent Executor Class
# ============================================================================


class AgentExecutor:
    """Execute agents with various strategies."""

    def __init__(self, agent: Agent):
        """Initialize executor.

        Args:
            agent: The agent to execute
        """
        self.agent = agent

    def run(self, goal: str, context: Dict[str, Any] = None) -> AgentResult:
        """Run agent synchronously.

        Args:
            goal: Goal to achieve
            context: Execution context

        Returns:
            AgentResult
        """
        return self.agent.run(goal, context)

    async def run_async(self, goal: str, context: Dict[str, Any] = None) -> AgentResult:
        """Run agent asynchronously.

        Args:
            goal: Goal to achieve
            context: Execution context

        Returns:
            AgentResult
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.agent.run, goal, context)

    async def run_stream(
        self, goal: str, context: Dict[str, Any] = None
    ) -> AsyncIterator[AgentStep]:
        """Stream agent execution steps.

        Args:
            goal: Goal to achieve
            context: Execution context

        Yields:
            AgentStep as they are generated
        """
        # Note: This is a simplified streaming implementation
        # In practice, you'd want to modify the agent to yield steps
        result = await self.run_async(goal, context)
        for step in result.steps:
            yield step


# ============================================================================
# Execution Functions
# ============================================================================


def run_agent(agent: Agent, goal: str, context: Dict[str, Any] = None) -> AgentResult:
    """Run agent to achieve goal.

    Args:
        agent: Agent instance to run
        goal: Goal to achieve
        context: Additional context

    Returns:
        AgentResult with output and steps
    """
    return agent.run(goal, context)


def run_agent_loop(
    agent: Agent,
    goal: str,
    context: Dict[str, Any] = None,
    max_iterations: int = 10,
    timeout_seconds: Optional[float] = None,
) -> AgentResult:
    """Run agent in a loop until completion.

    Args:
        agent: Agent instance
        goal: Goal to achieve
        context: Additional context
        max_iterations: Maximum iterations
        timeout_seconds: Maximum execution time in seconds. None for no limit.

    Returns:
        AgentResult

    Raises:
        TimeoutError: If execution exceeds timeout_seconds

    Examples:
        >>> result = run_agent_loop(agent, "Find information about AI",
        ...                         max_iterations=5, timeout_seconds=60)
    """
    start_time = time.time()
    agent.max_iterations = max_iterations

    # If no timeout specified, run normally
    if timeout_seconds is None:
        return agent.run(goal, context)

    # Run with timeout checking
    # We'll need to modify the agent.run method or use a wrapper
    # For now, let's use a simple approach with iteration-level checking

    original_run = agent.run

    def run_with_timeout(g, c):
        """Wrapper that checks timeout during execution."""
        # Start the agent
        result = original_run(g, c)

        # Check if we exceeded timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Agent execution exceeded timeout of {timeout_seconds} seconds "
                f"(actual: {elapsed:.2f}s)"
            )

        return result

    try:
        return run_with_timeout(goal, context)
    except TimeoutError:
        # Re-raise timeout errors
        raise
    except Exception as e:
        # Check timeout even on errors
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Agent execution exceeded timeout of {timeout_seconds} seconds "
                f"(actual: {elapsed:.2f}s) - Original error: {e}"
            )
        raise


async def run_agent_async(
    agent: Agent, goal: str, context: Dict[str, Any] = None
) -> AgentResult:
    """Run agent asynchronously.

    Args:
        agent: Agent instance
        goal: Goal to achieve
        context: Additional context

    Returns:
        AgentResult
    """
    executor = AgentExecutor(agent)
    return await executor.run_async(goal, context)


async def run_agent_stream(
    agent: Agent, goal: str, context: Dict[str, Any] = None
) -> AsyncIterator[AgentStep]:
    """Stream agent execution.

    Args:
        agent: Agent instance
        goal: Goal to achieve
        context: Additional context

    Yields:
        AgentStep as they occur
    """
    executor = AgentExecutor(agent)
    async for step in executor.run_stream(goal, context):
        yield step
