"""Agent orchestration utilities for LLM applications.

This module provides comprehensive agent frameworks and execution patterns.

Import Structure:
-----------------

## Top-level imports (most common):
```python
from kerb.agent import Agent, run_agent, AgentResult
from kerb.agent import Tool, create_tool
```

## Submodule imports (for organized access):
```python
from kerb.agent import patterns, tools, reasoning, memory, planning, teams, monitoring, execution
# Then: patterns.ReActAgent, tools.ToolRegistry, reasoning.Chain, execution.run_agent_loop
```

## Direct submodule access:
```python
from kerb.agent.patterns import ReActAgent, PlanAndExecuteAgent
from kerb.agent.tools import validate_tool_call, global_tool_registry
from kerb.agent.reasoning import Chain, SequentialChain, ParallelChain
from kerb.agent.memory import AgentMemory, WorkingMemory
from kerb.agent.planning import Planner, create_plan
from kerb.agent.teams import AgentTeam, delegate_task
from kerb.agent.monitoring import AgentTracer, evaluate_agent
from kerb.agent.execution import run_agent_loop, run_agent_async
```

Key Components:
---------------
- **Core**: Agent, AgentResult, AgentStep, AgentState, AgentStatus
- **Execution**: run_agent, run_agent_loop, run_agent_async, run_agent_stream, AgentExecutor
- **Patterns**: ReActAgent, PlanAndExecuteAgent, ChainOfThoughtAgent, TreeOfThoughtAgent, ReflexAgent
- **Tools**: Tool, ToolRegistry, create_tool, register_tool, execute_tool, global_tool_registry
- **Reasoning**: Chain, SequentialChain, ParallelChain, ConditionalChain, LoopChain
- **Memory**: AgentMemory, WorkingMemory, EpisodicMemory, save_agent_state, load_agent_state
- **Planning**: Planner, Plan, create_plan, execute_plan, replan, validate_plan
- **Teams**: AgentTeam, Conversation, delegate_task, aggregate_results
- **Monitoring**: AgentTracer, evaluate_agent, benchmark_agent, compare_agents
"""

# Make submodules available for organized access
from . import (execution, memory, monitoring, patterns, planning, reasoning,
               teams, tools)
# Import only the most commonly used items for top-level convenience
from .core import Agent, AgentResult
from .execution import run_agent
from .tools import Tool, create_tool

__all__ = [
    # Submodules (for organized access to all functionality)
    "patterns",
    "tools",
    "reasoning",
    "memory",
    "planning",
    "teams",
    "monitoring",
    "execution",
    # Most commonly used items
    "Agent",
    "run_agent",
    "AgentResult",
    "Tool",
    "create_tool",
]
