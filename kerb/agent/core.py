"""Core agent types and base classes.

This module provides the foundational types, exceptions, and base classes
for the agent system.
"""

from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .memory import AgentMemory
    from .monitoring import AgentTracer


# ============================================================================
# Custom Exceptions
# ============================================================================

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class ToolError(AgentError):
    """Error during tool execution."""
    pass


class PlanningError(AgentError):
    """Error during planning phase."""
    pass


class ExecutionError(AgentError):
    """Error during execution phase."""
    pass


# ============================================================================
# Core Data Classes
# ============================================================================

class AgentStatus(Enum):
    """Status of agent execution."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    PLANNING = "planning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentStep:
    """Represents a single step in agent execution.
    
    Attributes:
        step_number: Sequential step number
        thought: Agent's reasoning/thought process
        action: Action taken by the agent
        action_input: Input/arguments for the action
        observation: Result/observation from the action
        timestamp: When the step occurred
        metadata: Additional step metadata
    """
    step_number: int
    thought: str = ""
    action: str = ""
    action_input: Any = None
    observation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class AgentResult:
    """Result from agent execution.
    
    Attributes:
        output: Final output from the agent
        steps: List of execution steps
        status: Final status of execution
        total_time: Total execution time in seconds
        metadata: Additional result metadata
    """
    output: str
    steps: List[AgentStep]
    status: AgentStatus
    total_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'output': self.output,
            'steps': [step.to_dict() for step in self.steps],
            'status': self.status.value,
            'total_time': self.total_time,
            'metadata': self.metadata
        }


@dataclass
class AgentState:
    """Maintains agent state during execution.
    
    Attributes:
        goal: The agent's goal
        steps: History of execution steps
        current_step: Current step number
        status: Current execution status
        beliefs: Agent's current beliefs/knowledge
        context: Execution context
        max_iterations: Maximum allowed iterations
    """
    goal: str
    steps: List[AgentStep] = field(default_factory=list)
    current_step: int = 0
    status: AgentStatus = AgentStatus.IDLE
    beliefs: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 10
    
    def add_step(self, step: AgentStep) -> None:
        """Add a step to the execution history."""
        self.steps.append(step)
        self.current_step += 1
    
    def get_step_history(self, n: int = None) -> List[AgentStep]:
        """Get recent step history.
        
        Args:
            n: Number of recent steps to return. If None, returns all steps.
            
        Returns:
            List of recent steps
        """
        if n is None:
            return self.steps
        return self.steps[-n:]


# ============================================================================
# Base Agent Class
# ============================================================================

class Agent(ABC):
    """Base class for all agents.
    
    An agent perceives its environment, reasons about it, and takes actions
    to achieve its goals.
    """
    
    def __init__(
        self,
        name: str = "Agent",
        llm_func: Optional[Callable] = None,
        tools: Optional[List[Any]] = None,
        max_iterations: int = 10,
        memory: Optional["AgentMemory"] = None,
        verbose: bool = False
    ):
        """Initialize agent.
        
        Args:
            name: Name of the agent
            llm_func: Function to call LLM (should accept prompt and return response)
            tools: List of tools available to the agent
            max_iterations: Maximum number of iterations
            memory: Agent memory instance
            verbose: Whether to print execution details
        """
        self.name = name
        self.llm_func = llm_func
        self.tools = tools or []
        self.max_iterations = max_iterations
        
        # Import here to avoid circular imports
        if memory is None:
            from .memory import AgentMemory
            memory = AgentMemory()
        self.memory = memory
        
        self.verbose = verbose
        
        # Import here to avoid circular imports
        if verbose:
            from .monitoring import AgentTracer
            self.tracer = AgentTracer()
        else:
            self.tracer = None
    
    @abstractmethod
    def run(self, goal: str, context: Dict[str, Any] = None) -> AgentResult:
        """Run the agent to achieve a goal.
        
        Args:
            goal: The goal to achieve
            context: Additional context for execution
            
        Returns:
            AgentResult with output and execution steps
        """
        pass
    
    def _log(self, message: str) -> None:
        """Log a message if verbose."""
        if self.verbose:
            print(f"[{self.name}] {message}")
