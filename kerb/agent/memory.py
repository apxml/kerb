"""Agent memory management.

This module provides memory abstractions for agents including working memory,
episodic memory, and state persistence.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from .core import AgentState, AgentStatus, AgentStep

# ============================================================================
# Memory Classes
# ============================================================================


@dataclass
class AgentMemory:
    """Base memory for agents."""

    short_term: List[Dict[str, Any]] = field(default_factory=list)
    long_term: Dict[str, Any] = field(default_factory=dict)

    def remember(self, key: str, value: Any) -> None:
        """Store information in long-term memory."""
        self.long_term[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        """Retrieve information from long-term memory."""
        return self.long_term.get(key, default)

    def add_to_short_term(self, item: Dict[str, Any]) -> None:
        """Add item to short-term memory."""
        self.short_term.append(item)

    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term.clear()


@dataclass
class WorkingMemory:
    """Short-term working memory for current task."""

    items: List[Any] = field(default_factory=list)
    capacity: int = 7  # Miller's law: 7±2 items

    def add(self, item: Any) -> None:
        """Add item to working memory, removing oldest if at capacity."""
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)

    def clear(self) -> None:
        """Clear working memory."""
        self.items.clear()


@dataclass
class EpisodicMemory:
    """Long-term episodic memory for agent experiences."""

    episodes: List[Dict[str, Any]] = field(default_factory=list)

    def add_episode(self, episode: Dict[str, Any]) -> None:
        """Add an episode to memory."""
        episode["timestamp"] = datetime.now().isoformat()
        self.episodes.append(episode)

    def get_recent_episodes(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get n most recent episodes."""
        return self.episodes[-n:]


# ============================================================================
# State Persistence Functions
# ============================================================================


def save_agent_state(state: AgentState, filepath: str) -> None:
    """Save agent state to file.

    Args:
        state: Agent state to save
        filepath: Path to save file
    """
    state_dict = {
        "goal": state.goal,
        "steps": [step.to_dict() for step in state.steps],
        "current_step": state.current_step,
        "status": state.status.value,
        "beliefs": state.beliefs,
        "context": state.context,
        "max_iterations": state.max_iterations,
    }

    with open(filepath, "w") as f:
        json.dump(state_dict, f, indent=2)


def load_agent_state(filepath: str) -> AgentState:
    """Load agent state from file.

    Args:
        filepath: Path to state file

    Returns:
        Loaded agent state
    """
    with open(filepath, "r") as f:
        state_dict = json.load(f)

    steps = []
    for step_dict in state_dict.get("steps", []):
        step = AgentStep(
            step_number=step_dict["step_number"],
            thought=step_dict.get("thought", ""),
            action=step_dict.get("action", ""),
            action_input=step_dict.get("action_input"),
            observation=step_dict.get("observation", ""),
            metadata=step_dict.get("metadata", {}),
        )
        steps.append(step)

    state = AgentState(
        goal=state_dict["goal"],
        steps=steps,
        current_step=state_dict["current_step"],
        status=AgentStatus(state_dict["status"]),
        beliefs=state_dict.get("beliefs", {}),
        context=state_dict.get("context", {}),
        max_iterations=state_dict.get("max_iterations", 10),
    )

    return state
