"""Data classes for memory management.

This module provides core data structures for conversation memory:
- Entity: Represents an extracted entity with metadata
- ConversationSummary: Represents a conversation summary
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class Entity:
    """Represents an extracted entity with metadata."""
    name: str
    type: str
    mentions: int = 1
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    context: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = datetime.now().isoformat()
        if self.last_seen is None:
            self.last_seen = self.first_seen
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create entity from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"Entity(name='{self.name}', type='{self.type}', mentions={self.mentions})"


@dataclass
class ConversationSummary:
    """Represents a summary of conversation history."""
    summary: str
    message_count: int
    start_time: str
    end_time: str
    key_points: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        """Create summary from dictionary."""
        return cls(**data)
