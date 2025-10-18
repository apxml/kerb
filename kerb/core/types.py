"""
Core type definitions for the kerb library.

This module contains the fundamental data classes used across multiple packages
to ensure consistency and eliminate duplication.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from datetime import datetime


# ============================================================================
# Document Types
# ============================================================================

class DocumentFormat(Enum):
    """Supported document formats."""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    HTML = "html"
    MARKDOWN = "markdown"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    RTF = "rtf"
    ODT = "odt"
    EPUB = "epub"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """
    Universal document representation across the toolkit.
    
    Consolidates the Document classes from document/ and retrieval/ packages
    to provide a single, consistent document representation.
    
    Attributes:
        content: The text content of the document
        metadata: Additional metadata about the document
        id: Optional unique identifier for the document
        source: Optional source path or URL where document was loaded from
        format: Document format (defaults to UNKNOWN)
        score: Relevance score (used in retrieval contexts, defaults to 0.0)
        page_content: Optional list of content per page (for multi-page documents)
    
    Examples:
        >>> # Simple document
        >>> doc = Document(content="Hello, world!")
        
        >>> # Document with metadata
        >>> doc = Document(
        ...     content="Important document",
        ...     metadata={"author": "John", "created": "2025-01-01"},
        ...     source="doc.txt"
        ... )
        
        >>> # Retrieval result with score
        >>> doc = Document(
        ...     id="doc_123",
        ...     content="Relevant content",
        ...     score=0.95
        ... )
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    source: Optional[str] = None
    format: DocumentFormat = DocumentFormat.UNKNOWN
    score: float = 0.0
    page_content: Optional[List[str]] = None
    
    def __len__(self) -> int:
        """Return the length of the document content."""
        return len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary.
        
        Returns:
            Dictionary representation of the document
        """
        return {
            "content": self.content,
            "metadata": self.metadata,
            "id": self.id,
            "source": self.source,
            "format": self.format.value if isinstance(self.format, DocumentFormat) else self.format,
            "score": self.score,
            "page_content": self.page_content,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Create document from dictionary.
        
        Args:
            data: Dictionary with document data
            
        Returns:
            New Document instance
        """
        # Convert format string back to enum if present
        if "format" in data and isinstance(data["format"], str):
            try:
                data["format"] = DocumentFormat(data["format"])
            except ValueError:
                data["format"] = DocumentFormat.UNKNOWN
        return cls(**data)
    
    def __repr__(self) -> str:
        """String representation of the document."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        if self.id:
            return f"Document(id='{self.id}', score={self.score:.4f}, content='{content_preview}')"
        return f"Document(content='{content_preview}', source='{self.source}')"


# ============================================================================
# Message Types
# ============================================================================

class MessageRole(Enum):
    """Message roles in conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    """
    Universal message representation for conversations.
    
    Consolidates the Message classes from generation/ and memory/ packages
    to provide a single, consistent message representation.
    
    Attributes:
        role: The role of the message sender (system, user, assistant, etc.)
        content: The message content
        timestamp: Optional ISO format timestamp (auto-generated if not provided)
        metadata: Additional metadata about the message
        name: Optional name for the message sender (used in function calling)
        function_call: Optional function call information (legacy)
        tool_calls: Optional list of tool calls
    
    Examples:
        >>> # Simple user message
        >>> msg = Message(role="user", content="Hello!")
        
        >>> # System message with enum role
        >>> msg = Message(
        ...     role=MessageRole.SYSTEM,
        ...     content="You are a helpful assistant"
        ... )
        
        >>> # Message with metadata
        >>> msg = Message(
        ...     role="assistant",
        ...     content="Here's the answer",
        ...     metadata={"model": "gpt-4", "tokens": 150}
        ... )
    """
    role: Union[MessageRole, str]
    content: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        """Auto-generate timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary format.
        
        Returns:
            Dictionary representation suitable for API calls
        """
        role_str = self.role.value if isinstance(self.role, MessageRole) else self.role
        result = {
            "role": role_str,
            "content": self.content,
        }
        
        # Add optional fields only if present
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.metadata:
            result["metadata"] = self.metadata
        if self.name:
            result["name"] = self.name
        if self.function_call:
            result["function_call"] = self.function_call
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """
        Create message from dictionary.
        
        Args:
            data: Dictionary with message data
            
        Returns:
            New Message instance
        """
        # Convert role string to enum if it matches a known role
        if "role" in data and isinstance(data["role"], str):
            try:
                data["role"] = MessageRole(data["role"])
            except ValueError:
                # Keep as string if not a standard role
                pass
        return cls(**data)
    
    def __repr__(self) -> str:
        """String representation of the message."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        role_str = self.role.value if isinstance(self.role, MessageRole) else self.role
        return f"Message(role='{role_str}', content='{content_preview}')"


__all__ = [
    "Document",
    "DocumentFormat",
    "Message",
    "MessageRole",
]
