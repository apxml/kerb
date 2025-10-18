"""
Core types and shared data structures for the kerb library.

This package contains shared types used across multiple packages to
eliminate duplication and ensure consistency.

Import Structure:
-----------------

## Top-level imports (most common):
```python
from kerb.core import Document, Message
from kerb.core import ChainStrategy, CompressionStrategy
```

## Submodule imports (for organized access):
```python
from kerb.core import types, enums
# Then: types.DocumentFormat, enums.RerankMethod
```

## Direct submodule access (for less common items):
```python
from kerb.core.types import DocumentFormat, MessageRole
from kerb.core.enums import ReorderStrategy, ParseMode
```
"""

# Make submodules available for organized access
from . import enums, types
from .enums import ChainStrategy, ChunkingStrategy, CompressionStrategy
# Import only the most commonly used items for top-level convenience
from .types import Document, Message

__all__ = [
    # Submodules (for organized access to all types and enums)
    "types",
    "enums",
    # Most commonly used types
    "Document",
    "Message",
    # Most commonly used enums
    "ChainStrategy",
    "CompressionStrategy",
    "ChunkingStrategy",
]
