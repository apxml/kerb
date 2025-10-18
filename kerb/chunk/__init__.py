"""Text chunking utilities for processing large documents."""

# Submodules
from . import code, markdown, semantic, text, utils
from .code import CodeChunker
from .markdown import MarkdownChunker
# Specialized chunker classes
from .semantic import SemanticChunker
# Common text chunking functions (kept for convenience)
# Base class and core utilities
from .text import (Chunker, RecursiveChunker, SentenceChunker, overlap_chunker,
                   paragraph_chunker, recursive_chunker,
                   sentence_window_chunker, simple_chunker,
                   sliding_window_chunker, token_based_chunker)
# Utilities
from .utils import (chunk_text, custom_chunker, merge_chunks,
                    optimize_chunk_size)

__all__ = [
    # Core
    "Chunker",
    "chunk_text",
    # Submodules
    "text",
    "semantic",
    "code",
    "markdown",
    "utils",
    # Chunker classes
    "RecursiveChunker",
    "SentenceChunker",
    "SemanticChunker",
    "CodeChunker",
    "MarkdownChunker",
    # Common text chunking functions
    "simple_chunker",
    "overlap_chunker",
    "paragraph_chunker",
    "sliding_window_chunker",
    "token_based_chunker",
    "recursive_chunker",
    "sentence_window_chunker",
    # Utilities
    "merge_chunks",
    "optimize_chunk_size",
    "custom_chunker",
]
