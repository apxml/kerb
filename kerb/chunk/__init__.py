"""Text chunking utilities for processing large documents."""

# Submodules
from . import text, semantic, code, markdown, utils

# Base class and core utilities
from .text import Chunker, RecursiveChunker, SentenceChunker
from .utils import chunk_text

# Specialized chunker classes
from .semantic import SemanticChunker
from .code import CodeChunker
from .markdown import MarkdownChunker

# Common text chunking functions (kept for convenience)
from .text import (
    simple_chunker,
    overlap_chunker,
    paragraph_chunker,
    sliding_window_chunker,
    token_based_chunker,
    recursive_chunker,
    sentence_window_chunker,
)

# Utilities
from .utils import merge_chunks, optimize_chunk_size, custom_chunker

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
