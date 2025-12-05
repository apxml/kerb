"""Slick Toolkit - A modern toolkit library for LLM applications.

This package exposes subpackages that implement core functionality:
 - core: shared types and foundational classes used across packages
 - chunk: text chunking utilities
 - embedding: simple embedding generation and similarity helpers
 - tokenizer: token counting utilities for various LLM models
 - prompt: prompt management and engineering utilities
 - retrieval: retrieval and RAG utilities for search and context management
 - memory: conversation memory and entity tracking utilities
 - generation: unified LLM generation with multi-provider support
 - evaluation: evaluation and benchmarking utilities for LLM outputs
 - parsing: parsing and validation utilities for LLM outputs
 - document: document loading and processing utilities for various formats
 - preprocessing: text preprocessing and cleaning utilities for LLM inputs
 - agent: agent orchestration and execution patterns
 - cache: cache management utilities for LLM responses and embeddings
 - config: configuration management for models, providers, and environments
 - safety: safety and security utilities for LLM applications
 - multimodal: multi-modal processing utilities for images, audio, video, and vision models
 - context: context window management and optimization utilities
 - fine_tuning: fine-tuning utilities for preparing and managing LLM training data
 - testing: testing utilities for LLM application development
"""

__version__ = "0.1.5"

# Submodules are available for import but not eagerly loaded to avoid circular imports
# Users can import them explicitly: from kerb import chunk, agent, etc.
# or import specific items: from kerb.chunk import chunk_text

__all__ = [
    "core",
    "chunk",
    "embedding",
    "tokenizer",
    "prompt",
    "retrieval",
    "memory",
    "generation",
    "evaluation",
    "parsing",
    "document",
    "preprocessing",
    "agent",
    "cache",
    "config",
    "safety",
    "multimodal",
    "context",
    "fine_tuning",
    "testing",
    "__version__",
]


def __getattr__(name):
    """Lazy import submodules to avoid circular import issues."""
    # Handle __version__ specially
    if name == "__version__":
        return "0.1.5"
    
    # Lazy import submodules
    if name in __all__ and name != "__version__":
        import importlib
        module = importlib.import_module(f".{name}", package=__name__)
        # Cache it in the module namespace
        globals()[name] = module
        return module
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
