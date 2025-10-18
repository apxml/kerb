"""Retrieval utilities for LLM applications.

This module provides comprehensive retrieval tools for RAG (Retrieval-Augmented Generation):

Query Processing:
    rewrite_query() - Rewrite queries for better retrieval
    expand_query() - Expand queries into multiple variations
    generate_sub_queries() - Break complex queries into sub-queries

Search Methods:
    keyword_search() - BM25-like keyword search
    semantic_search() - Embedding-based semantic search
    hybrid_search() - Combined keyword + semantic search

Re-ranking:
    rerank_results() - Re-rank results by relevance, recency, popularity, diversity
    reciprocal_rank_fusion() - Combine multiple result lists
    diversify_results() - Apply MMR for result diversity

Context Management:
    compress_context() - Compress results to fit token limits
    filter_results() - Filter by score, metadata, deduplication

Formatting:
    format_results() - Format results for display
    results_to_context() - Convert results to LLM context string

Data Classes:
    Document - Represents a document with content and metadata (from core.types)
    SearchResult - Represents a ranked search result
    HybridSearchConfig - Configuration for hybrid search
    FilterConfig - Configuration for result filtering

Submodules:
    query - Query processing utilities
    search - Search methods (keyword, semantic, hybrid)
    reranking - Re-ranking and fusion utilities
    context - Context compression and filtering
    formatting - Result formatting utilities
    structures - Data structures and configuration classes
"""

# Import Document from core for re-export (commonly used with retrieval)
from kerb.core.types import Document

# Submodule imports for organized access
from . import context, formatting, query, reranking, search, structures
from .context import compress_context, filter_results
from .formatting import format_results, results_to_context
# Top-level imports: Most common functions from each category
from .query import expand_query, generate_sub_queries, rewrite_query
from .reranking import (diversify_results, reciprocal_rank_fusion,
                        rerank_results)
from .search import hybrid_search, keyword_search, semantic_search
# Top-level imports: Core data structures
from .structures import FilterConfig, HybridSearchConfig, SearchResult

__all__ = [
    # Core types (re-exported from core.types)
    "Document",
    # Data structures
    "SearchResult",
    "HybridSearchConfig",
    "FilterConfig",
    # Query processing
    "rewrite_query",
    "expand_query",
    "generate_sub_queries",
    # Search methods
    "keyword_search",
    "semantic_search",
    "hybrid_search",
    # Re-ranking
    "rerank_results",
    "reciprocal_rank_fusion",
    "diversify_results",
    # Context management
    "compress_context",
    "filter_results",
    # Formatting
    "format_results",
    "results_to_context",
    # Submodules
    "query",
    "search",
    "reranking",
    "context",
    "formatting",
    "structures",
]
