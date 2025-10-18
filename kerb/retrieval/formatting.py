"""Formatting utilities for search results.

This module provides functions for formatting and exporting search results.
"""

import json
from typing import List

from .structures import SearchResult


def format_results(
    results: List[SearchResult],
    format_style: str = "simple",
    include_metadata: bool = False,
) -> str:
    """Format search results for display.

    Args:
        results: Search results to format
        format_style: "simple", "detailed", or "json"
        include_metadata: Whether to include document metadata

    Returns:
        str: Formatted results

    Example:
        >>> results = keyword_search("python", docs)
        >>> print(format_results(results, format_style="detailed"))
    """
    if not results:
        return "No results found."

    if format_style == "simple":
        lines = []
        for result in results:
            lines.append(
                f"{result.rank}. [{result.score:.3f}] {result.document.content[:100]}..."
            )
        return "\n".join(lines)

    elif format_style == "detailed":
        lines = []
        for result in results:
            lines.append(
                f"--- Rank {result.rank} (Score: {result.score:.4f}, Method: {result.method}) ---"
            )
            lines.append(f"Doc ID: {result.document.id}")
            lines.append(f"Content: {result.document.content}")
            if include_metadata and result.document.metadata:
                lines.append(f"Metadata: {result.document.metadata}")
            lines.append("")
        return "\n".join(lines)

    elif format_style == "json":
        data = []
        for result in results:
            item = {
                "rank": result.rank,
                "score": result.score,
                "method": result.method,
                "document": {
                    "id": result.document.id,
                    "content": result.document.content,
                },
            }
            if include_metadata:
                item["document"]["metadata"] = result.document.metadata
            data.append(item)
        return json.dumps(data, indent=2)

    return str(results)


def results_to_context(
    results: List[SearchResult],
    separator: str = "\n\n---\n\n",
    include_source: bool = True,
) -> str:
    """Convert search results to a context string for LLM prompts.

    Args:
        results: Search results to convert
        separator: Separator between documents
        include_source: Whether to include document IDs

    Returns:
        str: Formatted context string

    Example:
        >>> results = hybrid_search(query, query_emb, docs, embeddings)
        >>> context = results_to_context(results)
        >>> prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    """
    if not results:
        return ""

    parts = []
    for result in results:
        if include_source:
            parts.append(f"[Source: {result.document.id}]\n{result.document.content}")
        else:
            parts.append(result.document.content)

    return separator.join(parts)
