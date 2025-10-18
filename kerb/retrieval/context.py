"""Context management utilities for retrieval.

This module provides functions for compressing and filtering search results.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from kerb.core.types import Document

from .structures import FilterConfig, SearchResult

if TYPE_CHECKING:
    from kerb.core.enums import CompressionStrategy


def compress_context(
    query: str,
    results: List[SearchResult],
    max_tokens: int = 2000,
    strategy: Union["CompressionStrategy", str] = "top_k",
) -> List[SearchResult]:
    """Compress retrieved context to fit within token limits.

    Args:
        query: The search query
        results: Search results to compress
        max_tokens: Maximum number of tokens (approximate)
        strategy: Compression strategy (CompressionStrategy enum or string: "top_k", "summarize", "filter", "truncate")

    Returns:
        List[SearchResult]: Compressed results

    Examples:
        >>> from kerb.core.enums import CompressionStrategy
        >>> compressed = compress_context(query, results, max_tokens=1000, strategy=CompressionStrategy.TOP_K)
    """
    from kerb.core.enums import CompressionStrategy, validate_enum_or_string

    if not results:
        return []

    # Validate and normalize strategy
    strategy_val = validate_enum_or_string(strategy, CompressionStrategy, "strategy")
    if isinstance(strategy_val, CompressionStrategy):
        strategy_str = strategy_val.value
    else:
        strategy_str = strategy_val

    # Rough token estimation (4 chars ≈ 1 token)
    chars_per_token = 4
    max_chars = max_tokens * chars_per_token

    if strategy_str == "top_k":
        # Simply take top results until token limit
        compressed = []
        total_chars = 0

        for result in results:
            doc_chars = len(result.document.content)
            if total_chars + doc_chars <= max_chars:
                compressed.append(result)
                total_chars += doc_chars
            else:
                break

        return compressed

    elif strategy_str == "summarize":
        # Smart compression: extract most relevant sentences from each document
        compressed = []
        query_terms = set(query.lower().split())
        total_chars = 0

        for result in results:
            if total_chars >= max_chars:
                break

            # Split into sentences
            sentences = [
                s.strip() + "." for s in result.document.content.split(".") if s.strip()
            ]

            # Score sentences by query term overlap
            sentence_scores = []
            for sent in sentences:
                sent_terms = set(sent.lower().split())
                overlap = len(query_terms & sent_terms)
                sentence_scores.append((sent, overlap))

            # Sort by relevance
            sentence_scores.sort(key=lambda x: x[1], reverse=True)

            # Take top sentences that fit
            compressed_content = []
            doc_chars = 0
            for sent, _ in sentence_scores:
                sent_chars = len(sent)
                if total_chars + doc_chars + sent_chars <= max_chars:
                    compressed_content.append(sent)
                    doc_chars += sent_chars
                else:
                    break

            if compressed_content:
                # Create compressed document
                compressed_doc = Document(
                    id=result.document.id,
                    content=" ".join(compressed_content),
                    metadata=result.document.metadata,
                    score=result.document.score,
                )

                compressed.append(
                    SearchResult(
                        document=compressed_doc,
                        score=result.score,
                        rank=result.rank,
                        method="compressed_smart",
                    )
                )
                total_chars += doc_chars

        return compressed

    elif strategy_str == "filter":
        # Extract only query-relevant excerpts
        compressed = []
        query_terms = set(query.lower().split())
        total_chars = 0

        for result in results:
            if total_chars >= max_chars:
                break

            content = result.document.content
            words = content.split()

            # Find windows containing query terms
            window_size = 50  # words per window
            best_window = []
            best_overlap = 0

            for i in range(0, len(words), window_size // 2):
                window = words[i : i + window_size]
                window_terms = set(w.lower() for w in window)
                overlap = len(query_terms & window_terms)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_window = window

            if best_window:
                excerpt = " ".join(best_window)
                excerpt_chars = len(excerpt)

                if total_chars + excerpt_chars <= max_chars:
                    compressed_doc = Document(
                        id=result.document.id,
                        content=excerpt,
                        metadata=result.document.metadata,
                        score=result.document.score,
                    )

                    compressed.append(
                        SearchResult(
                            document=compressed_doc,
                            score=result.score,
                            rank=result.rank,
                            method="compressed_extract",
                        )
                    )
                    total_chars += excerpt_chars

        return compressed

    elif strategy_str == "truncate":
        # Simple truncate each document to fit
        compressed = []
        total_chars = 0
        docs_to_include = min(
            len(results), max_chars // (max_chars // max(1, len(results)))
        )
        chars_per_doc = max_chars // max(1, docs_to_include)

        for i, result in enumerate(results[:docs_to_include]):
            content = result.document.content
            if len(content) > chars_per_doc:
                content = content[: chars_per_doc - 3] + "..."

            compressed_doc = Document(
                id=result.document.id,
                content=content,
                metadata=result.document.metadata,
                score=result.document.score,
            )

            compressed.append(
                SearchResult(
                    document=compressed_doc,
                    score=result.score,
                    rank=result.rank,
                    method="compressed_truncate",
                )
            )

        return compressed

    # Default: return top_k strategy
    return compress_context(query, results, max_tokens, "top_k")


def filter_results(
    results: List[SearchResult],
    min_score: Optional[float] = None,
    max_results: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    dedup_threshold: float = 0.9,
    config: Optional[FilterConfig] = None,
) -> List[SearchResult]:
    """Filter search results based on various criteria.

    Args:
        results: Search results to filter
        min_score: Minimum score threshold (ignored if config is provided)
        max_results: Maximum number of results (ignored if config is provided)
        metadata_filter: Filter by metadata fields (ignored if config is provided)
        dedup_threshold: Similarity threshold for deduplication (ignored if config is provided)
        config: FilterConfig object with all parameters (recommended)

    Returns:
        List[SearchResult]: Filtered results

    Examples:
        >>> # Using config object (recommended)
        >>> from kerb.retrieval import FilterConfig
        >>> config = FilterConfig(
        ...     min_score=0.5,
        ...     max_results=10,
        ...     metadata_filter={"category": "tech"},
        ...     dedup_threshold=0.9
        ... )
        >>> filtered = filter_results(results, config=config)

        >>> # Using individual parameters (backward compatible)
        >>> filtered = filter_results(
        ...     results,
        ...     min_score=0.5,
        ...     max_results=10,
        ...     metadata_filter={"category": "tech"}
        ... )
    """
    # Use config if provided, otherwise use individual parameters
    if config is not None:
        min_score = config.min_score
        max_results = config.max_results
        metadata_filter = config.metadata_filter
        dedup_threshold = config.dedup_threshold

    filtered = results

    # Filter by minimum score
    if min_score is not None:
        filtered = [r for r in filtered if r.score >= min_score]

    # Filter by metadata
    if metadata_filter:
        filtered = [
            r
            for r in filtered
            if all(r.document.metadata.get(k) == v for k, v in metadata_filter.items())
        ]

    # Deduplicate similar results
    if dedup_threshold < 1.0:
        deduped = []
        for result in filtered:
            is_duplicate = False
            for existing in deduped:
                # Simple content similarity check
                words1 = set(result.document.content.lower().split())
                words2 = set(existing.document.content.lower().split())

                if words1 and words2:
                    similarity = len(words1 & words2) / len(words1 | words2)
                    if similarity >= dedup_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduped.append(result)

        filtered = deduped

    # Limit results
    if max_results is not None:
        filtered = filtered[:max_results]

    # Update ranks
    for i, result in enumerate(filtered, 1):
        result.rank = i

    return filtered
