"""Re-ranking utilities for search results.

This module provides functions for re-ranking and fusing search results.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

from kerb.core.types import Document

from .structures import SearchResult

if TYPE_CHECKING:
    from kerb.core.enums import RerankMethod


def rerank_results(
    query: str,
    results: List[SearchResult],
    method: Union["RerankMethod", str] = "relevance",
    top_k: Optional[int] = None,
    scorer: Optional[Callable[[str, Document], float]] = None,
) -> List[SearchResult]:
    """Re-rank search results using additional signals.

    Args:
        query: The search query
        results: Initial search results
        method: Re-ranking method (RerankMethod enum or string: "relevance", "diversity", "mmr", "cross_encoder", "llm")
        top_k: Number of top results to return after re-ranking
        scorer: Custom scoring function for method="custom"

    Returns:
        List[SearchResult]: Re-ranked search results

    Examples:
        >>> # Using enum (recommended)
        >>> from kerb.core.enums import RerankMethod
        >>> results = keyword_search("python", docs)
        >>> reranked = rerank_results("python", results, method=RerankMethod.MMR)

        >>> # Using string (for backward compatibility)
        >>> results = keyword_search("python", docs)
        >>> reranked = rerank_results("python", results, method="relevance")
    """
    from kerb.core.enums import RerankMethod, validate_enum_or_string

    if not results:
        return []

    # Validate and normalize method
    method_val = validate_enum_or_string(method, RerankMethod, "method")
    if isinstance(method_val, RerankMethod):
        method_str = method_val.value
    else:
        method_str = method_val

    reranked = []

    if method_str == "relevance":
        # Score based on query term frequency in document
        query_terms = set(query.lower().split())
        for result in results:
            doc_terms = result.document.content.lower().split()
            relevance = sum(1 for term in doc_terms if term in query_terms)
            new_score = result.score * (1 + relevance * 0.1)

            reranked.append(
                SearchResult(
                    document=result.document,
                    score=new_score,
                    rank=result.rank,
                    method="reranked_relevance",
                )
            )

    elif method_str == "recency":
        # Boost recent documents (requires "date" in metadata)
        for result in results:
            recency_boost = 1.0
            if "date" in result.document.metadata:
                # Simple recency boost (can be enhanced with actual date parsing)
                recency_boost = 1.2
            new_score = result.score * recency_boost

            reranked.append(
                SearchResult(
                    document=result.document,
                    score=new_score,
                    rank=result.rank,
                    method="reranked_recency",
                )
            )

    elif method_str in ("popularity", "diversity"):
        # Boost popular documents (requires "views" or "likes" in metadata)
        for result in results:
            popularity = result.document.metadata.get("views", 0)
            popularity = result.document.metadata.get("likes", popularity)
            popularity_boost = 1.0 + (popularity * 0.001)  # Small boost per view/like
            new_score = result.score * popularity_boost

            reranked.append(
                SearchResult(
                    document=result.document,
                    score=new_score,
                    rank=result.rank,
                    method="reranked_popularity",
                )
            )

    elif method == "diversity":
        # Maximal Marginal Relevance (MMR) for diversity
        # Select documents that are relevant but diverse from already selected
        if not results:
            return []

        lambda_param = 0.5  # Balance between relevance and diversity
        selected = [results[0]]  # Start with top result
        reranked.append(
            SearchResult(
                document=results[0].document,
                score=results[0].score,
                rank=1,
                method="reranked_diversity",
            )
        )

        remaining = results[1:]

        while remaining and len(reranked) < len(results):
            best_mmr = -float("inf")
            best_result = None

            for result in remaining:
                # Relevance score
                relevance = result.score

                # Similarity to already selected (simplified - uses content overlap)
                max_similarity = 0.0
                for selected_result in reranked:
                    # Simple word overlap similarity
                    words1 = set(result.document.content.lower().split())
                    words2 = set(selected_result.document.content.lower().split())
                    if words1 and words2:
                        similarity = len(words1 & words2) / len(words1 | words2)
                        max_similarity = max(max_similarity, similarity)

                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_result = result

            if best_result:
                reranked.append(
                    SearchResult(
                        document=best_result.document,
                        score=best_mmr,
                        rank=len(reranked) + 1,
                        method="reranked_diversity",
                    )
                )
                remaining.remove(best_result)
            else:
                break

        # Update ranks
        for i, result in enumerate(reranked, 1):
            result.rank = i

        return reranked[:top_k] if top_k else reranked

    elif method == "custom" and scorer:
        # Use custom scoring function
        for result in results:
            new_score = scorer(query, result.document)
            reranked.append(
                SearchResult(
                    document=result.document,
                    score=new_score,
                    rank=result.rank,
                    method="reranked_custom",
                )
            )
    else:
        reranked = results

    # Sort by new scores
    if method != "diversity":  # Diversity already sorted
        reranked.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(reranked, 1):
            result.rank = i

    return reranked[:top_k] if top_k else reranked


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]], k: int = 60, top_k: Optional[int] = None
) -> List[SearchResult]:
    """Combine multiple result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: Multiple lists of search results to fuse
        k: RRF constant (typically 60)
        top_k: Number of top results to return

    Returns:
        List[SearchResult]: Fused and ranked results

    Example:
        >>> results1 = keyword_search("python", docs)
        >>> results2 = semantic_search(embed("python"), docs, embeddings)
        >>> fused = reciprocal_rank_fusion([results1, results2])
    """
    # Collect all unique documents
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for result_list in result_lists:
        for result in result_list:
            doc_id = result.document.id
            doc_map[doc_id] = result.document

            # RRF formula: score = 1 / (k + rank)
            rrf_score = 1.0 / (k + result.rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score

    # Sort by RRF score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Create fused results
    results = []
    for rank, (doc_id, score) in enumerate(sorted_docs, 1):
        results.append(
            SearchResult(
                document=doc_map[doc_id], score=score, rank=rank, method="rrf_fused"
            )
        )

    return results[:top_k] if top_k else results


def diversify_results(
    results: List[SearchResult], max_results: int = 10, diversity_factor: float = 0.5
) -> List[SearchResult]:
    """Diversify results using Maximal Marginal Relevance (MMR).

    Args:
        results: Search results to diversify
        max_results: Number of results to return
        diversity_factor: Balance between relevance (0) and diversity (1)

    Returns:
        List[SearchResult]: Diversified results

    Example:
        >>> results = semantic_search(query_emb, docs, embeddings, top_k=50)
        >>> diverse = diversify_results(results, max_results=10, diversity_factor=0.7)
    """
    if not results or len(results) <= max_results:
        return results

    # Use rerank_results with diversity method
    return rerank_results(
        query="",  # Not needed for diversity
        results=results,
        method="diversity",
        top_k=max_results,
    )
