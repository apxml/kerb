"""Similarity and distance metrics for embeddings."""

import math
from typing import List, Tuple, Union


def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vector1 (List[float]): First embedding vector
        vector2 (List[float]): Second embedding vector

    Returns:
        float: Cosine similarity score between -1 and 1 (1 = identical)

    Examples:
        from kerb.embedding import embed
        sim = cosine_similarity(embed("hello"), embed("hi"))
    """
    if len(vector1) != len(vector2):
        raise ValueError(
            f"Vectors must have same dimensions: {len(vector1)} vs {len(vector2)}"
        )

    if not vector1 or not vector2:
        return 0.0

    dot_prod = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(x * x for x in vector1))
    magnitude2 = math.sqrt(sum(x * x for x in vector2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_prod / (magnitude1 * magnitude2)


def euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
    """Calculate Euclidean (L2) distance between two vectors.

    Args:
        vector1 (List[float]): First embedding vector
        vector2 (List[float]): Second embedding vector

    Returns:
        float: Euclidean distance (0 = identical, higher = more different)
    """
    if len(vector1) != len(vector2):
        raise ValueError(
            f"Vectors must have same dimensions: {len(vector1)} vs {len(vector2)}"
        )

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))


def manhattan_distance(vector1: List[float], vector2: List[float]) -> float:
    """Calculate Manhattan (L1) distance between two vectors.

    Args:
        vector1 (List[float]): First embedding vector
        vector2 (List[float]): Second embedding vector

    Returns:
        float: Manhattan distance
    """
    if len(vector1) != len(vector2):
        raise ValueError(
            f"Vectors must have same dimensions: {len(vector1)} vs {len(vector2)}"
        )

    return sum(abs(a - b) for a, b in zip(vector1, vector2))


def dot_product(vector1: List[float], vector2: List[float]) -> float:
    """Calculate dot product between two vectors.

    Args:
        vector1 (List[float]): First embedding vector
        vector2 (List[float]): Second embedding vector

    Returns:
        float: Dot product score
    """
    if len(vector1) != len(vector2):
        raise ValueError(
            f"Vectors must have same dimensions: {len(vector1)} vs {len(vector2)}"
        )

    return sum(a * b for a, b in zip(vector1, vector2))


def batch_similarity(
    query_vector: List[float], vectors: List[List[float]], metric: str = "cosine"
) -> List[float]:
    """Calculate similarity between a query vector and multiple vectors.

    Args:
        query_vector (List[float]): Query embedding vector
        vectors (List[List[float]]): List of embedding vectors to compare
        metric (str): Distance metric ("cosine", "euclidean", "manhattan", "dot")

    Returns:
        List[float]: Similarity/distance scores

    Examples:
        from kerb.embedding import embed, embed_batch
        query = embed("search query")
        docs = embed_batch(["doc1", "doc2", "doc3"])
        scores = batch_similarity(query, docs, metric="cosine")
    """
    metric_funcs = {
        "cosine": cosine_similarity,
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "dot": dot_product,
    }

    if metric not in metric_funcs:
        raise ValueError(
            f"Unknown metric: {metric}. Choose from {list(metric_funcs.keys())}"
        )

    func = metric_funcs[metric]
    return [func(query_vector, vec) for vec in vectors]


def top_k_similar(
    query_vector: List[float],
    vectors: List[List[float]],
    k: int = 5,
    metric: str = "cosine",
    return_scores: bool = False,
) -> Union[List[int], List[Tuple[int, float]]]:
    """Find top-k most similar vectors to a query vector.

    Args:
        query_vector (List[float]): Query embedding vector
        vectors (List[List[float]]): List of embedding vectors to search
        k (int): Number of top results to return
        metric (str): Distance metric ("cosine", "euclidean", "manhattan", "dot")
        return_scores (bool): If True, return (index, score) tuples

    Returns:
        List[int] or List[Tuple[int, float]]: Top-k indices (or index-score pairs)

    Examples:
        from kerb.embedding import embed, embed_batch
        query = embed("search query")
        docs = embed_batch(["doc1", "doc2", "doc3"])
        indices = top_k_similar(query, docs, k=2)
        # Or with scores
        results = top_k_similar(query, docs, k=2, return_scores=True)
    """
    scores = batch_similarity(query_vector, vectors, metric=metric)

    # For distance metrics, lower is better
    reverse = metric in ["cosine", "dot"]
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=reverse)

    top_k_results = indexed_scores[:k]

    if return_scores:
        return top_k_results
    else:
        return [idx for idx, _ in top_k_results]
