"""Analysis functions for embeddings."""

from typing import List

from .metrics import batch_similarity, cosine_similarity


def embedding_dimension(vector: List[float]) -> int:
    """Get the dimension of an embedding vector.

    Args:
        vector (List[float]): Embedding vector

    Returns:
        int: Vector dimension
    """
    return len(vector)


def pairwise_similarities(
    vectors: List[List[float]], metric: str = "cosine"
) -> List[List[float]]:
    """Calculate pairwise similarities between all vectors.

    Returns a similarity matrix where element [i][j] is the similarity
    between vectors[i] and vectors[j].

    Args:
        vectors (List[List[float]]): List of embedding vectors
        metric (str): Distance metric to use

    Returns:
        List[List[float]]: N x N similarity matrix

    Examples:
        from kerb.embedding import embed_batch
        docs = embed_batch(["doc1", "doc2", "doc3"])
        sim_matrix = pairwise_similarities(docs)
    """
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            scores = batch_similarity(vectors[i], [vectors[j]], metric=metric)
            score = scores[0]
            matrix[i][j] = score
            matrix[j][i] = score

    return matrix


def cluster_embeddings(
    vectors: List[List[float]], threshold: float = 0.8
) -> List[List[int]]:
    """Simple clustering of embeddings based on similarity threshold.

    Groups embeddings that are similar above the threshold.

    Args:
        vectors (List[List[float]]): List of embedding vectors
        threshold (float): Similarity threshold for clustering (0-1)

    Returns:
        List[List[int]]: List of clusters (each cluster is a list of indices)

    Examples:
        from kerb.embedding import embed_batch
        docs = embed_batch(["doc1", "doc2 similar to 1", "doc3 different"])
        clusters = cluster_embeddings(docs, threshold=0.7)
    """
    n = len(vectors)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        cluster = [i]
        visited[i] = True

        for j in range(i + 1, n):
            if visited[j]:
                continue

            sim = cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                cluster.append(j)
                visited[j] = True

        clusters.append(cluster)

    return clusters
