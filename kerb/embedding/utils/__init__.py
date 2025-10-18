"""Utility functions for embeddings."""

from .metrics import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
    dot_product,
    batch_similarity,
    top_k_similar,
)
from .operations import (
    normalize_vector,
    vector_magnitude,
    mean_pooling,
    weighted_mean_pooling,
    max_pooling,
)
from .analysis import (
    embedding_dimension,
    pairwise_similarities,
    cluster_embeddings,
)

__all__ = [
    # Metrics
    "cosine_similarity",
    "euclidean_distance",
    "manhattan_distance",
    "dot_product",
    "batch_similarity",
    "top_k_similar",
    # Operations
    "normalize_vector",
    "vector_magnitude",
    "mean_pooling",
    "weighted_mean_pooling",
    "max_pooling",
    # Analysis
    "embedding_dimension",
    "pairwise_similarities",
    "cluster_embeddings",
]
