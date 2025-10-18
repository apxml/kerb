"""Utility functions for embeddings."""

from .analysis import (cluster_embeddings, embedding_dimension,
                       pairwise_similarities)
from .metrics import (batch_similarity, cosine_similarity, dot_product,
                      euclidean_distance, manhattan_distance, top_k_similar)
from .operations import (max_pooling, mean_pooling, normalize_vector,
                         vector_magnitude, weighted_mean_pooling)

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
