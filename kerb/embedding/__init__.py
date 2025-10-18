"""Embedding utilities for converting text to vector representations.

This module provides flexible embedding generation with multiple model options:
- Local/hash-based (no dependencies, for testing)
- Sentence Transformers (local ML models, high quality)
- OpenAI API (cloud-based, highest quality)

Usage Examples:
    # Common usage - core functions
    from kerb.embedding import embed, embed_batch

    vec = embed("Hello world")
    vecs = embed_batch(["Hello", "World"])

    # Provider-specific usage
    from kerb.embedding.providers import OpenAIEmbedder, LocalEmbedder
    from kerb.embedding.providers import SentenceTransformerEmbedder

    embedder = OpenAIEmbedder(model_name="text-embedding-3-large")
    vec = embedder.embed("Hello")

    # Utilities
    from kerb.embedding.utils import cosine_similarity, euclidean_distance

    similarity = cosine_similarity(vec1, vec2)
"""

# Submodule imports for specialized use
from . import providers, utils
# Core embedding functions (most commonly used)
from .embedder import (EmbeddingModel, ModelBackend,  # Enums; Core functions
                       embed, embed_async, embed_batch, embed_batch_async,
                       embed_batch_stream, embed_batch_stream_async)
# Import provider classes and functions for convenience
from .providers import (LocalEmbedder, OpenAIEmbedder,
                        SentenceTransformerEmbedder, local_embed, openai_embed,
                        openai_embed_async, openai_embed_batch,
                        openai_embed_batch_async, sentence_transformer_embed,
                        sentence_transformer_embed_batch)
# Import commonly used utilities to top level
from .utils import (batch_similarity, cluster_embeddings, cosine_similarity,
                    dot_product, embedding_dimension, euclidean_distance,
                    manhattan_distance, max_pooling, mean_pooling,
                    normalize_vector, pairwise_similarities, top_k_similar,
                    vector_magnitude, weighted_mean_pooling)

__all__ = [
    # Enums
    "EmbeddingModel",
    "ModelBackend",
    # Core functions (most common)
    "embed",
    "embed_batch",
    "embed_async",
    "embed_batch_async",
    "embed_batch_stream",
    "embed_batch_stream_async",
    # Submodules
    "providers",
    "utils",
    # Provider classes
    "LocalEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    # Provider functions
    "local_embed",
    "openai_embed",
    "openai_embed_batch",
    "openai_embed_async",
    "openai_embed_batch_async",
    "sentence_transformer_embed",
    "sentence_transformer_embed_batch",
    # Similarity metrics
    "cosine_similarity",
    "euclidean_distance",
    "manhattan_distance",
    "dot_product",
    "batch_similarity",
    "top_k_similar",
    # Vector utilities
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
