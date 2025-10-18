"""Local embedding providers (run on your machine, no API calls).

This module includes:
1. Hash-based embeddings (no dependencies, for testing/prototyping)
2. Sentence Transformers (local ML models, high quality)
"""

import hashlib
from typing import Any, Dict, List

# Model cache for loaded ML models
_model_cache: Dict[str, Any] = {}


def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize a vector to unit length (L2 norm = 1).

    Args:
        vector (List[float]): Input vector

    Returns:
        List[float]: Normalized vector
    """
    import math

    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


# ============================================================================
# Hash-based Local Embeddings
# ============================================================================


def local_embed(text: str, dimensions: int = 384) -> List[float]:
    """Generate embedding using local hash-based method.

    This is a simple, deterministic embedding that requires no external models.
    Suitable for testing, prototyping, or when you don't need semantic quality.

    Args:
        text (str): Text to embed
        dimensions (int): Embedding dimension

    Returns:
        List[float]: Normalized embedding vector
    """
    if not text:
        return [0.0] * dimensions

    # Hash-based embedding
    text_hash = hashlib.md5(text.encode()).hexdigest()

    vector = []
    for i in range(dimensions):
        char_index = i % len(text_hash)
        char_value = ord(text_hash[char_index])
        normalized_value = (char_value - 127.5) / 127.5
        vector.append(normalized_value)

    return normalize_vector(vector)


class LocalEmbedder:
    """Local hash-based embedder

    This is a simple, deterministic embedding that requires no external models.
    Suitable for testing, prototyping, or when you don't need semantic quality.

    Args:
        dimensions (int): Embedding dimension (default: 384)

    Examples:
        embedder = LocalEmbedder(dimensions=512)
        vec = embedder.embed("Hello world")
        vecs = embedder.embed_batch(["Hello", "World"])
    """

    def __init__(self, dimensions: int = 384):
        """Initialize the local embedder.

        Args:
            dimensions (int): Embedding dimension
        """
        self.dimensions = dimensions

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text (str): Text to embed

        Returns:
            List[float]: Embedding vector
        """
        return local_embed(text, dimensions=self.dimensions)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts (List[str]): Texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        return [self.embed(text) for text in texts]


# ============================================================================
# Sentence Transformers (Local ML Models)
# ============================================================================


def sentence_transformer_embed(
    text: str, model_name: str = "all-MiniLM-L6-v2", **kwargs
) -> List[float]:
    """Generate embedding using Sentence Transformers (local ML model).

    Requires: pip install sentence-transformers

    Args:
        text (str): Text to embed
        model_name (str): Model name (default: "all-MiniLM-L6-v2")
        **kwargs: Additional model parameters

    Returns:
        List[float]: Embedding vector

    Popular models:
        - "all-MiniLM-L6-v2" (384 dim, fast)
        - "all-mpnet-base-v2" (768 dim, quality)
        - "all-MiniLM-L12-v2" (384 dim, balanced)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )

    # Get or cache model
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)

    model = _model_cache[model_name]
    embedding = model.encode(text, **kwargs)

    return embedding.tolist()


def sentence_transformer_embed_batch(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    **kwargs,
) -> List[List[float]]:
    """Generate embeddings for multiple texts using Sentence Transformers.

    More efficient than calling sentence_transformer_embed repeatedly.

    Args:
        texts (List[str]): Texts to embed
        model_name (str): Model name (default: "all-MiniLM-L6-v2")
        batch_size (int): Batch size for processing
        **kwargs: Additional model parameters

    Returns:
        List[List[float]]: List of embedding vectors
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )

    # Get or cache model
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)

    model = _model_cache[model_name]
    embeddings = model.encode(texts, batch_size=batch_size, **kwargs)

    return [emb.tolist() for emb in embeddings]


class SentenceTransformerEmbedder:
    """Sentence Transformers embedding provider (runs locally).

    Requires: pip install sentence-transformers

    Args:
        model_name (str): Model name (default: "all-MiniLM-L6-v2")

    Examples:
        embedder = SentenceTransformerEmbedder(model_name="all-mpnet-base-v2")
        vec = embedder.embed("Hello world")
        vecs = embedder.embed_batch(["Hello", "World"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the Sentence Transformer embedder.

        Args:
            model_name (str): Model name
        """
        self.model_name = model_name

    def embed(self, text: str, **kwargs) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text (str): Text to embed
            **kwargs: Additional model parameters

        Returns:
            List[float]: Embedding vector
        """
        return sentence_transformer_embed(text, self.model_name, **kwargs)

    def embed_batch(
        self, texts: List[str], batch_size: int = 32, **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts (List[str]): Texts to embed
            batch_size (int): Batch size for processing
            **kwargs: Additional model parameters

        Returns:
            List[List[float]]: List of embedding vectors
        """
        return sentence_transformer_embed_batch(
            texts, self.model_name, batch_size, **kwargs
        )
