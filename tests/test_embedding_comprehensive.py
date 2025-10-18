"""Comprehensive tests for the embedding module."""

import pytest

from kerb.embedding import (  # Core functions; Similarity metrics; Vector utilities; Analysis; Backend-specific
    batch_similarity, cluster_embeddings, cosine_similarity, dot_product,
    embed, embed_batch, embedding_dimension, euclidean_distance, local_embed,
    manhattan_distance, max_pooling, mean_pooling, normalize_vector,
    pairwise_similarities, top_k_similar, vector_magnitude,
    weighted_mean_pooling)

# ============================================================================
# Core Embedding Function Tests
# ============================================================================


def test_embed_basic():
    """Test basic embedding generation."""
    vec = embed("Hello, world!")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(v, float) for v in vec)


def test_embed_with_model():
    """Test embedding with explicit model."""
    vec = embed("Hello", model="local", dimensions=256)
    assert len(vec) == 256


def test_embed_empty_text():
    """Test embedding of empty text."""
    vec = embed("")
    assert isinstance(vec, list)
    assert all(v == 0.0 for v in vec)


def test_embed_same_text_same_embedding():
    """Test that same text produces same embedding."""
    text = "consistent text"
    vec1 = embed(text)
    vec2 = embed(text)
    assert vec1 == vec2


def test_embed_batch_basic():
    """Test batch embedding generation."""
    texts = ["first", "second", "third"]
    vectors = embed_batch(texts)
    assert len(vectors) == 3
    assert all(isinstance(vec, list) for vec in vectors)


def test_embed_batch_empty_list():
    """Test batch embedding with empty list."""
    vectors = embed_batch([])
    assert vectors == []


def test_embed_batch_consistency():
    """Test that batch embedding produces same results as individual."""
    texts = ["text1", "text2"]
    batch_vecs = embed_batch(texts)
    individual_vecs = [embed(t) for t in texts]
    assert batch_vecs == individual_vecs


# ============================================================================
# Similarity Metric Tests
# ============================================================================


def test_cosine_similarity_identical():
    """Test cosine similarity of identical vectors."""
    vec = embed("test text")
    sim = cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_similarity_different():
    """Test cosine similarity of different vectors."""
    vec1 = embed("hello")
    vec2 = embed("world")
    sim = cosine_similarity(vec1, vec2)
    assert -1.0 <= sim <= 1.0
    assert sim < 1.0


def test_cosine_similarity_dimension_mismatch():
    """Test that dimension mismatch raises error."""
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0]
    with pytest.raises(ValueError, match="same dimensions"):
        cosine_similarity(vec1, vec2)


def test_euclidean_distance_identical():
    """Test Euclidean distance of identical vectors."""
    vec = embed("test")
    dist = euclidean_distance(vec, vec)
    assert abs(dist) < 1e-6


def test_euclidean_distance_different():
    """Test Euclidean distance of different vectors."""
    vec1 = embed("hello")
    vec2 = embed("world")
    dist = euclidean_distance(vec1, vec2)
    assert dist > 0


def test_manhattan_distance():
    """Test Manhattan distance calculation."""
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [4.0, 5.0, 6.0]
    dist = manhattan_distance(vec1, vec2)
    assert dist == 9.0


def test_dot_product():
    """Test dot product calculation."""
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [4.0, 5.0, 6.0]
    prod = dot_product(vec1, vec2)
    assert prod == 32.0  # 1*4 + 2*5 + 3*6


def test_batch_similarity_cosine():
    """Test batch similarity with cosine metric."""
    query = embed("query")
    docs = embed_batch(["doc1", "doc2", "doc3"])
    scores = batch_similarity(query, docs, metric="cosine")
    assert len(scores) == 3
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_batch_similarity_euclidean():
    """Test batch similarity with euclidean metric."""
    query = embed("query")
    docs = embed_batch(["doc1", "doc2"])
    scores = batch_similarity(query, docs, metric="euclidean")
    assert len(scores) == 2
    assert all(s >= 0 for s in scores)


def test_batch_similarity_invalid_metric():
    """Test that invalid metric raises error."""
    query = embed("query")
    docs = embed_batch(["doc1"])
    with pytest.raises(ValueError, match="Unknown metric"):
        batch_similarity(query, docs, metric="invalid")


def test_top_k_similar_basic():
    """Test top-k similar vector retrieval."""
    query = embed("query")
    docs = embed_batch(["doc1", "doc2", "doc3", "doc4"])
    top_k = top_k_similar(query, docs, k=2)
    assert len(top_k) == 2
    assert all(isinstance(idx, int) for idx in top_k)


def test_top_k_similar_with_scores():
    """Test top-k similar with score return."""
    query = embed("query")
    docs = embed_batch(["doc1", "doc2", "doc3"])
    results = top_k_similar(query, docs, k=2, return_scores=True)
    assert len(results) == 2
    assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
    indices, scores = zip(*results)
    assert all(isinstance(idx, int) for idx in indices)
    assert all(isinstance(score, float) for score in scores)


def test_top_k_similar_k_larger_than_list():
    """Test top-k when k > number of vectors."""
    query = embed("query")
    docs = embed_batch(["doc1", "doc2"])
    top_k = top_k_similar(query, docs, k=10)
    assert len(top_k) == 2  # Should return all available


# ============================================================================
# Vector Utility Tests
# ============================================================================


def test_normalize_vector():
    """Test vector normalization."""
    vec = [3.0, 4.0]  # Length 5
    normalized = normalize_vector(vec)
    magnitude = sum(x * x for x in normalized) ** 0.5
    assert abs(magnitude - 1.0) < 1e-6


def test_normalize_zero_vector():
    """Test normalizing zero vector."""
    vec = [0.0, 0.0, 0.0]
    normalized = normalize_vector(vec)
    assert normalized == vec


def test_vector_magnitude():
    """Test magnitude calculation."""
    vec = [3.0, 4.0]
    mag = vector_magnitude(vec)
    assert abs(mag - 5.0) < 1e-6


def test_mean_pooling():
    """Test mean pooling of vectors."""
    vectors = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    mean = mean_pooling(vectors)
    assert mean == [4.0, 5.0, 6.0]


def test_mean_pooling_empty():
    """Test mean pooling with empty list."""
    assert mean_pooling([]) == []


def test_mean_pooling_dimension_mismatch():
    """Test mean pooling with mismatched dimensions."""
    vectors = [
        [1.0, 2.0],
        [1.0, 2.0, 3.0],  # Different dimension
    ]
    with pytest.raises(ValueError, match="same dimensions"):
        mean_pooling(vectors)


def test_weighted_mean_pooling():
    """Test weighted mean pooling."""
    vectors = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    weights = [0.25, 0.75]
    result = weighted_mean_pooling(vectors, weights)
    # (1*0.25 + 3*0.75, 2*0.25 + 4*0.75) = (2.5, 3.5)
    assert abs(result[0] - 2.5) < 1e-6
    assert abs(result[1] - 3.5) < 1e-6


def test_weighted_mean_pooling_auto_normalize():
    """Test that weights are automatically normalized."""
    vectors = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    weights = [1.0, 3.0]  # Will be normalized to [0.25, 0.75]
    result = weighted_mean_pooling(vectors, weights)
    assert abs(result[0] - 2.5) < 1e-6
    assert abs(result[1] - 3.5) < 1e-6


def test_weighted_mean_pooling_mismatched_length():
    """Test weighted pooling with mismatched lengths."""
    vectors = [[1.0, 2.0], [3.0, 4.0]]
    weights = [1.0]  # Only one weight for two vectors
    with pytest.raises(ValueError, match="must match"):
        weighted_mean_pooling(vectors, weights)


def test_max_pooling():
    """Test max pooling across vectors."""
    vectors = [
        [1.0, 5.0, 3.0],
        [4.0, 2.0, 6.0],
        [3.0, 4.0, 1.0],
    ]
    result = max_pooling(vectors)
    assert result == [4.0, 5.0, 6.0]


def test_max_pooling_single_vector():
    """Test max pooling with single vector."""
    vectors = [[1.0, 2.0, 3.0]]
    result = max_pooling(vectors)
    assert result == [1.0, 2.0, 3.0]


# ============================================================================
# Analysis Function Tests
# ============================================================================


def test_embedding_dimension():
    """Test getting embedding dimension."""
    vec = embed("test", backend="local")
    dim = embedding_dimension(vec)
    assert isinstance(dim, int)
    assert dim > 0


def test_pairwise_similarities():
    """Test pairwise similarity matrix."""
    vectors = embed_batch(["a", "b", "c"])
    matrix = pairwise_similarities(vectors)

    # Should be NxN matrix
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)

    # Diagonal should be 1.0 (self-similarity)
    for i in range(3):
        assert abs(matrix[i][i] - 1.0) < 1e-6

    # Should be symmetric
    assert abs(matrix[0][1] - matrix[1][0]) < 1e-6


def test_cluster_embeddings_basic():
    """Test basic clustering."""
    # Create vectors that should cluster together
    vectors = [
        [1.0, 0.0, 0.0],
        [0.99, 0.1, 0.0],  # Similar to first
        [0.0, 1.0, 0.0],  # Different
    ]
    vectors = [normalize_vector(v) for v in vectors]

    clusters = cluster_embeddings(vectors, threshold=0.9)

    # Should have at least 2 clusters
    assert len(clusters) >= 2
    assert all(isinstance(cluster, list) for cluster in clusters)


def test_cluster_embeddings_high_threshold():
    """Test clustering with high threshold (each item separate)."""
    vectors = embed_batch(["a", "b", "c"])
    clusters = cluster_embeddings(vectors, threshold=0.99)

    # With high threshold, likely each item is its own cluster
    assert len(clusters) >= 1


# ============================================================================
# Local Backend Tests
# ============================================================================


def test_local_embed_dimensions():
    """Test local embedding with custom dimensions."""
    vec = local_embed("test", dimensions=128)
    assert len(vec) == 128


def test_local_embed_normalized():
    """Test that local embeddings are normalized."""
    vec = local_embed("test")
    magnitude = sum(x * x for x in vec) ** 0.5
    assert abs(magnitude - 1.0) < 1e-6


def test_local_embed_deterministic():
    """Test that local embeddings are deterministic."""
    text = "deterministic test"
    vec1 = local_embed(text)
    vec2 = local_embed(text)
    assert vec1 == vec2


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_workflow():
    """Test a complete embedding workflow."""
    # Generate embeddings
    documents = ["Document 1", "Document 2", "Document 3"]
    doc_embeddings = embed_batch(documents, dimensions=256)

    # Create query
    query = "Document 1"
    query_embedding = embed(query, dimensions=256)

    # Find similar documents
    top_docs = top_k_similar(query_embedding, doc_embeddings, k=2, return_scores=True)

    assert len(top_docs) == 2
    # First result should be most similar
    assert top_docs[0][0] == 0  # Index of "Document 1"


def test_semantic_search_simulation():
    """Test simulating semantic search."""
    # Index documents
    documents = [
        "Python is a programming language",
        "Java is also a programming language",
        "The sky is blue",
    ]
    doc_embeddings = embed_batch(documents, dimensions=256)

    # Search query
    query = "programming languages"
    query_embedding = embed(query, dimensions=256)

    # Find top-2 similar
    results = top_k_similar(query_embedding, doc_embeddings, k=2, return_scores=True)

    # Should return 2 results
    assert len(results) == 2

    # Note: With hash-based embeddings, semantic quality isn't guaranteed,
    # but the mechanics should work


def test_document_averaging():
    """Test averaging embeddings of related documents."""
    docs = [
        "First part of the story",
        "Second part of the story",
        "Third part of the story",
    ]
    embeddings = embed_batch(docs, dimensions=256)

    # Average the embeddings
    avg_embedding = mean_pooling(embeddings)

    # Should have same dimension
    assert len(avg_embedding) == len(embeddings[0])

    # Should be a valid embedding
    mag = vector_magnitude(avg_embedding)
    assert mag > 0
