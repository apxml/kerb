from kerb.embedding import cosine_similarity, embed, embed_batch


def test_embedding_dimension_and_norm():
    vec = embed("hello", dimensions=16)
    assert len(vec) == 16
    # Vector should be normalized (unit length within tolerance)
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_embedding_empty_text():
    vec = embed("", dimensions=8)
    assert vec == [0.0] * 8


def test_batch_embedding():
    texts = ["a", "b", "a"]
    vectors = embed_batch(texts, dimensions=8)
    assert len(vectors) == 3
    assert vectors[0] == vectors[2]  # same input => same embedding
    assert vectors[0] != vectors[1]


def test_similarity():
    v1 = embed("hello", dimensions=16)
    v2 = embed("hello", dimensions=16)
    v3 = embed("different", dimensions=16)
    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-6
    sim_diff = cosine_similarity(v1, v3)
    assert sim_diff < 1.0
    assert 0.0 <= sim_diff <= 1.0
