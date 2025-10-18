"""Comprehensive tests for the retrieval module."""

import pytest

from kerb.retrieval import (Document, SearchResult, compress_context,
                            diversify_results, expand_query, filter_results,
                            format_results, generate_sub_queries,
                            hybrid_search, keyword_search,
                            reciprocal_rank_fusion, rerank_results,
                            results_to_context, rewrite_query, semantic_search)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            content="Python is a high-level programming language with dynamic typing.",
            metadata={"category": "programming", "language": "python", "views": 1000},
        ),
        Document(
            id="doc2",
            content="Asynchronous programming in Python uses async and await keywords.",
            metadata={"category": "programming", "language": "python", "views": 500},
        ),
        Document(
            id="doc3",
            content="JavaScript is widely used for web development and Node.js.",
            metadata={
                "category": "programming",
                "language": "javascript",
                "views": 800,
            },
        ),
        Document(
            id="doc4",
            content="Machine learning involves training models on data.",
            metadata={"category": "data_science", "views": 1200},
        ),
        Document(
            id="doc5",
            content="Python offers excellent libraries for data science and ML.",
            metadata={"category": "data_science", "language": "python", "views": 1500},
        ),
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing (simplified)."""
    # Simple mock embeddings (in practice, use real embeddings)
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2],
        [0.3, 0.3, 0.3, 0.3],
        [0.15, 0.25, 0.35, 0.45],
    ]


# ============================================================================
# Data Classes Tests
# ============================================================================


def test_document_creation():
    """Test Document data class."""
    doc = Document(
        id="test1", content="Test content", metadata={"key": "value"}, score=0.5
    )
    assert doc.id == "test1"
    assert doc.content == "Test content"
    assert doc.metadata["key"] == "value"
    assert doc.score == 0.5


def test_document_default_metadata():
    """Test Document with default metadata."""
    doc = Document(id="test1", content="Test")
    assert doc.metadata == {}
    assert doc.score == 0.0


def test_search_result_creation():
    """Test SearchResult data class."""
    doc = Document(id="test1", content="Test")
    result = SearchResult(document=doc, score=0.8, rank=1, method="keyword")
    assert result.document.id == "test1"
    assert result.score == 0.8
    assert result.rank == 1
    assert result.method == "keyword"


# ============================================================================
# Query Processing Tests
# ============================================================================


def test_rewrite_query_clear():
    """Test query rewriting with 'clear' style."""
    query = "what is the best python framework"
    rewritten = rewrite_query(query, style="clear")
    assert "the" not in rewritten.lower()
    assert "python" in rewritten.lower()


def test_rewrite_query_detailed():
    """Test query rewriting with 'detailed' style."""
    query = "python async"
    rewritten = rewrite_query(query, style="detailed")
    assert len(rewritten) > len(query)
    assert "python async" in rewritten.lower()


def test_rewrite_query_keywords():
    """Test query rewriting with 'keyword' style."""
    query = "how to use python for machine learning"
    rewritten = rewrite_query(query, style="keyword")
    assert "how" not in rewritten.lower()
    assert "python" in rewritten.lower()


def test_rewrite_query_detailed():
    """Test query rewriting with 'detailed' style."""
    query = "python async"
    rewritten = rewrite_query(query, style="detailed")
    assert len(rewritten) >= len(query)  # Detailed should expand


def test_rewrite_query_max_length():
    """Test query rewriting with max length."""
    query = "this is a very long query that should be truncated"
    rewritten = rewrite_query(query, style="clear", max_length=20)
    assert len(rewritten) <= 20


def test_expand_query_synonyms():
    """Test query expansion with synonyms."""
    query = "ML models"
    expanded = expand_query(query, method="synonyms")
    assert len(expanded) >= 1
    assert query in expanded


def test_expand_query_custom():
    """Test query expansion with custom expansions."""
    query = "python"
    expansions = ["python programming", "python language"]
    expanded = expand_query(query, expansions=expansions)
    assert "python programming" in expanded
    assert "python language" in expanded


def test_expand_query_related_terms():
    """Test query expansion with related terms."""
    query = "python and javascript"
    expanded = expand_query(query, method="related_terms")
    assert len(expanded) >= 1
    assert query in expanded


def test_generate_sub_queries():
    """Test sub-query generation."""
    query = "How to implement authentication in Python?"
    sub_queries = generate_sub_queries(query, max_queries=3)
    assert len(sub_queries) <= 3
    assert len(sub_queries) >= 1


# ============================================================================
# Search Tests
# ============================================================================


def test_keyword_search_basic(sample_documents):
    """Test basic keyword search."""
    results = keyword_search("python", sample_documents, top_k=5)
    assert len(results) <= 5
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].rank == 1


def test_keyword_search_scoring(sample_documents):
    """Test keyword search scoring."""
    results = keyword_search("python async", sample_documents, top_k=5)
    # Document with both terms should score higher
    assert len(results) > 0
    # Scores should be in descending order
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_keyword_search_empty_query(sample_documents):
    """Test keyword search with empty query."""
    results = keyword_search("", sample_documents, top_k=5)
    assert len(results) >= 0


def test_keyword_search_no_matches(sample_documents):
    """Test keyword search with no matches."""
    results = keyword_search("xyzabc123", sample_documents, top_k=5)
    # Should still return results but with low scores
    assert len(results) >= 0


def test_semantic_search_basic(sample_documents, sample_embeddings):
    """Test basic semantic search."""
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = semantic_search(
        query_embedding, sample_documents, sample_embeddings, top_k=3
    )
    assert len(results) <= 3
    assert all(isinstance(r, SearchResult) for r in results)


def test_semantic_search_cosine(sample_documents, sample_embeddings):
    """Test semantic search with cosine similarity."""
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = semantic_search(
        query_embedding, sample_documents, sample_embeddings, similarity_metric="cosine"
    )
    assert len(results) > 0
    assert all(-1 <= r.score <= 1 for r in results)


def test_semantic_search_dot_product(sample_documents, sample_embeddings):
    """Test semantic search with dot product."""
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = semantic_search(
        query_embedding, sample_documents, sample_embeddings, similarity_metric="dot"
    )
    assert len(results) > 0


def test_semantic_search_euclidean(sample_documents, sample_embeddings):
    """Test semantic search with euclidean distance."""
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = semantic_search(
        query_embedding,
        sample_documents,
        sample_embeddings,
        similarity_metric="euclidean",
    )
    assert len(results) > 0


def test_hybrid_search_basic(sample_documents, sample_embeddings):
    """Test basic hybrid search."""
    query = "python programming"
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = hybrid_search(
        query, query_embedding, sample_documents, sample_embeddings, top_k=3
    )
    assert len(results) <= 3
    assert all(isinstance(r, SearchResult) for r in results)


def test_hybrid_search_weighted(sample_documents, sample_embeddings):
    """Test hybrid search with weighted fusion."""
    query = "python"
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = hybrid_search(
        query,
        query_embedding,
        sample_documents,
        sample_embeddings,
        keyword_weight=0.7,
        semantic_weight=0.3,
        fusion_method="weighted",
    )
    assert len(results) > 0


def test_hybrid_search_rrf(sample_documents, sample_embeddings):
    """Test hybrid search with RRF fusion."""
    query = "python"
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = hybrid_search(
        query, query_embedding, sample_documents, sample_embeddings, fusion_method="rrf"
    )
    assert len(results) > 0


# ============================================================================
# Re-ranking Tests
# ============================================================================


def test_rerank_relevance(sample_documents):
    """Test re-ranking by relevance."""
    initial_results = keyword_search("python", sample_documents, top_k=5)
    reranked = rerank_results("python async", initial_results, method="relevance")
    assert len(reranked) == len(initial_results)


def test_rerank_popularity(sample_documents):
    """Test re-ranking by popularity."""
    initial_results = keyword_search("python", sample_documents, top_k=5)
    reranked = rerank_results("python", initial_results, method="popularity")
    assert len(reranked) == len(initial_results)


def test_rerank_diversity(sample_documents):
    """Test re-ranking for diversity."""
    initial_results = keyword_search("python", sample_documents, top_k=5)
    reranked = rerank_results("python", initial_results, method="diversity", top_k=3)
    assert len(reranked) <= 3


def test_rerank_custom_scorer(sample_documents):
    """Test re-ranking with custom scorer."""
    initial_results = keyword_search("python", sample_documents, top_k=5)

    def custom_scorer(query, doc):
        return doc.metadata.get("views", 0) * 0.001

    reranked = rerank_results(
        "python", initial_results, method="custom", scorer=custom_scorer
    )
    assert len(reranked) > 0


def test_reciprocal_rank_fusion(sample_documents):
    """Test reciprocal rank fusion."""
    results1 = keyword_search("python", sample_documents, top_k=3)
    results2 = keyword_search("programming", sample_documents, top_k=3)

    fused = reciprocal_rank_fusion([results1, results2], top_k=5)
    assert len(fused) <= 5
    assert all(isinstance(r, SearchResult) for r in fused)


def test_reciprocal_rank_fusion_single_list(sample_documents):
    """Test RRF with single result list."""
    results = keyword_search("python", sample_documents, top_k=3)
    fused = reciprocal_rank_fusion([results])
    assert len(fused) == len(results)


def test_diversify_results(sample_documents):
    """Test result diversification."""
    results = keyword_search("python", sample_documents, top_k=5)
    diverse = diversify_results(results, max_results=3, diversity_factor=0.5)
    assert len(diverse) <= 3


# ============================================================================
# Context Compression Tests
# ============================================================================


def test_compress_context_top_k(sample_documents):
    """Test context compression with top_k strategy."""
    results = keyword_search("python", sample_documents, top_k=5)
    compressed = compress_context("python", results, max_tokens=100, strategy="top_k")
    assert len(compressed) <= len(results)


def test_compress_context_summarize(sample_documents):
    """Test context compression with summarize strategy."""
    results = keyword_search("python", sample_documents, top_k=5)
    compressed = compress_context(
        "python", results, max_tokens=100, strategy="summarize"
    )
    assert len(compressed) <= len(results)


def test_compress_context_filter(sample_documents):
    """Test context compression with filter strategy."""
    results = keyword_search("python", sample_documents, top_k=5)
    compressed = compress_context("python", results, max_tokens=100, strategy="filter")
    assert len(compressed) <= len(results)


def test_compress_context_empty_results():
    """Test context compression with empty results."""
    compressed = compress_context("python", [], max_tokens=100)
    assert len(compressed) == 0


# ============================================================================
# Filtering Tests
# ============================================================================


def test_filter_results_min_score(sample_documents):
    """Test filtering by minimum score."""
    results = keyword_search("python", sample_documents, top_k=5)
    filtered = filter_results(results, min_score=0.5)
    assert all(r.score >= 0.5 for r in filtered)


def test_filter_results_max_results(sample_documents):
    """Test filtering by max results."""
    results = keyword_search("python", sample_documents, top_k=5)
    filtered = filter_results(results, max_results=2)
    assert len(filtered) <= 2


def test_filter_results_metadata(sample_documents):
    """Test filtering by metadata."""
    results = keyword_search("programming", sample_documents, top_k=5)
    filtered = filter_results(results, metadata_filter={"language": "python"})
    assert all(r.document.metadata.get("language") == "python" for r in filtered)


def test_filter_results_deduplication(sample_documents):
    """Test filtering with deduplication."""
    results = keyword_search("python", sample_documents, top_k=5)
    filtered = filter_results(results, dedup_threshold=0.5)
    # Should have fewer or equal results after dedup
    assert len(filtered) <= len(results)


def test_filter_results_combined(sample_documents):
    """Test filtering with multiple criteria."""
    results = keyword_search("python", sample_documents, top_k=5)
    filtered = filter_results(
        results,
        min_score=0.1,
        max_results=3,
        metadata_filter={"category": "programming"},
    )
    assert len(filtered) <= 3
    assert all(r.document.metadata.get("category") == "programming" for r in filtered)


# ============================================================================
# Formatting Tests
# ============================================================================


def test_format_results_simple(sample_documents):
    """Test simple result formatting."""
    results = keyword_search("python", sample_documents, top_k=3)
    formatted = format_results(results, format_style="simple")
    assert isinstance(formatted, str)
    assert len(formatted) > 0


def test_format_results_detailed(sample_documents):
    """Test detailed result formatting."""
    results = keyword_search("python", sample_documents, top_k=3)
    formatted = format_results(results, format_style="detailed", include_metadata=True)
    assert isinstance(formatted, str)
    assert "Metadata" in formatted


def test_format_results_json(sample_documents):
    """Test JSON result formatting."""
    results = keyword_search("python", sample_documents, top_k=3)
    formatted = format_results(results, format_style="json")
    assert isinstance(formatted, str)
    # Should be valid JSON
    import json

    parsed = json.loads(formatted)
    assert isinstance(parsed, list)


def test_format_results_empty():
    """Test formatting empty results."""
    formatted = format_results([], format_style="simple")
    assert "No results" in formatted


def test_results_to_context(sample_documents):
    """Test converting results to context string."""
    results = keyword_search("python", sample_documents, top_k=3)
    context = results_to_context(results, include_source=True)
    assert isinstance(context, str)
    assert len(context) > 0
    assert "[Source:" in context


def test_results_to_context_no_source(sample_documents):
    """Test context conversion without source."""
    results = keyword_search("python", sample_documents, top_k=3)
    context = results_to_context(results, include_source=False)
    assert isinstance(context, str)
    assert "[Source:" not in context


def test_results_to_context_custom_separator(sample_documents):
    """Test context conversion with custom separator."""
    results = keyword_search("python", sample_documents, top_k=3)
    separator = "\n===\n"
    context = results_to_context(results, separator=separator)
    assert separator in context


def test_results_to_context_empty():
    """Test context conversion with empty results."""
    context = results_to_context([])
    assert context == ""


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_empty_documents_keyword_search():
    """Test keyword search with empty documents."""
    results = keyword_search("python", [], top_k=5)
    assert len(results) == 0


def test_empty_documents_semantic_search():
    """Test semantic search with empty documents."""
    results = semantic_search([0.1, 0.2], [], [], top_k=5)
    assert len(results) == 0


def test_mismatched_embeddings(sample_documents):
    """Test semantic search with mismatched embeddings."""
    query_embedding = [0.1, 0.2, 0.3]
    # Only 2 embeddings for 5 documents
    doc_embeddings = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    results = semantic_search(
        query_embedding, sample_documents, doc_embeddings, top_k=5
    )
    assert len(results) == 0  # Should return empty due to mismatch


def test_large_top_k(sample_documents):
    """Test with top_k larger than available documents."""
    results = keyword_search("python", sample_documents, top_k=100)
    assert len(results) <= len(sample_documents)


def test_zero_top_k(sample_documents):
    """Test with top_k of 0."""
    results = keyword_search("python", sample_documents, top_k=0)
    assert len(results) == 0


def test_special_characters_query(sample_documents):
    """Test query with special characters."""
    results = keyword_search("python!!!", sample_documents, top_k=5)
    assert len(results) >= 0


def test_unicode_query(sample_documents):
    """Test query with unicode characters."""
    results = keyword_search("python 编程", sample_documents, top_k=5)
    assert len(results) >= 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_rag_pipeline(sample_documents, sample_embeddings):
    """Test complete RAG pipeline."""
    # Query processing
    query = "python async"
    expanded = expand_query(query, method="synonyms")
    assert len(expanded) >= 1

    # Hybrid search
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = hybrid_search(
        expanded[0], query_embedding, sample_documents, sample_embeddings, top_k=5
    )
    assert len(results) > 0

    # Re-ranking
    reranked = rerank_results(query, results, method="relevance")
    assert len(reranked) > 0

    # Filtering and compression
    filtered = filter_results(reranked, min_score=0.0)
    compressed = compress_context(query, filtered, max_tokens=500)
    assert len(compressed) > 0

    # Convert to context
    context = results_to_context(compressed)
    assert isinstance(context, str)
    assert len(context) > 0


def test_multi_query_fusion(sample_documents):
    """Test fusion of multiple query results."""
    queries = ["python", "programming", "async"]
    all_results = []

    for q in queries:
        results = keyword_search(q, sample_documents, top_k=3)
        all_results.append(results)

    fused = reciprocal_rank_fusion(all_results, top_k=5)
    assert len(fused) <= 5
    assert len(fused) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
