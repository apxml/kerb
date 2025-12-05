"""
Vector Operations Example
=========================

=========================

This example demonstrates vector operations for combining and analyzing embeddings.

Main concepts:
- Normalizing vectors to unit length
- Pooling operations (mean, weighted, max)
- Vector arithmetic for semantic operations
- Clustering embeddings by similarity
- Analyzing embedding dimensions
"""

from kerb.embedding import (
    embed,
    embed_batch,
    normalize_vector,
    vector_magnitude,
    mean_pooling,
    weighted_mean_pooling,
    max_pooling,
    embedding_dimension,
    cluster_embeddings,
    cosine_similarity
)


def main():
    """Run vector operations example."""
    
    print("="*80)
    print("VECTOR OPERATIONS EXAMPLE")
    print("="*80)
    
    # 1. Vector normalization
    print("\n1. VECTOR NORMALIZATION")
    print("-"*80)
    
    text = "Vector normalization ensures unit length"
    vec = embed(text)
    
    print(f"Text: '{text}'")
    print(f"Original magnitude: {vector_magnitude(vec):.6f}")
    
    normalized = normalize_vector(vec)
    print(f"Normalized magnitude: {vector_magnitude(normalized):.6f}")
    print(f"First 3 values (original): {[round(v, 4) for v in vec[:3]]}")
    print(f"First 3 values (normalized): {[round(v, 4) for v in normalized[:3]]}")
    
    # 2. Mean pooling for document averaging
    print("\n2. MEAN POOLING - DOCUMENT AVERAGING")
    print("-"*80)
    
    # Multiple sentences from the same document
    sentences = [
        "Climate change affects global temperatures.",
        "Rising sea levels threaten coastal cities.",
        "Carbon emissions contribute to global warming."
    ]
    
    print("Document sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    
    # Embed each sentence
    sentence_embeddings = embed_batch(sentences)
    
    # Average to get document-level embedding
    document_embedding = mean_pooling(sentence_embeddings)
    
    print(f"\nDocument embedding dimension: {len(document_embedding)}")
    print(f"Document embedding magnitude: {vector_magnitude(document_embedding):.4f}")
    
    # Compare with a query
    query = "environmental impact of climate change"
    query_emb = embed(query)
    
    similarity = cosine_similarity(query_emb, document_embedding)
    print(f"\nSimilarity with query '{query}': {similarity:.4f}")
    
    # 3. Weighted mean pooling
    print("\n3. WEIGHTED MEAN POOLING")
    print("-"*80)
    
    # Different parts of a document with different importance
    document_parts = [
        "Title: Introduction to Machine Learning",
        "Abstract: This paper discusses ML fundamentals.",
        "Conclusion: ML transforms data into insights."
    ]
    
    # Assign importance weights
    weights = [3.0, 2.0, 1.0]  # Title most important, conclusion least
    
    print("Document parts with weights:")
    for part, weight in zip(document_parts, weights):
        print(f"  [{weight:.1f}] {part}")
    
    part_embeddings = embed_batch(document_parts)
    weighted_embedding = weighted_mean_pooling(part_embeddings, weights)
    
    print(f"\nWeighted document embedding created")
    print(f"Dimension: {len(weighted_embedding)}")
    
    # 4. Max pooling
    print("\n4. MAX POOLING")
    print("-"*80)
    
    phrases = [
        "Important key concept",
        "Another critical point",
        "Final main idea"
    ]
    
    print("Phrases:")
    for phrase in phrases:
        print(f"  - {phrase}")
    
    phrase_embeddings = embed_batch(phrases)
    
    # Max pooling takes maximum value per dimension
    max_pooled = max_pooling(phrase_embeddings)
    
    print(f"\nMax pooled embedding dimension: {len(max_pooled)}")
    print(f"This captures the strongest features across all phrases")
    
    # Compare pooling methods
    mean_pooled = mean_pooling(phrase_embeddings)
    
    print(f"\nMagnitude comparison:")
    print(f"  Mean pooling: {vector_magnitude(mean_pooled):.4f}")
    print(f"  Max pooling: {vector_magnitude(max_pooled):.4f}")
    
    # 5. Semantic vector arithmetic
    print("\n5. SEMANTIC VECTOR ARITHMETIC")
    print("-"*80)
    
    # Classic example: king - man + woman = queen
    # Here we demonstrate with technical terms
    
    word1 = "programming"
    word2 = "Python"
    word3 = "Java"
    
    emb1 = embed(word1)
    emb2 = embed(word2)
    emb3 = embed(word3)
    
    print(f"Concept: '{word1}' is to '{word2}' as X is to '{word3}'")
    
    # Vector arithmetic: programming - Python + Java
    result_vector = [
        emb1[i] - emb2[i] + emb3[i] 
        for i in range(len(emb1))
    ]
    
    # Normalize result
    result_vector = normalize_vector(result_vector)
    
    # Compare with candidate words
    candidates = [
        "coding",
        "software",
        "development",
        "algorithm"
    ]
    
    print(f"\nTop candidates for X:")
    candidate_embs = embed_batch(candidates)
    
    similarities = [
        (word, cosine_similarity(result_vector, emb))
        for word, emb in zip(candidates, candidate_embs)
    ]
    
    for word, sim in sorted(similarities, key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {word}: {sim:.4f}")
    
    # 6. Clustering embeddings
    print("\n6. CLUSTERING EMBEDDINGS")
    print("-"*80)
    
    # Documents from different topics
    mixed_docs = [
        "Python programming language",
        "Java software development",
        "Apple fruit nutrition",
        "Banana health benefits",
        "JavaScript web framework",
        "Orange vitamin C content"
    ]
    
    print(f"Clustering {len(mixed_docs)} documents:")
    for doc in mixed_docs:
        print(f"  - {doc}")
    
    mixed_embeddings = embed_batch(mixed_docs)
    
    # Cluster into groups based on similarity threshold
    clusters = cluster_embeddings(mixed_embeddings, threshold=0.6)
    
    print(f"\nFound {len(clusters)} groups:")
    
    # Group documents by cluster
    for cluster_id, cluster_indices in enumerate(clusters):
        print(f"\nCluster {cluster_id}:")
        for idx in cluster_indices:
            print(f"  - {mixed_docs[idx]}")
    
    # 7. Dimension analysis
    print("\n7. DIMENSION ANALYSIS")
    print("-"*80)
    
    sample_text = "Analyzing embedding dimensions"
    sample_emb = embed(sample_text)
    
    dim = embedding_dimension(sample_emb)
    print(f"Embedding dimension: {dim}")
    
    # Analyze value distribution
    positive_count = sum(1 for v in sample_emb if v > 0)
    negative_count = sum(1 for v in sample_emb if v < 0)
    zero_count = sum(1 for v in sample_emb if v == 0)
    
    print(f"\nValue distribution:")
    print(f"  Positive values: {positive_count} ({positive_count/dim*100:.1f}%)")
    print(f"  Negative values: {negative_count} ({negative_count/dim*100:.1f}%)")
    print(f"  Zero values: {zero_count}")
    
    # Value range
    print(f"\nValue range:")
    print(f"  Min: {min(sample_emb):.6f}")
    print(f"  Max: {max(sample_emb):.6f}")
    print(f"  Mean: {sum(sample_emb)/len(sample_emb):.6f}")
    
    # 8. Building composite embeddings
    print("\n8. BUILDING COMPOSITE EMBEDDINGS")
    print("-"*80)
    
    # Combine multiple information sources
    title_text = "Advanced Machine Learning Techniques"
    content_text = "This document covers deep learning, neural networks, and optimization"
    tags = ["machine learning", "deep learning", "AI"]
    
    print("Creating composite embedding from:")
    print(f"  Title: {title_text}")
    print(f"  Content: {content_text}")
    print(f"  Tags: {', '.join(tags)}")
    
    # Embed each component
    title_emb = embed(title_text)
    content_emb = embed(content_text)
    tag_embeddings = embed_batch(tags)
    tag_emb = mean_pooling(tag_embeddings)
    
    # Combine with different weights
    composite_emb = weighted_mean_pooling(
        [title_emb, content_emb, tag_emb],
        weights=[3.0, 2.0, 1.0]  # Title most important
    )
    
    print(f"\nComposite embedding created:")
    print(f"  Dimension: {len(composite_emb)}")
    print(f"  Magnitude: {vector_magnitude(composite_emb):.4f}")
    
    # Test against query
    test_query = "AI and neural networks"
    test_query_emb = embed(test_query)
    
    similarity = cosine_similarity(test_query_emb, composite_emb)
    print(f"\nSimilarity with query '{test_query}': {similarity:.4f}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKEY TAKEAWAYS:")
    print("- Normalize vectors for consistent magnitude")
    print("- Mean pooling averages multiple embeddings")
    print("- Weighted pooling emphasizes important parts")
    print("- Max pooling captures strongest features")
    print("- Vector arithmetic enables semantic operations")
    print("- Clustering groups similar embeddings")
    print("- Composite embeddings combine multiple sources")


if __name__ == "__main__":
    main()
