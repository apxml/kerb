"""
Similarity Metrics Example
==========================

==========================

This example demonstrates different similarity and distance metrics for comparing embeddings.

Main concepts:
- Cosine similarity for semantic similarity
- Euclidean distance for spatial distance
- Manhattan distance for grid-based distance
- Dot product for raw similarity
- Choosing the right metric for your use case
"""

from kerb.embedding import (
    embed,
    embed_batch,
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
    dot_product,
    batch_similarity,
    pairwise_similarities
)


def main():
    """Run similarity metrics example."""
    
    print("="*80)
    print("SIMILARITY METRICS EXAMPLE")
    print("="*80)
    
    # 1. Cosine similarity (most common for embeddings)
    print("\n1. COSINE SIMILARITY")
    print("-"*80)
    
    text1 = "Machine learning is a subset of artificial intelligence"
    text2 = "AI includes machine learning and deep learning"
    text3 = "Pizza is a popular Italian food"
    
    emb1 = embed(text1)
    emb2 = embed(text2)
    emb3 = embed(text3)
    
    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)
    sim_2_3 = cosine_similarity(emb2, emb3)
    
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Text 3: '{text3}'")
    print(f"\nSimilarity (1, 2): {sim_1_2:.4f} - Related topics")
    print(f"Similarity (1, 3): {sim_1_3:.4f} - Unrelated topics")
    print(f"Similarity (2, 3): {sim_2_3:.4f} - Unrelated topics")
    
    # 2. Euclidean distance
    print("\n2. EUCLIDEAN DISTANCE")
    print("-"*80)
    
    text_a = "Python programming language"
    text_b = "Python coding and development"
    text_c = "JavaScript web framework"
    
    emb_a = embed(text_a)
    emb_b = embed(text_b)
    emb_c = embed(text_c)
    
    dist_a_b = euclidean_distance(emb_a, emb_b)
    dist_a_c = euclidean_distance(emb_a, emb_c)
    
    print(f"Text A: '{text_a}'")
    print(f"Text B: '{text_b}'")
    print(f"Text C: '{text_c}'")
    print(f"\nDistance (A, B): {dist_a_b:.4f} - Very similar")
    print(f"Distance (A, C): {dist_a_c:.4f} - Different")
    print("\nNote: Lower distance = more similar")
    
    # 3. Manhattan distance
    print("\n3. MANHATTAN DISTANCE")
    print("-"*80)
    
    query = "data analysis"
    doc1 = "analyzing data with statistics"
    doc2 = "data visualization tools"
    
    q_emb = embed(query)
    d1_emb = embed(doc1)
    d2_emb = embed(doc2)
    
    man_dist_1 = manhattan_distance(q_emb, d1_emb)
    man_dist_2 = manhattan_distance(q_emb, d2_emb)
    
    print(f"Query: '{query}'")
    print(f"Doc 1: '{doc1}'")
    print(f"Doc 2: '{doc2}'")
    print(f"\nManhattan distance to Doc 1: {man_dist_1:.4f}")
    print(f"Manhattan distance to Doc 2: {man_dist_2:.4f}")
    
    # 4. Dot product
    print("\n4. DOT PRODUCT")
    print("-"*80)
    
    sentence1 = "Natural language understanding"
    sentence2 = "Language processing systems"
    sentence3 = "Computer vision algorithms"
    
    s1_emb = embed(sentence1)
    s2_emb = embed(sentence2)
    s3_emb = embed(sentence3)
    
    dot_1_2 = dot_product(s1_emb, s2_emb)
    dot_1_3 = dot_product(s1_emb, s3_emb)
    
    print(f"Sentence 1: '{sentence1}'")
    print(f"Sentence 2: '{sentence2}'")
    print(f"Sentence 3: '{sentence3}'")
    print(f"\nDot product (1, 2): {dot_1_2:.4f} - Related")
    print(f"Dot product (1, 3): {dot_1_3:.4f} - Less related")
    print("\nNote: Higher dot product = more aligned")
    
    # 5. Batch similarity comparison
    print("\n5. BATCH SIMILARITY WITH DIFFERENT METRICS")
    print("-"*80)
    
    query_text = "cloud computing infrastructure"
    documents = [
        "Cloud services and platforms",
        "Infrastructure as a service",
        "Traditional on-premise servers",
        "Mobile app development",
        "Database management systems"
    ]
    
    query_emb = embed(query_text)
    doc_embeddings = embed_batch(documents)
    
    print(f"Query: '{query_text}'")
    print("\nComparing metrics:")
    
    # Cosine similarity
    cosine_scores = batch_similarity(query_emb, doc_embeddings, metric="cosine")
    print("\nCosine Similarity (higher = more similar):")
    for i, (doc, score) in enumerate(zip(documents, cosine_scores), 1):
        print(f"  {i}. [{score:.4f}] {doc}")
    
    # Euclidean distance
    euclidean_scores = batch_similarity(query_emb, doc_embeddings, metric="euclidean")
    print("\nEuclidean Distance (lower = more similar):")
    for i, (doc, score) in enumerate(zip(documents, euclidean_scores), 1):
        print(f"  {i}. [{score:.4f}] {doc}")
    
    # 6. Pairwise similarities
    print("\n6. PAIRWISE SIMILARITIES")
    print("-"*80)
    
    concepts = [
        "Machine learning",
        "Deep learning",
        "Neural networks",
        "Data mining"
    ]
    
    concept_embeddings = embed_batch(concepts)
    
    print("Computing all pairwise similarities:")
    print("\nConcepts:", concepts)
    
    # Get pairwise similarity matrix
    similarity_matrix = pairwise_similarities(concept_embeddings, metric="cosine")
    
    print("\nSimilarity matrix:")
    print("       ", "  ".join(f"{i:5d}" for i in range(len(concepts))))
    for i, row in enumerate(similarity_matrix):
        print(f"{i}: ", "  ".join(f"{val:5.3f}" for val in row))
    
    # 7. Metric selection guide
    print("\n7. CHOOSING THE RIGHT METRIC")
    print("-"*80)
    
    test_query = "software engineering practices"
    test_docs = [
        "Engineering best practices",
        "Software development lifecycle",
        "Quality assurance testing"
    ]
    
    test_q_emb = embed(test_query)
    test_doc_embs = embed_batch(test_docs)
    
    print(f"Query: '{test_query}'")
    print("\nMetric comparison:")
    
    for metric_name in ["cosine", "euclidean", "manhattan", "dot"]:
        scores = batch_similarity(test_q_emb, test_doc_embs, metric=metric_name)
        
        if metric_name == "cosine" or metric_name == "dot":
            best_idx = scores.index(max(scores))
            best_score = max(scores)
            direction = "higher is better"
        else:
            best_idx = scores.index(min(scores))
            best_score = min(scores)
            direction = "lower is better"
        
        print(f"\n{metric_name.upper()} ({direction}):")
        print(f"  Best match: '{test_docs[best_idx]}'")
        print(f"  Score: {best_score:.4f}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKEY TAKEAWAYS:")
    print("- Cosine similarity: Best for normalized vectors (range: -1 to 1)")
    print("- Euclidean distance: Good for spatial similarity (lower = better)")
    print("- Manhattan distance: Robust to outliers (lower = better)")
    print("- Dot product: Fast but sensitive to magnitude")
    print("- For embeddings, cosine similarity is typically the best choice")
    print("- batch_similarity() efficiently compares one vector to many")
    print("- pairwise_similarities() creates similarity matrices")


if __name__ == "__main__":
    main()
