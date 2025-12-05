"""
Clustering and Analysis Example
===============================

===============================

This example demonstrates clustering embeddings to discover topics and patterns.

Main concepts:
- Clustering documents by semantic similarity
- Analyzing cluster characteristics
- Finding representative documents
- Topic discovery
- Visualizing embedding relationships
"""

from kerb.embedding import (
    embed,
    embed_batch,
    cluster_embeddings,
    cosine_similarity,
    mean_pooling,
    pairwise_similarities
)
from collections import defaultdict


def main():
    """Run clustering and analysis example."""
    
    print("="*80)
    print("CLUSTERING AND ANALYSIS EXAMPLE")
    print("="*80)
    
    # 1. Basic clustering
    print("\n1. BASIC DOCUMENT CLUSTERING")
    print("-"*80)
    
    documents = [
        # Technology cluster
        "Python programming language basics",
        "JavaScript web development",
        "Software engineering practices",
        # Health cluster
        "Healthy eating and nutrition",
        "Exercise and fitness routines",
        "Mental health and wellness",
        # Science cluster
        "Physics and quantum mechanics",
        "Chemistry and molecular structure",
        "Biology and genetics research"
    ]
    
    print(f"Clustering {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    
    # Generate embeddings
    embeddings = embed_batch(documents)
    
    # Cluster based on similarity threshold
    # Lower threshold = fewer, larger clusters
    # Higher threshold = more, smaller clusters
    clusters_list = cluster_embeddings(embeddings, threshold=0.5)
    
    print(f"\nFound {len(clusters_list)} groups:")
    
    # Display clusters
    for cluster_id, cluster_indices in enumerate(clusters_list):
        print(f"\nCluster {cluster_id}:")
        for idx in cluster_indices:
            print(f"  - {documents[idx]}")
    
    # 2. Topic discovery
    print("\n2. TOPIC DISCOVERY")
    print("-"*80)
    
    # Large collection of documents
    articles = [
        "Machine learning algorithms for prediction",
        "Deep neural networks architecture",
        "Natural language processing systems",
        "Computer vision applications",
        "Reinforcement learning agents",
        
        "Climate change and global warming",
        "Renewable energy sources",
        "Environmental conservation efforts",
        "Sustainable development goals",
        
        "Financial market analysis",
        "Investment strategies and portfolios",
        "Economic policy and impact",
        "Banking and financial services",
        
        "Medical research and discoveries",
        "Healthcare technology innovations",
        "Disease prevention methods",
        "Public health initiatives"
    ]
    
    print(f"Discovering topics in {len(articles)} articles...")
    
    article_embeddings = embed_batch(articles)
    
    # Cluster to discover topics - use threshold-based clustering
    # Adjust threshold to find natural groupings
    topic_clusters = cluster_embeddings(article_embeddings, threshold=0.4)
    
    # Build topics dict for later use
    topics = {}
    topic_embeddings_list = {}
    
    for topic_id, cluster_indices in enumerate(topic_clusters):
        topics[topic_id] = [articles[idx] for idx in cluster_indices]
        topic_embeddings_list[topic_id] = [article_embeddings[idx] for idx in cluster_indices]
    
    # Analyze each topic
    print(f"\nDiscovered {len(topic_clusters)} topics:")
    
    for topic_id, cluster_indices in enumerate(topic_clusters):
        print(f"\nTopic {topic_id} ({len(cluster_indices)} articles):")
        
        # Show sample articles
        for i, idx in enumerate(cluster_indices[:3]):
            print(f"  - {articles[idx]}")
        
        if len(cluster_indices) > 3:
            print(f"  ... and {len(cluster_indices) - 3} more")
    
    # 3. Cluster representatives
    print("\n3. FINDING CLUSTER REPRESENTATIVES")
    print("-"*80)
    
    print("Finding most representative document in each cluster:")
    
    for topic_id in sorted(topics.keys()):
        # Get centroid of cluster
        cluster_embs = topic_embeddings_list[topic_id]
        centroid = mean_pooling(cluster_embs)
        
        # Find document closest to centroid
        similarities = [
            cosine_similarity(centroid, emb)
            for emb in cluster_embs
        ]
        
        best_idx = similarities.index(max(similarities))
        representative = topics[topic_id][best_idx]
        
        print(f"\nTopic {topic_id} representative:")
        print(f"  '{representative}'")
        print(f"  Similarity to centroid: {max(similarities):.4f}")
    
    # 4. Cluster cohesion analysis
    print("\n4. CLUSTER COHESION ANALYSIS")
    print("-"*80)
    
    print("Analyzing how well-formed each cluster is:")
    
    for topic_id in sorted(topics.keys()):
        cluster_embs = topic_embeddings_list[topic_id]
        
        if len(cluster_embs) < 2:
            continue
        
        # Calculate pairwise similarities within cluster
        sim_matrix = pairwise_similarities(cluster_embs, metric="cosine")
        
        # Get average similarity (excluding diagonal)
        total_sim = 0
        count = 0
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                total_sim += sim_matrix[i][j]
                count += 1
        
        avg_similarity = total_sim / count if count > 0 else 0
        
        print(f"\nTopic {topic_id}:")
        print(f"  Documents: {len(topics[topic_id])}")
        print(f"  Average intra-cluster similarity: {avg_similarity:.4f}")
        print(f"  Cohesion: {'High' if avg_similarity > 0.5 else 'Medium' if avg_similarity > 0.3 else 'Low'}")
    
    # 5. Optimal cluster number
    print("\n5. TESTING DIFFERENT THRESHOLDS")
    print("-"*80)
    
    test_docs = [
        "Python programming", "Java development", "C++ coding",
        "Apple nutrition", "Banana benefits", "Orange vitamins",
        "Soccer sports", "Basketball games", "Tennis matches",
        "Rock music", "Jazz songs", "Classical symphony"
    ]
    
    test_embeddings = embed_batch(test_docs)
    
    print(f"Testing different thresholds on {len(test_docs)} documents:")
    
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        clusters_result = cluster_embeddings(test_embeddings, threshold=threshold)
        
        # Count cluster sizes
        cluster_sizes = [len(cluster) for cluster in clusters_result]
        
        print(f"\n  threshold={threshold}: {len(clusters_result)} clusters, sizes: {sorted(cluster_sizes)}")
    
    # 6. Hierarchical grouping
    print("\n6. MULTI-LEVEL CLUSTERING")
    print("-"*80)
    
    hierarchical_docs = [
        "Machine learning basics",
        "Deep learning advanced",
        "Neural networks",
        "Linear regression",
        "Decision trees"
    ]
    
    print("Documents:")
    for i, doc in enumerate(hierarchical_docs, 1):
        print(f"  {i}. {doc}")
    
    hier_embeddings = embed_batch(hierarchical_docs)
    
    # Level 1: Loose clustering (low threshold = fewer clusters)
    print("\nLevel 1 (loose clustering, threshold=0.3):")
    clusters_loose = cluster_embeddings(hier_embeddings, threshold=0.3)
    
    for cluster_id, cluster_indices in enumerate(clusters_loose):
        print(f"  Cluster {cluster_id}:")
        for idx in cluster_indices:
            print(f"    - {hierarchical_docs[idx]}")
    
    # Level 2: Tight clustering (high threshold = more clusters)
    print("\nLevel 2 (tight clustering, threshold=0.7):")
    clusters_tight = cluster_embeddings(hier_embeddings, threshold=0.7)
    
    for cluster_id, cluster_indices in enumerate(clusters_tight):
        print(f"  Cluster {cluster_id}:")
        for idx in cluster_indices:
            print(f"    - {hierarchical_docs[idx]}")
    
    # 7. Outlier detection
    print("\n7. OUTLIER DETECTION")
    print("-"*80)
    
    mixed_docs = [
        "Python programming language",
        "Java software development",
        "JavaScript web applications",
        "C++ system programming",
        "Banana fruit nutrition",  # Outlier
        "Ruby programming language"
    ]
    
    print("Documents (with one outlier):")
    for i, doc in enumerate(mixed_docs, 1):
        print(f"  {i}. {doc}")
    
    mixed_embeddings = embed_batch(mixed_docs)
    
    # Cluster to identify outliers - use high threshold
    # Outliers will form singleton clusters
    clusters_result = cluster_embeddings(mixed_embeddings, threshold=0.6)
    
    print("\nCluster sizes:")
    for cluster_id, cluster_indices in enumerate(clusters_result):
        print(f"  Cluster {cluster_id}: {len(cluster_indices)} documents")
    
    # Find potential outliers (singleton clusters or very small clusters)
    outlier_threshold = 2
    print(f"\nPotential outliers (clusters with < {outlier_threshold} documents):")
    
    for cluster_id, cluster_indices in enumerate(clusters_result):
        if len(cluster_indices) < outlier_threshold:
            for idx in cluster_indices:
                print(f"  - {mixed_docs[idx]}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKEY TAKEAWAYS:")
    print("- Clustering groups similar documents automatically")
    print("- cluster_embeddings() uses threshold-based similarity")
    print("- Lower threshold = fewer, larger clusters")
    print("- Higher threshold = more, smaller clusters")
    print("- Find representatives by computing cluster centroids")
    print("- Analyze cohesion with intra-cluster similarity")
    print("- Test different thresholds to find optimal grouping")
    print("- Small clusters may indicate outliers")
    print("- Multi-level clustering reveals structure at different granularities")


if __name__ == "__main__":
    main()
