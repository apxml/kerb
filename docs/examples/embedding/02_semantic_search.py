"""
Semantic Search Example
=======================

This example demonstrates how to build a semantic search system using embeddings.

Main concepts:
- Creating a searchable document collection
- Computing query-document similarities
- Ranking results by relevance
- Finding top-k most similar documents
"""

from kerb.embedding import (
    embed,
    embed_batch,
    cosine_similarity,
    top_k_similar,
    batch_similarity
)


def main():
    """Run semantic search example."""
    
    print("="*80)
    print("SEMANTIC SEARCH EXAMPLE")
    print("="*80)
    
    # 1. Create a document collection
    print("\n1. BUILDING DOCUMENT COLLECTION")
    print("-"*80)
    
    documents = [
        "Python is a high-level programming language",
        "Machine learning models learn patterns from data",
        "Natural language processing helps computers understand text",
        "Deep neural networks have multiple layers",
        "Data science combines statistics and programming",
        "Artificial intelligence enables machines to think",
        "Software engineering involves designing and building systems",
        "Cloud computing provides scalable infrastructure",
        "Database systems store and manage data efficiently",
        "Web development creates interactive user interfaces"
    ]
    
    print(f"Indexing {len(documents)} documents...")
    
    # Generate embeddings for all documents
    doc_embeddings = embed_batch(documents)
    print(f"Generated {len(doc_embeddings)} embeddings")
    print(f"Each embedding has {len(doc_embeddings[0])} dimensions")
    
    # 2. Search with a query
    print("\n2. SEMANTIC SEARCH")
    print("-"*80)
    
    query = "I want to learn about AI and neural networks"
    print(f"Query: '{query}'")
    
    # Generate query embedding
    query_embedding = embed(query)
    
    # Calculate similarity with all documents
    similarities = batch_similarity(query_embedding, doc_embeddings, metric="cosine")
    
    # Display all results sorted by relevance
    print("\nAll results (sorted by relevance):")
    results = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
    
    for i, (doc, score) in enumerate(results[:5], 1):
        print(f"{i}. [{score:.4f}] {doc}")
    
    # 3. Top-K retrieval
    print("\n3. TOP-K RETRIEVAL")
    print("-"*80)
    
    query2 = "programming languages and software development"
    print(f"Query: '{query2}'")
    
    query_emb2 = embed(query2)
    
    # Get top 3 most similar documents
    top_3_indices = top_k_similar(query_emb2, doc_embeddings, k=3)
    
    print(f"\nTop 3 results:")
    for rank, idx in enumerate(top_3_indices, 1):
        similarity = cosine_similarity(query_emb2, doc_embeddings[idx])
        print(f"{rank}. [{similarity:.4f}] {documents[idx]}")
    
    # 4. Multiple query terms
    print("\n4. MULTI-TERM QUERY")
    print("-"*80)
    
    queries = [
        "machine learning and data",
        "web applications",
        "database management"
    ]
    
    for query in queries:
        query_emb = embed(query)
        top_idx = top_k_similar(query_emb, doc_embeddings, k=1)[0]
        similarity = cosine_similarity(query_emb, doc_embeddings[top_idx])
        
        print(f"\nQuery: '{query}'")
        print(f"  Best match [{similarity:.4f}]: {documents[top_idx]}")
    
    # 5. Similarity threshold filtering
    print("\n5. THRESHOLD FILTERING")
    print("-"*80)
    
    query3 = "quantum computing"
    print(f"Query: '{query3}'")
    
    query_emb3 = embed(query3)
    similarities3 = batch_similarity(query_emb3, doc_embeddings, metric="cosine")
    
    # Only show results above threshold
    threshold = 0.3
    print(f"\nResults with similarity > {threshold}:")
    
    relevant_results = [
        (doc, sim) for doc, sim in zip(documents, similarities3) if sim > threshold
    ]
    
    if relevant_results:
        for doc, sim in sorted(relevant_results, key=lambda x: x[1], reverse=True):
            print(f"  [{sim:.4f}] {doc}")
    else:
        print(f"  No results found above threshold {threshold}")
        print(f"  Best match: [{max(similarities3):.4f}] {documents[similarities3.index(max(similarities3))]}")
    
    # 6. Building a simple search engine
    print("\n6. SIMPLE SEARCH ENGINE")
    print("-"*80)
    
    class SimpleSearchEngine:
        """A basic semantic search engine."""

# %%
# Setup and Imports
# -----------------
        

# %%
#   Init  
# --------

        def __init__(self, documents):
            self.documents = documents
            self.embeddings = embed_batch(documents)
            print(f"Indexed {len(documents)} documents")
        

# %%
# Search
# ------

        def search(self, query, top_k=3):
            """Search for relevant documents."""
            query_emb = embed(query)
            top_indices = top_k_similar(query_emb, self.embeddings, k=top_k)
            
            results = []
            for idx in top_indices:
                sim = cosine_similarity(query_emb, self.embeddings[idx])
                results.append({
                    'document': self.documents[idx],
                    'score': sim,
                    'index': idx
                })
            return results
    
    # Create search engine
    engine = SimpleSearchEngine(documents)
    
    # Perform searches
    test_queries = [
        "artificial intelligence",
        "coding and programming"
    ]
    
    for query in test_queries:
        print(f"\nSearch: '{query}'")
        results = engine.search(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['score']:.4f}] {result['document']}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKEY TAKEAWAYS:")
    print("- Semantic search finds meaning-based matches, not just keywords")
    print("- Use batch_similarity() for efficient query-document comparison")
    print("- top_k_similar() quickly retrieves most relevant results")
    print("- Cosine similarity is ideal for normalized embedding vectors")
    print("- Threshold filtering helps control result quality")


if __name__ == "__main__":
    main()
