"""
Search Methods Example
======================

This example demonstrates different search approaches for RAG systems.

Main concepts:
- Keyword search (BM25-like)
- Semantic search with embeddings
- Hybrid search combining both approaches
- Choosing the right search method for your use case
"""

from kerb.retrieval import (
    Document,
    keyword_search,
    semantic_search,
    hybrid_search
)
from kerb.embedding import embed, embed_batch


def create_sample_documents():
    """Create a sample document collection."""
    return [
        Document(
            id="doc1",
            content="Python is a high-level programming language known for its simplicity and readability. "
                   "It supports object-oriented, functional, and procedural programming paradigms.",
            metadata={"category": "programming", "language": "python", "difficulty": "beginner"}
        ),
        Document(
            id="doc2",
            content="Asynchronous programming in Python allows concurrent execution using async/await syntax. "
                   "This is essential for I/O-bound operations and building scalable applications.",
            metadata={"category": "programming", "language": "python", "difficulty": "intermediate"}
        ),
        Document(
            id="doc3",
            content="JavaScript is the primary language for web development, running in browsers and on servers "
                   "via Node.js. It's event-driven and supports asynchronous programming.",
            metadata={"category": "programming", "language": "javascript", "difficulty": "beginner"}
        ),
        Document(
            id="doc4",
            content="Machine learning models learn patterns from data to make predictions. "
                   "Common algorithms include neural networks, decision trees, and support vector machines.",
            metadata={"category": "ai", "difficulty": "intermediate"}
        ),
        Document(
            id="doc5",
            content="Natural Language Processing (NLP) enables computers to understand human language. "
                   "Applications include chatbots, translation, and sentiment analysis.",
            metadata={"category": "ai", "subcategory": "nlp", "difficulty": "intermediate"}
        ),
        Document(
            id="doc6",
            content="REST APIs provide a standardized way for applications to communicate over HTTP. "
                   "They use standard HTTP methods like GET, POST, PUT, and DELETE.",
            metadata={"category": "web", "difficulty": "beginner"}
        ),
        Document(
            id="doc7",
            content="Docker containerization allows packaging applications with their dependencies. "
                   "Containers are lightweight, portable, and ensure consistency across environments.",
            metadata={"category": "devops", "difficulty": "intermediate"}
        ),
        Document(
            id="doc8",
            content="Kubernetes orchestrates containerized applications at scale. "
                   "It handles deployment, scaling, and management of containerized workloads.",
            metadata={"category": "devops", "difficulty": "advanced"}
        ),
        Document(
            id="doc9",
            content="GraphQL is a query language for APIs that allows clients to request exactly the data they need. "
                   "It provides a more flexible alternative to REST APIs.",
            metadata={"category": "web", "difficulty": "intermediate"}
        ),
        Document(
            id="doc10",
            content="Transformer models revolutionized NLP with attention mechanisms. "
                   "They power modern language models like GPT and BERT.",
            metadata={"category": "ai", "subcategory": "nlp", "difficulty": "advanced"}
        ),
    ]


def main():
    """Run search methods examples."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("SEARCH METHODS FOR RAG SYSTEMS")
    print("="*80)
    
    # Setup
    documents = create_sample_documents()
    print(f"\nCreated {len(documents)} sample documents")
    print("Categories:", set(doc.metadata.get("category") for doc in documents))
    
    
    # 1. Keyword Search (BM25-like)
    print("\n\n1. KEYWORD SEARCH (BM25-LIKE)")
    print("-"*80)
    print("Best for: Exact term matching, technical queries, known terminology\n")
    
    query = "python programming language"
    print(f"Query: '{query}'\n")
    
    results = keyword_search(query, documents, top_k=5)
    
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"  Rank {result.rank} | Score: {result.score:.3f} | {result.document.id}")
        print(f"    {result.document.content[:80]}...")
        print(f"    Category: {result.document.metadata.get('category')}")
    
    # Different keyword queries
    print("\nAdditional keyword searches:")
    test_queries = [
        "asynchronous concurrent",
        "API HTTP REST",
        "container docker kubernetes"
    ]
    
    for tq in test_queries:
        results = keyword_search(tq, documents, top_k=3)
        print(f"\n  '{tq}'")
        print(f"    Top result: {results[0].document.id} (score: {results[0].score:.3f})")
    
    
    # 2. Semantic Search
    print("\n\n2. SEMANTIC SEARCH")
    print("-"*80)
    print("Best for: Conceptual matching, synonyms, understanding intent\n")
    
    # Generate embeddings for documents
    print("Generating embeddings...")
    doc_texts = [doc.content for doc in documents]
    doc_embeddings = embed_batch(doc_texts)
    print(f"Generated {len(doc_embeddings)} document embeddings")
    
    # Semantic query
    query = "building scalable web services"
    print(f"\nQuery: '{query}'")
    query_embedding = embed(query)
    
    results = semantic_search(
        query_embedding=query_embedding,
        documents=documents,
        document_embeddings=doc_embeddings,
        top_k=5
    )
    
    print(f"\nFound {len(results)} results:")
    for result in results:
        print(f"  Rank {result.rank} | Similarity: {result.score:.3f} | {result.document.id}")
        print(f"    {result.document.content[:80]}...")
        print(f"    Category: {result.document.metadata.get('category')}")
    
    # Compare with keyword search
    print("\nComparison: Semantic vs Keyword")
    kw_results = keyword_search(query, documents, top_k=3)
    
    print(f"\n  Query: '{query}'")
    print("\n  Semantic top-3:")
    for r in results[:3]:
        print(f"    {r.document.id}: {r.score:.3f}")
    print("\n  Keyword top-3:")
    for r in kw_results[:3]:
        print(f"    {r.document.id}: {r.score:.3f}")
    
    
    # 3. Hybrid Search
    print("\n\n3. HYBRID SEARCH")
    print("-"*80)
    print("Best for: Combining exact matching with conceptual understanding\n")
    
    query = "async programming patterns"
    print(f"Query: '{query}'\n")
    
    # Generate query embedding
    query_embedding = embed(query)
    
    # Hybrid search with different fusion methods
    print("Testing different fusion methods:\n")
    
    fusion_methods = ["weighted", "rrf", "max"]
    
    for method in fusion_methods:
        results = hybrid_search(
            query=query,
            query_embedding=query_embedding,
            documents=documents,
            document_embeddings=doc_embeddings,
            keyword_weight=0.5,
            semantic_weight=0.5,
            fusion_method=method,
            top_k=3
        )
        
        print(f"  {method.upper()} fusion:")
        for r in results:
            print(f"    {r.rank}. {r.document.id} (score: {r.score:.3f})")
        print()
    
    
    # 4. Weight Tuning
    print("\n4. WEIGHT TUNING")
    print("-"*80)
    print("Adjust keyword/semantic balance for optimal results.\n")
    
    query = "machine learning neural networks"
    query_embedding = embed(query)
    
    print(f"Query: '{query}'\n")
    
    weight_configs = [
        (0.8, 0.2, "Keyword-focused"),
        (0.5, 0.5, "Balanced"),
        (0.2, 0.8, "Semantic-focused")
    ]
    
    for kw_weight, sem_weight, description in weight_configs:
        results = hybrid_search(
            query=query,
            query_embedding=query_embedding,
            documents=documents,
            document_embeddings=doc_embeddings,
            keyword_weight=kw_weight,
            semantic_weight=sem_weight,
            top_k=3
        )
        
        print(f"  {description} (KW:{kw_weight}, SEM:{sem_weight}):")
        for r in results:
            print(f"    {r.rank}. {r.document.id} - {r.document.content[:60]}... ({r.score:.3f})")
        print()
    
    
    # 5. Use Case Selection Guide
    print("\n5. SEARCH METHOD SELECTION GUIDE")
    print("-"*80)
    
    use_cases = [
        {
            "scenario": "User searches for exact error message",
            "query": "TypeError: 'NoneType' object is not subscriptable",
            "recommended": "Keyword",
            "reason": "Exact string matching is crucial"
        },
        {
            "scenario": "User asks conceptual question",
            "query": "How do I make my app handle many users?",
            "recommended": "Semantic",
            "reason": "Need to understand intent (scalability)"
        },
        {
            "scenario": "User searches technical docs",
            "query": "kubernetes deployment configuration",
            "recommended": "Hybrid",
            "reason": "Specific terms + conceptual understanding"
        },
        {
            "scenario": "User needs code examples",
            "query": "async await example",
            "recommended": "Keyword-heavy hybrid",
            "reason": "Specific syntax but flexible on context"
        }
    ]
    
    for uc in use_cases:
        print(f"\n  Scenario: {uc['scenario']}")
        print(f"  Query: '{uc['query']}'")
        print(f"  Recommended: {uc['recommended']}")
        print(f"  Reason: {uc['reason']}")
    
    
    # 6. Performance Comparison
    print("\n\n6. PERFORMANCE CHARACTERISTICS")
    print("-"*80)
    
    print("\nKeyword Search:")
    print("  + Fast, no embeddings needed")
    print("  + Exact term matching")
    print("  + Works well for technical queries")
    print("  - Misses synonyms and paraphrases")
    print("  - Sensitive to vocabulary mismatch")
    
    print("\nSemantic Search:")
    print("  + Understands meaning and intent")
    print("  + Handles synonyms and paraphrases")
    print("  + Better for natural language queries")
    print("  - Requires embedding generation (slower)")
    print("  - May miss exact technical terms")
    
    print("\nHybrid Search:")
    print("  + Best of both approaches")
    print("  + Tunable balance via weights")
    print("  + Robust across query types")
    print("  - More complex to implement")
    print("  - Requires both keyword and semantic infrastructure")
    
    
    print("\n" + "="*80)
    print("Search methods demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    main()
