"""
Re-ranking and Fusion Example
=============================

This example demonstrates result re-ranking and fusion techniques for RAG.

Main concepts:
- Re-ranking results with different strategies
- Reciprocal rank fusion for combining multiple result sets
- Maximal Marginal Relevance (MMR) for diversity
- Custom scoring functions
"""

from kerb.retrieval import (
    Document,
    SearchResult,
    keyword_search,
    rerank_results,
    reciprocal_rank_fusion,
    diversify_results
)


def create_sample_documents():
    """Create a sample document collection with metadata."""
    return [
        Document(
            id="doc1",
            content="Python is a high-level programming language known for its simplicity.",
            metadata={"category": "programming", "views": 1500, "date": "2024-01-15", "author": "Alice"}
        ),
        Document(
            id="doc2",
            content="Python's async/await syntax enables asynchronous programming patterns.",
            metadata={"category": "programming", "views": 800, "date": "2024-03-10", "author": "Bob"}
        ),
        Document(
            id="doc3",
            content="Python offers excellent libraries for data science and machine learning.",
            metadata={"category": "data_science", "views": 2000, "date": "2024-02-20", "author": "Alice"}
        ),
        Document(
            id="doc4",
            content="Asynchronous programming allows handling multiple tasks concurrently.",
            metadata={"category": "programming", "views": 600, "date": "2024-01-05", "author": "Charlie"}
        ),
        Document(
            id="doc5",
            content="Python web frameworks like FastAPI support async/await natively.",
            metadata={"category": "web", "views": 1200, "date": "2024-04-01", "author": "Bob"}
        ),
        Document(
            id="doc6",
            content="Machine learning with Python requires understanding of async patterns for data loading.",
            metadata={"category": "data_science", "views": 900, "date": "2024-03-15", "author": "Alice"}
        ),
        Document(
            id="doc7",
            content="JavaScript also supports asynchronous programming with promises and async/await.",
            metadata={"category": "programming", "views": 1100, "date": "2024-02-28", "author": "Charlie"}
        ),
        Document(
            id="doc8",
            content="Python's asyncio module is the foundation for asynchronous programming.",
            metadata={"category": "programming", "views": 1800, "date": "2024-04-10", "author": "Alice"}
        ),
    ]


def main():
    """Run re-ranking and fusion examples."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("RE-RANKING AND FUSION FOR RAG SYSTEMS")
    print("="*80)
    
    documents = create_sample_documents()
    print(f"\nCreated {len(documents)} sample documents\n")
    
    
    # 1. Basic Re-ranking
    print("\n1. BASIC RE-RANKING STRATEGIES")
    print("-"*80)
    
    query = "python async programming"
    initial_results = keyword_search(query, documents, top_k=6)
    
    print(f"Query: '{query}'")
    print(f"\nInitial keyword search results:")
    for r in initial_results[:4]:
        print(f"  {r.rank}. {r.document.id} (score: {r.score:.3f})")
    
    # Re-rank by relevance
    print("\n  Re-rank by RELEVANCE:")
    relevance_ranked = rerank_results(query, initial_results, method="relevance", top_k=4)
    for r in relevance_ranked:
        print(f"    {r.rank}. {r.document.id} (score: {r.score:.3f})")
    
    # Re-rank by popularity
    print("\n  Re-rank by POPULARITY (views):")
    popularity_ranked = rerank_results(query, initial_results, method="popularity", top_k=4)
    for r in popularity_ranked:
        views = r.document.metadata.get('views', 0)
        print(f"    {r.rank}. {r.document.id} (score: {r.score:.3f}, views: {views})")
    
    # Re-rank by recency
    print("\n  Re-rank by RECENCY (date):")
    recency_ranked = rerank_results(query, initial_results, method="recency", top_k=4)
    for r in recency_ranked:
        date = r.document.metadata.get('date', 'N/A')
        print(f"    {r.rank}. {r.document.id} (score: {r.score:.3f}, date: {date})")
    
    
    # 2. Custom Scoring
    print("\n\n2. CUSTOM SCORING FUNCTIONS")
    print("-"*80)
    print("Define custom re-ranking logic for specific use cases.\n")
    

# %%
# Category Booster
# ----------------

    def category_booster(query: str, doc: Document) -> float:
        """Boost scores for documents in 'programming' category."""
        base_score = 1.0
        if doc.metadata.get('category') == 'programming':
            base_score *= 1.5
        
        # Additional boost for author
        if doc.metadata.get('author') == 'Alice':
            base_score *= 1.2
        
        return base_score
    
    print("Custom scorer: Boost 'programming' category and author 'Alice'\n")
    custom_ranked = rerank_results(
        query, 
        initial_results, 
        method="custom", 
        scorer=category_booster,
        top_k=4
    )
    
    for r in custom_ranked:
        category = r.document.metadata.get('category')
        author = r.document.metadata.get('author')
        print(f"  {r.rank}. {r.document.id} (score: {r.score:.3f})")
        print(f"       Category: {category}, Author: {author}")
    
    
    # 3. Reciprocal Rank Fusion (RRF)
    print("\n\n3. RECIPROCAL RANK FUSION")
    print("-"*80)
    print("Combine multiple ranked lists into a single unified ranking.\n")
    
    # Create multiple result sets
    query1 = "python programming"
    query2 = "async await"
    query3 = "asynchronous"
    
    results1 = keyword_search(query1, documents, top_k=5)
    results2 = keyword_search(query2, documents, top_k=5)
    results3 = keyword_search(query3, documents, top_k=5)
    
    print("Combining results from 3 queries:")
    print(f"  1. '{query1}'")
    print(f"  2. '{query2}'")
    print(f"  3. '{query3}'")
    
    # Fuse the results
    fused_results = reciprocal_rank_fusion([results1, results2, results3], k=60)
    
    print(f"\nFused top results:")
    for r in fused_results[:5]:
        print(f"  {r.rank}. {r.document.id} (score: {r.score:.3f})")
        print(f"       {r.document.content[:70]}...")
    
    
    # 4. Diversity with MMR
    print("\n\n4. MAXIMAL MARGINAL RELEVANCE (MMR)")
    print("-"*80)
    print("Balance relevance with diversity to avoid redundant results.\n")
    
    query = "python async"
    initial_results = keyword_search(query, documents, top_k=8)
    
    print(f"Query: '{query}'")
    print(f"\nBefore diversity (top 5):")
    for r in initial_results[:5]:
        print(f"  {r.rank}. {r.document.id}")
        print(f"       {r.document.content[:60]}...")
    
    # Apply diversification
    diverse_results = diversify_results(
        initial_results,
        max_results=5,
        diversity_factor=0.5  # 0 = pure relevance, 1 = pure diversity
    )
    
    print(f"\nAfter diversity (top 5):")
    for r in diverse_results:
        print(f"  {r.rank}. {r.document.id}")
        print(f"       {r.document.content[:60]}...")
    
    # Compare different diversity factors
    print("\n\nDiversity factor comparison:")
    for diversity in [0.0, 0.3, 0.7, 1.0]:
        diverse = diversify_results(initial_results, max_results=3, diversity_factor=diversity)
        doc_ids = [r.document.id for r in diverse]
        print(f"  diversity={diversity}: {doc_ids}")
    
    
    # 5. Multi-Stage Re-ranking Pipeline
    print("\n\n5. MULTI-STAGE RE-RANKING PIPELINE")
    print("-"*80)
    print("Combine multiple re-ranking strategies for optimal results.\n")
    
    query = "python async web development"
    print(f"Query: '{query}'\n")
    
    # Stage 1: Initial retrieval
    stage1 = keyword_search(query, documents, top_k=8)
    print(f"Stage 1 - Initial retrieval: {len(stage1)} results")
    
    # Stage 2: Re-rank by relevance
    stage2 = rerank_results(query, stage1, method="relevance", top_k=6)
    print(f"Stage 2 - Relevance re-rank: {len(stage2)} results")
    
    # Stage 3: Apply diversity
    stage3 = diversify_results(stage2, max_results=4, diversity_factor=0.4)
    print(f"Stage 3 - Diversification: {len(stage3)} results")
    
    # Stage 4: Final popularity boost
    stage4 = rerank_results(query, stage3, method="popularity", top_k=3)
    print(f"Stage 4 - Popularity boost: {len(stage4)} results")
    
    print(f"\nFinal ranked results:")
    for r in stage4:
        print(f"  {r.rank}. {r.document.id} (score: {r.score:.3f})")
        print(f"       {r.document.content[:70]}...")
        print(f"       Views: {r.document.metadata.get('views')}, "
              f"Category: {r.document.metadata.get('category')}")
    
    
    # 6. Fusion Strategy Comparison
    print("\n\n6. FUSION STRATEGY COMPARISON")
    print("-"*80)
    
    # Create diverse query sets
    q1_results = keyword_search("python async", documents, top_k=5)
    q2_results = keyword_search("asynchronous programming", documents, top_k=5)
    
    print("Comparing different fusion approaches:\n")
    
    # RRF with different k values
    for k_val in [10, 60, 100]:
        fused = reciprocal_rank_fusion([q1_results, q2_results], k=k_val)
        top_ids = [r.document.id for r in fused[:3]]
        print(f"  RRF (k={k_val:3d}): {top_ids}")
    
    
    # 7. Use Case Examples
    print("\n\n7. RAG RE-RANKING USE CASES")
    print("-"*80)
    
    scenarios = [
        {
            "name": "News/Blog Search",
            "strategy": "recency",
            "reason": "Users want latest information"
        },
        {
            "name": "Documentation Search",
            "strategy": "relevance",
            "reason": "Accuracy more important than recency"
        },
        {
            "name": "Community Q&A",
            "strategy": "popularity",
            "reason": "Popular answers are often better"
        },
        {
            "name": "Research Paper Search",
            "strategy": "diversity",
            "reason": "Users want variety of perspectives"
        },
        {
            "name": "E-commerce Product Search",
            "strategy": "multi-stage (relevance + popularity + recency)",
            "reason": "Balance multiple signals"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n  Use Case: {scenario['name']}")
        print(f"  Strategy: {scenario['strategy']}")
        print(f"  Rationale: {scenario['reason']}")
    
    
    print("\n" + "="*80)
    print("Re-ranking and fusion demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    main()
