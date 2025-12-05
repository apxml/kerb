"""
Context Management Example
==========================

This example demonstrates context compression and filtering for RAG systems.

Main concepts:
- Compressing context to fit token limits
- Filtering results by score, metadata, and deduplication
- Managing context window constraints
- Optimizing context for LLM consumption
"""

from kerb.retrieval import (
    Document,
    SearchResult,
    keyword_search,
    compress_context,
    filter_results,
    FilterConfig
)


def create_sample_documents():
    """Create sample documents with varying lengths."""
    return [
        Document(
            id="doc1",
            content="Python is a high-level, interpreted programming language known for its simplicity "
                   "and readability. It supports multiple programming paradigms including procedural, "
                   "object-oriented, and functional programming. Python's design philosophy emphasizes "
                   "code readability with significant use of whitespace.",
            metadata={"category": "python", "length": "medium", "quality": 0.9}
        ),
        Document(
            id="doc2",
            content="Async/await in Python enables asynchronous programming, allowing code to handle "
                   "I/O operations without blocking. This is particularly useful for web servers, "
                   "database queries, and other I/O-bound tasks.",
            metadata={"category": "python", "length": "short", "quality": 0.85}
        ),
        Document(
            id="doc3",
            content="Python's standard library includes asyncio, a framework for writing concurrent code. "
                   "It provides event loops, coroutines, tasks, and synchronization primitives. "
                   "The async/await syntax makes asynchronous code look and behave more like synchronous code, "
                   "making it easier to read and maintain.",
            metadata={"category": "python", "length": "medium", "quality": 0.95}
        ),
        Document(
            id="doc4",
            content="FastAPI is a modern web framework for building APIs with Python. It's built on "
                   "Starlette and Pydantic, offering automatic API documentation, data validation, "
                   "and native async support.",
            metadata={"category": "web", "length": "short", "quality": 0.8}
        ),
        Document(
            id="doc5",
            content="When working with async Python, you need to understand event loops, coroutines, "
                   "and the difference between concurrent and parallel execution. The asyncio module "
                   "provides the core infrastructure, while libraries like aiohttp extend it for "
                   "HTTP operations.",
            metadata={"category": "python", "length": "medium", "quality": 0.88}
        ),
        Document(
            id="doc6",
            content="Type hints in Python improve code quality and enable better IDE support. "
                   "They're especially useful in async code where return types can be complex.",
            metadata={"category": "python", "length": "short", "quality": 0.75}
        ),
        Document(
            id="doc7",
            content="Error handling in async Python requires special attention. You need to handle "
                   "exceptions in coroutines, manage task cancellation properly, and understand "
                   "how exceptions propagate through the event loop. Using try/except blocks in "
                   "async functions works similarly to synchronous code, but you also need to "
                   "consider CancelledError and other async-specific exceptions.",
            metadata={"category": "python", "length": "long", "quality": 0.92}
        ),
        Document(
            id="doc8",
            content="Python performance optimization often involves profiling, caching, and using "
                   "compiled extensions. For async code, focus on reducing I/O wait times.",
            metadata={"category": "performance", "length": "short", "quality": 0.7}
        ),
    ]


def main():
    """Run context management examples."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("CONTEXT MANAGEMENT FOR RAG SYSTEMS")
    print("="*80)
    
    documents = create_sample_documents()
    print(f"\nCreated {len(documents)} sample documents")
    
    # Calculate total content length
    total_chars = sum(len(doc.content) for doc in documents)
    print(f"Total content: ~{total_chars} characters (~{total_chars // 4} tokens)")
    
    
    # 1. Basic Context Compression
    print("\n\n1. BASIC CONTEXT COMPRESSION")
    print("-"*80)
    print("Compress results to fit within token limits.\n")
    
    query = "python async programming"
    results = keyword_search(query, documents, top_k=8)
    
    print(f"Initial results: {len(results)} documents")
    total_tokens = sum(len(r.document.content) // 4 for r in results)
    print(f"Estimated tokens: ~{total_tokens}")
    
    # Compress to different token limits
    token_limits = [500, 300, 150]
    
    for max_tokens in token_limits:
        compressed = compress_context(query, results, max_tokens=max_tokens)
        actual_tokens = sum(len(r.document.content) // 4 for r in compressed)
        print(f"\n  Compressed to {max_tokens} tokens:")
        print(f"    Results: {len(compressed)} documents")
        print(f"    Actual tokens: ~{actual_tokens}")
        print(f"    Documents: {[r.document.id for r in compressed]}")
    
    
    # 2. Compression Strategies
    print("\n\n2. COMPRESSION STRATEGIES")
    print("-"*80)
    
    results = keyword_search(query, documents, top_k=8)
    
    strategies = ["top_k", "score_threshold", "relevance_weighted"]
    max_tokens = 400
    
    print(f"Target: {max_tokens} tokens\n")
    
    for strategy in strategies:
        compressed = compress_context(
            query, 
            results, 
            max_tokens=max_tokens,
            strategy=strategy
        )
        actual_tokens = sum(len(r.document.content) // 4 for r in compressed)
        
        print(f"  Strategy: {strategy}")
        print(f"    Results: {len(compressed)} documents")
        print(f"    Tokens: ~{actual_tokens}")
        print(f"    Doc IDs: {[r.document.id for r in compressed]}")
        print()
    
    
    # 3. Result Filtering
    print("\n3. RESULT FILTERING")
    print("-"*80)
    print("Filter results based on various criteria.\n")
    
    results = keyword_search(query, documents, top_k=8)
    
    # Filter by minimum score
    print("  Filter by score >= 0.5:")
    filtered = filter_results(results, min_score=0.5)
    print(f"    {len(results)} -> {len(filtered)} results")
    for r in filtered[:4]:
        print(f"      {r.document.id}: {r.score:.3f}")
    
    # Filter by metadata
    print("\n  Filter by category = 'python':")
    filtered = filter_results(results, metadata_filter={"category": "python"})
    print(f"    {len(results)} -> {len(filtered)} results")
    for r in filtered[:4]:
        print(f"      {r.document.id}: {r.document.metadata.get('category')}")
    
    # Filter by quality threshold
    print("\n  Filter by quality >= 0.85:")
    # Note: Lambda filters not directly supported, filter manually
    filtered = [r for r in results if r.document.metadata.get('quality', 0) >= 0.85]
    print(f"    {len(results)} -> {len(filtered)} results")
    for r in filtered:
        quality = r.document.metadata.get('quality')
        print(f"      {r.document.id}: quality={quality}")
    
    
    # Deduplication
    print("\n\n4. DEDUPLICATION")
    print("-"*80)
    print("Remove duplicate or very similar results.\n")
    
    # Create results with duplicates
    duplicated_results = results + results[:3]  # Add some duplicates
    
    print(f"Before deduplication: {len(duplicated_results)} results")
    print(f"  IDs: {[r.document.id for r in duplicated_results]}")
    
    # Deduplicate by using dedup_threshold
    deduplicated = filter_results(duplicated_results, dedup_threshold=0.9)
    
    print(f"\nAfter deduplication: {len(deduplicated)} results")
    print(f"  IDs: {[r.document.id for r in deduplicated]}")
    
    
    # 5. Advanced Filtering with FilterConfig
    print("\n\n5. ADVANCED FILTERING WITH FILTERCONFIG")
    print("-"*80)
    
    results = keyword_search(query, documents, top_k=8)
    
    # Create filter configuration
    filter_config = FilterConfig(
        min_score=0.3,
        max_results=5,
        metadata_filter={"category": "python"},
        dedup_threshold=0.9
    )
    
    print(f"Filter configuration:")
    print(f"  - Min score: {filter_config.min_score}")
    print(f"  - Max results: {filter_config.max_results}")
    print(f"  - Category: python")
    print(f"  - Dedup threshold: {filter_config.dedup_threshold}")
    
    filtered = filter_results(results, config=filter_config)
    
    print(f"\nResults: {len(filtered)} documents")
    for r in filtered:
        print(f"  {r.document.id}: score={r.score:.3f}, "
              f"category={r.document.metadata.get('category')}")
    
    
    # 6. Context Window Management
    print("\n\n6. CONTEXT WINDOW MANAGEMENT")
    print("-"*80)
    print("Manage token budgets for different LLM context windows.\n")
    
    results = keyword_search(query, documents, top_k=8)
    
    # Different model context windows
    models = [
        ("GPT-3.5", 4096, 2000),   # model, total_context, budget_for_results
        ("GPT-4", 8192, 4000),
        ("Claude", 100000, 8000),
        ("Llama-2-7B", 4096, 1500)
    ]
    
    for model_name, total_context, result_budget in models:
        compressed = compress_context(query, results, max_tokens=result_budget)
        actual_tokens = sum(len(r.document.content) // 4 for r in compressed)
        
        print(f"  {model_name:15} | Context: {total_context:6} | Budget: {result_budget:4} | "
              f"Used: ~{actual_tokens:4} | Docs: {len(compressed)}")
    
    
    # 7. Dynamic Context Allocation
    print("\n\n7. DYNAMIC CONTEXT ALLOCATION")
    print("-"*80)
    print("Allocate context based on result importance.\n")
    
    results = keyword_search(query, documents, top_k=6)
    

# %%
# Allocate Tokens
# ---------------

    def allocate_tokens(results, total_budget):
        """Allocate more tokens to higher-ranked results."""
        allocated = []
        remaining_budget = total_budget
        
        for i, result in enumerate(results):
            # Higher ranks get more space
            weight = 1.0 / (i + 1)
            doc_tokens = len(result.document.content) // 4
            
            # Allocate proportional tokens
            if remaining_budget > 0:
                allocated_tokens = min(doc_tokens, remaining_budget)
                allocated.append({
                    'doc_id': result.document.id,
                    'rank': i + 1,
                    'score': result.score,
                    'tokens': allocated_tokens,
                    'weight': weight
                })
                remaining_budget -= allocated_tokens
        
        return allocated
    
    allocation = allocate_tokens(results, total_budget=500)
    
    print(f"Total budget: 500 tokens\n")
    print(f"  Rank | Doc ID | Score | Allocated Tokens | Weight")
    print(f"  " + "-"*55)
    for item in allocation:
        print(f"  {item['rank']:4} | {item['doc_id']:6} | {item['score']:.3f} | "
              f"{item['tokens']:15} | {item['weight']:.3f}")
    
    total_allocated = sum(item['tokens'] for item in allocation)
    print(f"\n  Total allocated: {total_allocated} tokens")
    
    
    # 8. Context Optimization Pipeline
    print("\n\n8. CONTEXT OPTIMIZATION PIPELINE")
    print("-"*80)
    print("Complete pipeline for production RAG systems.\n")
    
    def optimize_context_for_llm(query: str, documents, max_tokens: int = 2000):
        """Optimize context for LLM consumption."""
        # Step 1: Initial retrieval
        results = keyword_search(query, documents, top_k=10)
        print(f"  Step 1: Retrieved {len(results)} documents")
        
        # Step 2: Filter by quality and relevance
        filtered = filter_results(
            results,
            min_score=0.3,
            metadata_filter={"quality": 0.75}  # Exact match for quality >= 0.75 would need manual filter
        )
        # Manual filter for quality >= 0.75
        filtered = [r for r in filtered if r.document.metadata.get('quality', 0) >= 0.75]
        print(f"  Step 2: Filtered to {len(filtered)} high-quality documents")
        
        # Step 3: Deduplicate
        deduped = filter_results(filtered, dedup_threshold=0.9)
        print(f"  Step 3: Deduplicated to {len(deduped)} unique documents")
        
        # Step 4: Compress to token limit
        compressed = compress_context(query, deduped, max_tokens=max_tokens)
        actual_tokens = sum(len(r.document.content) // 4 for r in compressed)
        print(f"  Step 4: Compressed to {len(compressed)} documents (~{actual_tokens} tokens)")
        
        return compressed
    
    print("Running optimization pipeline:\n")
    optimized = optimize_context_for_llm(query, documents, max_tokens=400)
    
    print(f"\nFinal optimized context:")
    for r in optimized:
        print(f"  {r.document.id}: {len(r.document.content) // 4} tokens, "
              f"score={r.score:.3f}, quality={r.document.metadata.get('quality')}")
    
    
    print("\n" + "="*80)
    print("Context management demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    main()
