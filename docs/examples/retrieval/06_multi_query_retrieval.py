"""Multi-Query Retrieval Example

This example demonstrates advanced multi-query retrieval patterns.

Main concepts:
- Query decomposition for complex questions
- Multi-query retrieval with fusion
- Query expansion strategies
- Parallel retrieval optimization
"""

from kerb.retrieval import (
    Document,
    expand_query,
    generate_sub_queries,
    keyword_search,
    semantic_search,
    reciprocal_rank_fusion,
    rerank_results
)
from kerb.embedding import embed


def create_sample_documents():
    """Create a diverse document collection."""
    return [
        Document(
            id="doc1",
            content="Python supports multiple programming paradigms including object-oriented, functional, and procedural programming.",
            metadata={"lang": "python", "topic": "paradigms"}
        ),
        Document(
            id="doc2",
            content="JavaScript is single-threaded but supports asynchronous programming through callbacks, promises, and async/await.",
            metadata={"lang": "javascript", "topic": "async"}
        ),
        Document(
            id="doc3",
            content="Rust provides memory safety without garbage collection through its ownership system and borrow checker.",
            metadata={"lang": "rust", "topic": "memory"}
        ),
        Document(
            id="doc4",
            content="Type systems in programming languages help catch errors at compile time and improve code documentation.",
            metadata={"topic": "types"}
        ),
        Document(
            id="doc5",
            content="Concurrency in Go is achieved through goroutines and channels, making concurrent programming more accessible.",
            metadata={"lang": "go", "topic": "concurrency"}
        ),
        Document(
            id="doc6",
            content="Functional programming emphasizes immutability, pure functions, and higher-order functions.",
            metadata={"topic": "paradigms"}
        ),
        Document(
            id="doc7",
            content="Python's asyncio library enables writing asynchronous code using async/await syntax, similar to JavaScript.",
            metadata={"lang": "python", "topic": "async"}
        ),
        Document(
            id="doc8",
            content="Memory management in C requires manual allocation and deallocation, which can lead to bugs if not done carefully.",
            metadata={"lang": "c", "topic": "memory"}
        ),
        Document(
            id="doc9",
            content="Static typing catches errors before runtime, while dynamic typing offers more flexibility during development.",
            metadata={"topic": "types"}
        ),
        Document(
            id="doc10",
            content="Thread-based concurrency can lead to race conditions and deadlocks if not properly synchronized.",
            metadata={"topic": "concurrency"}
        ),
    ]


def main():
    """Run multi-query retrieval examples."""
    
    print("="*80)
    print("MULTI-QUERY RETRIEVAL PATTERNS")
    print("="*80)
    
    documents = create_sample_documents()
    print(f"\nDocument collection: {len(documents)} documents")
    
    
    # 1. Query Decomposition
    print("\n\n1. QUERY DECOMPOSITION")
    print("-"*80)
    print("Break complex queries into simpler sub-queries.\n")
    
    complex_query = "Compare memory management in Python and Rust, and explain their concurrency models"
    print(f"Complex query:\n  '{complex_query}'\n")
    
    sub_queries = generate_sub_queries(complex_query, max_queries=4)
    print(f"Generated {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")
    
    # Retrieve for each sub-query
    print("\nRetrieving for each sub-query:")
    all_results = []
    for i, sq in enumerate(sub_queries, 1):
        results = keyword_search(sq, documents, top_k=3)
        all_results.append(results)
        doc_ids = [r.document.id for r in results]
        print(f"  Sub-query {i}: {doc_ids}")
    
    # Fuse results
    fused = reciprocal_rank_fusion(all_results, k=60)
    print(f"\nFused results (top 5):")
    for r in fused[:5]:
        print(f"  {r.rank}. {r.document.id} (score: {r.score:.3f})")
        print(f"     {r.document.content[:70]}...")
    
    
    # 2. Query Expansion
    print("\n\n2. QUERY EXPANSION STRATEGIES")
    print("-"*80)
    print("Generate multiple variations of a query for broader coverage.\n")
    
    base_query = "asynchronous programming"
    print(f"Base query: '{base_query}'\n")
    
    # Expand with different methods
    expansion_methods = ["synonyms", "related", "specificity"]
    
    for method in expansion_methods:
        expanded = expand_query(base_query, method=method)
        print(f"{method.upper()} expansion:")
        for i, variant in enumerate(expanded[:3], 1):
            print(f"  {i}. {variant}")
        print()
    
    
    # 3. Multi-Variant Retrieval
    print("\n3. MULTI-VARIANT RETRIEVAL")
    print("-"*80)
    print("Retrieve using multiple query variants and fuse results.\n")
    
    query = "concurrent programming"
    print(f"Original query: '{query}'\n")
    
    # Generate variants
    variants = expand_query(query, method="related")[:3]
    print(f"Query variants:")
    for i, v in enumerate(variants, 1):
        print(f"  {i}. {v}")
    
    # Retrieve for each variant
    print(f"\nRetrieving with each variant:")
    variant_results = []
    for i, variant in enumerate(variants, 1):
        results = keyword_search(variant, documents, top_k=4)
        variant_results.append(results)
        print(f"  Variant {i}: {len(results)} results")
    
    # Fuse
    fused = reciprocal_rank_fusion(variant_results, k=60)
    print(f"\nFused top results:")
    for r in fused[:5]:
        print(f"  {r.document.id}: {r.document.metadata.get('topic')}")
    
    
    # 4. Hybrid Multi-Query Strategy
    print("\n\n4. HYBRID MULTI-QUERY STRATEGY")
    print("-"*80)
    print("Combine decomposition and expansion for comprehensive retrieval.\n")
    
    user_query = "How do type systems improve code quality?"
    print(f"User query: '{user_query}'\n")
    
    # Step 1: Decompose
    sub_queries = generate_sub_queries(user_query, max_queries=2)
    print(f"Step 1 - Decomposition:")
    for sq in sub_queries:
        print(f"  - {sq}")
    
    # Step 2: Expand each sub-query
    print(f"\nStep 2 - Expansion:")
    expanded_subs = []
    for sq in sub_queries:
        expanded = expand_query(sq, method="synonyms")[:2]
        expanded_subs.extend(expanded)
        print(f"  - {sq}")
        for exp in expanded:
            print(f"    -> {exp}")
    
    # Step 3: Retrieve for all variants
    print(f"\nStep 3 - Retrieval:")
    all_variant_results = []
    for variant in expanded_subs[:4]:  # Limit to 4 to keep it manageable
        results = keyword_search(variant, documents, top_k=3)
        all_variant_results.append(results)
    print(f"  Retrieved from {len(all_variant_results)} variants")
    
    # Step 4: Fuse
    final_results = reciprocal_rank_fusion(all_variant_results, k=60)
    print(f"\nStep 4 - Fusion (top 4):")
    for r in final_results[:4]:
        print(f"  {r.document.id}: {r.document.content[:60]}...")
    
    
    # 5. Weighted Multi-Query Fusion
    print("\n\n5. WEIGHTED MULTI-QUERY FUSION")
    print("-"*80)
    print("Give different weights to different query types.\n")
    
    query = "memory safety"
    print(f"Query: '{query}'\n")
    
    # Different query types
    exact_query = query
    expanded_query = expand_query(query, method="related")[0]
    detailed_query = f"detailed information about {query}"
    
    print(f"Query variations:")
    print(f"  Exact:    '{exact_query}'")
    print(f"  Expanded: '{expanded_query}'")
    print(f"  Detailed: '{detailed_query}'")
    
    # Retrieve
    exact_results = keyword_search(exact_query, documents, top_k=5)
    expanded_results = keyword_search(expanded_query, documents, top_k=5)
    detailed_results = keyword_search(detailed_query, documents, top_k=5)
    
    # Simulate weighted fusion by adjusting scores
    print(f"\nApplying weights (exact: 2x, expanded: 1.5x, detailed: 1x):")
    
    for r in exact_results:
        r.score *= 2.0
    for r in expanded_results:
        r.score *= 1.5
    
    # Fuse
    weighted_fused = reciprocal_rank_fusion([exact_results, expanded_results, detailed_results])
    
    print(f"Weighted fused results:")
    for r in weighted_fused[:4]:
        print(f"  {r.document.id} (score: {r.score:.3f})")
    
    
    # 6. Semantic Multi-Query
    print("\n\n6. SEMANTIC MULTI-QUERY RETRIEVAL")
    print("-"*80)
    print("Use embeddings with multiple query variants.\n")
    
    from kerb.embedding import embed_batch
    
    query = "programming language features"
    print(f"Base query: '{query}'\n")
    
    # Generate variants
    variants = expand_query(query, method="specificity")[:3]
    print(f"Variants:")
    for i, v in enumerate(variants, 1):
        print(f"  {i}. {v}")
    
    # Embed documents and queries
    doc_texts = [doc.content for doc in documents]
    doc_embeddings = embed_batch(doc_texts)
    variant_embeddings = embed_batch(variants)
    
    # Retrieve for each variant
    print(f"\nSemantic search for each variant:")
    semantic_results = []
    for i, variant_emb in enumerate(variant_embeddings, 1):
        results = semantic_search(
            query_embedding=variant_emb,
            documents=documents,
            document_embeddings=doc_embeddings,
            top_k=4
        )
        semantic_results.append(results)
        print(f"  Variant {i}: {[r.document.id for r in results[:3]]}")
    
    # Fuse
    fused_semantic = reciprocal_rank_fusion(semantic_results, k=60)
    print(f"\nFused semantic results:")
    for r in fused_semantic[:5]:
        print(f"  {r.document.id}: {r.score:.3f}")
    
    
    # 7. Adaptive Multi-Query
    print("\n\n7. ADAPTIVE MULTI-QUERY STRATEGY")
    print("-"*80)
    print("Adjust strategy based on initial results.\n")
    
    query = "type checking"
    print(f"Query: '{query}'\n")
    
    # Initial retrieval
    initial = keyword_search(query, documents, top_k=5)
    print(f"Initial retrieval: {len(initial)} results")
    print(f"  Average score: {sum(r.score for r in initial) / len(initial):.3f}")
    
    # Check if we need expansion
    avg_score = sum(r.score for r in initial) / len(initial)
    
    if avg_score < 1.0:  # Low scores suggest poor match
        print(f"\nLow scores detected - applying query expansion")
        
        expanded_queries = expand_query(query, method="related")[:2]
        print(f"  Expanded queries: {expanded_queries}")
        
        expansion_results = []
        for eq in expanded_queries:
            results = keyword_search(eq, documents, top_k=5)
            expansion_results.append(results)
        
        # Fuse with original
        final = reciprocal_rank_fusion([initial] + expansion_results, k=60)
        print(f"\nAfter expansion: {len(final)} unique results")
        print(f"  Top results: {[r.document.id for r in final[:5]]}")
    else:
        print(f"\nGood scores - using initial results")
        final = initial
    
    
    # 8. Production Multi-Query Pipeline
    print("\n\n8. PRODUCTION MULTI-QUERY PIPELINE")
    print("-"*80)
    
    def multi_query_rag(query_text, documents, max_queries=3):
        """Production-ready multi-query retrieval."""
        results = []
        
        # 1. Use original query
        original_results = keyword_search(query_text, documents, top_k=6)
        results.append(original_results)
        
        # 2. Generate and use sub-queries
        sub_queries = generate_sub_queries(query_text, max_queries=max_queries)
        for sq in sub_queries[:2]:  # Limit to 2
            sq_results = keyword_search(sq, documents, top_k=5)
            results.append(sq_results)
        
        # 3. Use expanded query
        expanded = expand_query(query_text, method="related")
        if expanded:
            exp_results = keyword_search(expanded[0], documents, top_k=5)
            results.append(exp_results)
        
        # 4. Fuse all results
        fused = reciprocal_rank_fusion(results, k=60)
        
        # 5. Re-rank for final quality
        reranked = rerank_results(query_text, fused, method="relevance", top_k=5)
        
        return reranked
    
    test_query = "comparing programming paradigms"
    print(f"Query: '{test_query}'\n")
    print("Running multi-query pipeline...")
    
    final_results = multi_query_rag(test_query, documents)
    
    print(f"\nFinal results ({len(final_results)} documents):")
    for r in final_results:
        print(f"  {r.rank}. {r.document.id} (score: {r.score:.3f})")
        print(f"     {r.document.content[:70]}...")
    
    
    print("\n" + "="*80)
    print("Multi-query retrieval demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    main()
