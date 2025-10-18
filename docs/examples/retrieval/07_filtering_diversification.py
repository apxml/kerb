"""Filtering and Diversification Example

This example demonstrates result filtering and diversity optimization.

Main concepts:
- Filtering by scores and metadata
- Deduplication strategies
- Maximal Marginal Relevance (MMR) for diversity
- Balancing relevance and diversity
"""

from kerb.retrieval import (
    Document,
    SearchResult,
    keyword_search,
    filter_results,
    diversify_results,
    FilterConfig
)


def create_sample_documents():
    """Create documents with metadata for filtering."""
    return [
        Document(
            id="py-web-1",
            content="FastAPI is a modern Python web framework with automatic API documentation and async support.",
            metadata={"lang": "python", "category": "web", "year": 2024, "quality": 0.95, "author": "alice"}
        ),
        Document(
            id="py-web-2",
            content="Flask is a lightweight Python web framework that's easy to get started with.",
            metadata={"lang": "python", "category": "web", "year": 2023, "quality": 0.85, "author": "bob"}
        ),
        Document(
            id="py-web-3",
            content="Django is a full-featured Python web framework with built-in admin panel and ORM.",
            metadata={"lang": "python", "category": "web", "year": 2024, "quality": 0.90, "author": "alice"}
        ),
        Document(
            id="py-data-1",
            content="Pandas is the essential Python library for data manipulation and analysis.",
            metadata={"lang": "python", "category": "data", "year": 2024, "quality": 0.92, "author": "charlie"}
        ),
        Document(
            id="py-data-2",
            content="NumPy provides support for large multi-dimensional arrays and mathematical functions in Python.",
            metadata={"lang": "python", "category": "data", "year": 2023, "quality": 0.88, "author": "alice"}
        ),
        Document(
            id="py-ml-1",
            content="scikit-learn is a machine learning library offering simple and efficient data mining tools.",
            metadata={"lang": "python", "category": "ml", "year": 2024, "quality": 0.94, "author": "bob"}
        ),
        Document(
            id="js-web-1",
            content="React is a JavaScript library for building user interfaces with reusable components.",
            metadata={"lang": "javascript", "category": "web", "year": 2024, "quality": 0.91, "author": "charlie"}
        ),
        Document(
            id="js-web-2",
            content="Express.js is a minimal Node.js web application framework.",
            metadata={"lang": "javascript", "category": "web", "year": 2023, "quality": 0.82, "author": "alice"}
        ),
        Document(
            id="py-async-1",
            content="Python's asyncio enables writing asynchronous code for concurrent I/O operations.",
            metadata={"lang": "python", "category": "async", "year": 2024, "quality": 0.89, "author": "bob"}
        ),
        Document(
            id="py-test-1",
            content="pytest is a mature testing framework for Python with powerful fixtures and plugins.",
            metadata={"lang": "python", "category": "testing", "year": 2024, "quality": 0.87, "author": "charlie"}
        ),
    ]


def main():
    """Run filtering and diversification examples."""
    
    print("="*80)
    print("FILTERING AND DIVERSIFICATION")
    print("="*80)
    
    documents = create_sample_documents()
    print(f"\nCreated {len(documents)} documents")
    print(f"Categories: {set(d.metadata['category'] for d in documents)}")
    print(f"Languages: {set(d.metadata['lang'] for d in documents)}")
    
    
    # 1. Score-Based Filtering
    print("\n\n1. SCORE-BASED FILTERING")
    print("-"*80)
    
    query = "python web framework"
    results = keyword_search(query, documents, top_k=10)
    
    print(f"Query: '{query}'")
    print(f"Initial results: {len(results)} documents\n")
    
    # Different score thresholds
    thresholds = [0.0, 0.5, 1.0, 2.0]
    
    for threshold in thresholds:
        filtered = filter_results(results, min_score=threshold)
        print(f"  Threshold {threshold:.1f}: {len(filtered)} results")
        if filtered:
            scores = [r.score for r in filtered]
            print(f"    Score range: {min(scores):.3f} - {max(scores):.3f}")
    
    
    # 2. Metadata Filtering
    print("\n\n2. METADATA FILTERING")
    print("-"*80)
    
    query = "python"
    results = keyword_search(query, documents, top_k=10)
    
    print(f"Query: '{query}'")
    print(f"Initial results: {len(results)} documents\n")
    
    # Filter by language
    print("Filter by language='python':")
    python_only = filter_results(results, metadata_filter={"lang": "python"})
    print(f"  {len(results)} -> {len(python_only)} results")
    for r in python_only[:4]:
        print(f"    {r.document.id}: {r.document.metadata['lang']}")
    
    # Filter by category
    print("\nFilter by category='web':")
    web_only = filter_results(results, metadata_filter={"category": "web"})
    print(f"  {len(results)} -> {len(web_only)} results")
    for r in web_only:
        print(f"    {r.document.id}: {r.document.metadata['category']}")
    
    # Filter by year
    print("\nFilter by year=2024:")
    recent = filter_results(results, metadata_filter={"year": 2024})
    print(f"  {len(results)} -> {len(recent)} results")
    for r in recent[:4]:
        print(f"    {r.document.id}: {r.document.metadata['year']}")
    
    
    # 3. Lambda-Based Filtering
    print("\n\n3. MANUAL FILTERING FOR COMPLEX CONDITIONS")
    print("-"*80)
    print("Use manual filtering for lambda-like complex logic.\n")
    
    results = keyword_search("python", documents, top_k=10)
    
    # Filter by quality >= 0.9
    print("Filter by quality >= 0.9:")
    high_quality = [r for r in results if r.document.metadata.get('quality', 0) >= 0.9]
    print(f"  {len(results)} -> {len(high_quality)} results")
    for r in high_quality:
        print(f"    {r.document.id}: quality={r.document.metadata['quality']}")
    
    # Filter by author in list
    print("\nFilter by author in ['alice', 'bob']:")
    author_filter = [r for r in results if r.document.metadata.get('author') in ["alice", "bob"]]
    print(f"  {len(results)} -> {len(author_filter)} results")
    for r in author_filter[:4]:
        print(f"    {r.document.id}: author={r.document.metadata['author']}")
    
    
    # 4. Combined Filtering
    print("\n\n4. COMBINED FILTERING")
    print("-"*80)
    print("Apply multiple filters simultaneously.\n")
    
    results = keyword_search("python", documents, top_k=10)
    
    # First apply basic filters
    combined = filter_results(
        results,
        min_score=0.5,
        max_results=10,
        metadata_filter={
            "lang": "python",
            "year": 2024
        }
    )
    
    # Then manually filter for quality
    combined = [r for r in combined if r.document.metadata.get('quality', 0) >= 0.85]
    combined = combined[:5]  # Limit to 5
    
    print("Filters applied:")
    print("  - Min score: 0.5")
    print("  - Language: python")
    print("  - Quality: >= 0.85")
    print("  - Year: 2024")
    print("  - Max results: 5")
    
    print(f"\nResults: {len(combined)} documents")
    for r in combined:
        print(f"  {r.document.id}: score={r.score:.3f}, quality={r.document.metadata['quality']}")
    
    
    # 5. Deduplication
    print("\n\n5. DEDUPLICATION")
    print("-"*80)
    
    # Create duplicates
    results = keyword_search("python web", documents, top_k=6)
    duplicated = results + results[:3]
    
    print(f"Before deduplication: {len(duplicated)} results")
    print(f"  Document IDs: {[r.document.id for r in duplicated]}")
    
    deduplicated = filter_results(duplicated, dedup_threshold=0.9)
    
    print(f"\nAfter deduplication: {len(deduplicated)} results")
    print(f"  Document IDs: {[r.document.id for r in deduplicated]}")
    
    
    # 6. Diversity with MMR
    print("\n\n6. MAXIMAL MARGINAL RELEVANCE (MMR)")
    print("-"*80)
    print("Balance relevance with diversity.\n")
    
    query = "python"
    results = keyword_search(query, documents, top_k=10)
    
    print(f"Original results (top 5):")
    for r in results[:5]:
        print(f"  {r.document.id}: {r.document.metadata['category']}")
    
    # Apply different diversity factors
    diversity_factors = [0.0, 0.3, 0.7, 1.0]
    
    print("\nDiversity factor comparison:")
    for factor in diversity_factors:
        diverse = diversify_results(results, max_results=5, diversity_factor=factor)
        categories = [r.document.metadata['category'] for r in diverse]
        unique_cats = len(set(categories))
        
        print(f"  Factor {factor:.1f}: {unique_cats} unique categories")
        print(f"    Categories: {categories}")
    
    
    # 7. Category Diversity
    print("\n\n7. CATEGORY DIVERSITY ANALYSIS")
    print("-"*80)
    
    results = keyword_search("python", documents, top_k=8)
    
    print("Without diversity:")
    print(f"  Results: {[r.document.id for r in results[:5]]}")
    categories_before = [r.document.metadata['category'] for r in results[:5]]
    print(f"  Categories: {categories_before}")
    print(f"  Unique: {len(set(categories_before))}")
    
    print("\nWith diversity (factor=0.5):")
    diverse = diversify_results(results, max_results=5, diversity_factor=0.5)
    print(f"  Results: {[r.document.id for r in diverse]}")
    categories_after = [r.document.metadata['category'] for r in diverse]
    print(f"  Categories: {categories_after}")
    print(f"  Unique: {len(set(categories_after))}")
    
    
    # 8. FilterConfig Usage
    print("\n\n8. FILTERCONFIG FOR REUSABILITY")
    print("-"*80)
    
    # Define reusable filter configurations
    strict_config = FilterConfig(
        min_score=1.0,
        max_results=3,
        metadata_filter={},  # Will filter manually for quality
        dedup_threshold=0.9
    )
    
    moderate_config = FilterConfig(
        min_score=0.5,
        max_results=5,
        metadata_filter={},
        dedup_threshold=0.9
    )
    
    lenient_config = FilterConfig(
        min_score=0.0,
        max_results=10,
        metadata_filter={},
        dedup_threshold=0.9
    )
    
    results = keyword_search("python framework", documents, top_k=10)
    
    configs = [
        ("Strict", strict_config, 0.9),
        ("Moderate", moderate_config, 0.85),
        ("Lenient", lenient_config, 0.0)
    ]
    
    print(f"Original: {len(results)} results\n")
    
    for name, config, quality_threshold in configs:
        filtered = filter_results(results, config=config)
        # Apply quality filter manually
        if quality_threshold > 0:
            filtered = [r for r in filtered if r.document.metadata.get('quality', 0) >= quality_threshold]
        
        print(f"{name:10}: {len(filtered)} results (min_score={config.min_score:.1f}, quality>={quality_threshold})")
        for r in filtered[:3]:
            print(f"            {r.document.id}: score={r.score:.3f}, "
                  f"quality={r.document.metadata['quality']}")
        print()
    
    
    # 9. Production Filtering Pipeline
    print("\n9. PRODUCTION FILTERING PIPELINE")
    print("-"*80)
    
    def production_filter(results, context="general"):
        """Apply context-aware filtering."""
        
        if context == "strict":
            # High precision, low recall
            filtered = filter_results(
                results,
                min_score=1.5,
                dedup_threshold=0.9
            )
            filtered = [r for r in filtered if r.document.metadata.get('quality', 0) >= 0.9]
        elif context == "balanced":
            # Balanced precision and recall
            filtered = filter_results(
                results,
                min_score=0.5,
                dedup_threshold=0.9
            )
            filtered = [r for r in filtered if r.document.metadata.get('quality', 0) >= 0.85]
            filtered = diversify_results(filtered, max_results=8, diversity_factor=0.4)
        else:  # lenient
            # High recall, lower precision
            filtered = filter_results(
                results,
                min_score=0.0,
                dedup_threshold=0.9
            )
            filtered = diversify_results(filtered, max_results=10, diversity_factor=0.6)
        
        return filtered
    
    query = "python libraries"
    results = keyword_search(query, documents, top_k=10)
    
    print(f"Query: '{query}'")
    print(f"Initial: {len(results)} results\n")
    
    for context in ["strict", "balanced", "lenient"]:
        filtered = production_filter(results, context=context)
        categories = [r.document.metadata['category'] for r in filtered]
        unique_cats = len(set(categories))
        
        print(f"{context.upper():10}: {len(filtered)} results, {unique_cats} unique categories")
        print(f"            IDs: {[r.document.id for r in filtered[:5]]}")
    
    
    # 10. Best Practices
    print("\n\n10. FILTERING BEST PRACTICES")
    print("-"*80)
    
    practices = [
        "Use score thresholds to ensure relevance",
        "Apply metadata filters for domain-specific constraints",
        "Deduplicate to remove redundant results",
        "Use MMR for diverse results when appropriate",
        "Balance diversity factor based on use case (0.3-0.5 typical)",
        "Filter before diversification for better performance",
        "Consider quality metrics in metadata",
        "Use FilterConfig for reusable configurations",
        "Monitor filtered result counts to avoid over-filtering",
        "A/B test different filtering strategies"
    ]
    
    for i, practice in enumerate(practices, 1):
        print(f"  {i:2}. {practice}")
    
    
    print("\n" + "="*80)
    print("Filtering and diversification demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    main()
