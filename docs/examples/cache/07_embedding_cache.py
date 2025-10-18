"""Embedding Cache Example

This example demonstrates caching embeddings from embedding APIs.

Main concepts:
- Generating cache keys for text embeddings
- Avoiding redundant embedding API calls
- Handling different embedding models
- Batch embedding optimization
- Vector similarity with cached embeddings
"""

from kerb.cache import (
    create_memory_cache,
    generate_embedding_key,
)


def mock_embedding_api_call(text, model="text-embedding-ada-002"):
    """Simulate an embedding API call.
    
    In production, this would call OpenAI, Cohere, or other embedding APIs.
    """
    # Simulate cost
    cost = 0.0001 * len(text.split())
    
    # Generate mock embedding (in reality, this would be a 1536-dim vector)
    # Using simple hash-based mock for demonstration
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    
    # Create fake embedding vector
    embedding = [(hash_val >> i) % 100 / 100.0 for i in range(8)]
    
    return {
        "embedding": embedding,
        "model": model,
        "dimensions": len(embedding),
        "tokens": len(text.split()) * 4,
        "cost": cost
    }


def main():
    """Run embedding cache example."""
    
    print("="*80)
    print("EMBEDDING CACHE EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # 1. Basic embedding caching
    # ========================================================================
    print("\n" + "-"*80)
    print("1. BASIC EMBEDDING CACHING")
    print("-"*80)
    
    cache = create_memory_cache(max_size=1000)
    
    text1 = "Machine learning is a subset of artificial intelligence"
    
    # First call - cache miss
    print(f"\nText: '{text1}'")
    print("First call (cache miss)...")
    
    key1 = generate_embedding_key(text1, model="text-embedding-ada-002")
    cached = cache.get(key1)
    
    if cached is None:
        print("  ✗ Cache miss - calling embedding API")
        result1 = mock_embedding_api_call(text1)
        cache.set(key1, result1)
        print(f"  Embedding: {result1['embedding'][:3]}... ({result1['dimensions']} dims)")
        print(f"  Cost: ${result1['cost']:.6f}")
    
    # Second call - cache hit
    print("\nSecond call (should hit cache)...")
    cached = cache.get(key1)
    
    if cached:
        print("  ✓ Cache hit - no API call needed!")
        print(f"  Embedding: {cached['embedding'][:3]}... ({cached['dimensions']} dims)")
        print(f"  Saved: ${cached['cost']:.6f}")
    
    # ========================================================================
    # 2. Model-specific caching
    # ========================================================================
    print("\n" + "-"*80)
    print("2. MODEL-SPECIFIC CACHING")
    print("-"*80)
    
    same_text = "Python is a programming language"
    
    # Different models generate different embeddings
    key_ada = generate_embedding_key(same_text, model="text-embedding-ada-002")
    key_3_large = generate_embedding_key(same_text, model="text-embedding-3-large")
    
    print(f"\nText: '{same_text}'")
    print("\nCache keys for different models:")
    print(f"  text-embedding-ada-002:   {key_ada[:16]}...")
    print(f"  text-embedding-3-large:   {key_3_large[:16]}...")
    print("\nEach model has its own cache entry!")
    
    # ========================================================================
    # 3. Batch embedding with caching
    # ========================================================================
    print("\n" + "-"*80)
    print("3. BATCH EMBEDDING WITH CACHING")
    print("-"*80)
    
    embed_cache = create_memory_cache(max_size=1000)
    
    documents = [
        "Machine learning is amazing",
        "Python is great for ML",
        "Machine learning is amazing",  # Duplicate
        "Deep learning uses neural networks",
        "Python is great for ML",        # Duplicate
        "AI is transforming industries",
        "Machine learning is amazing",  # Duplicate again
    ]
    
    print(f"\nEmbedding {len(documents)} documents...")
    
    api_calls = 0
    cache_hits = 0
    total_cost = 0.0
    embeddings = []
    
    for i, doc in enumerate(documents, 1):
        key = generate_embedding_key(doc, model="text-embedding-ada-002")
        cached = embed_cache.get(key)
        
        if cached is None:
            # API call needed
            result = mock_embedding_api_call(doc)
            embed_cache.set(key, result)
            embeddings.append(result["embedding"])
            api_calls += 1
            total_cost += result["cost"]
            print(f"  {i}. ✗ API call: '{doc[:40]}'")
        else:
            # Cache hit
            embeddings.append(cached["embedding"])
            cache_hits += 1
            print(f"  {i}. ✓ Cached:   '{doc[:40]}'")
    
    print(f"\nBatch Summary:")
    print(f"  Documents:     {len(documents)}")
    print(f"  API calls:     {api_calls}")
    print(f"  Cache hits:    {cache_hits}")
    print(f"  Total cost:    ${total_cost:.6f}")
    print(f"  Efficiency:    {cache_hits}/{len(documents)} cached")
    
    # ========================================================================
    # 4. Practical: Document search with cached embeddings
    # ========================================================================
    print("\n" + "-"*80)
    print("4. PRACTICAL: DOCUMENT SEARCH")
    print("-"*80)
    
    class EmbeddingSearchEngine:
        """Simple search engine with cached embeddings."""
        
        def __init__(self):
            self.cache = create_memory_cache(max_size=10000)
            self.documents = []
            self.embeddings = []
        
        def add_document(self, doc):
            """Add document with cached embedding."""
            key = generate_embedding_key(doc, model="text-embedding-ada-002")
            cached = self.cache.get(key)
            
            if cached is None:
                # Compute embedding
                result = mock_embedding_api_call(doc)
                self.cache.set(key, result)
                embedding = result["embedding"]
                print(f"    ✗ Computed embedding for: '{doc[:40]}'")
            else:
                # Use cached
                embedding = cached["embedding"]
                print(f"    ✓ Used cached for: '{doc[:40]}'")
            
            self.documents.append(doc)
            self.embeddings.append(embedding)
        
        def search(self, query, top_k=3):
            """Search using cached query embedding."""
            key = generate_embedding_key(query, model="text-embedding-ada-002")
            cached = self.cache.get(key)
            
            if cached is None:
                result = mock_embedding_api_call(query)
                self.cache.set(key, result)
                query_embedding = result["embedding"]
            else:
                query_embedding = cached["embedding"]
            
            # Simple cosine similarity (simplified for demo)
            def similarity(emb1, emb2):
                return sum(a * b for a, b in zip(emb1, emb2))
            
            # Rank documents
            scores = [(doc, similarity(query_embedding, emb)) 
                     for doc, emb in zip(self.documents, self.embeddings)]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            return scores[:top_k]
    
    # Use search engine
    search_engine = EmbeddingSearchEngine()
    
    print("\nBuilding document index:")
    docs = [
        "Python programming language",
        "Machine learning algorithms",
        "Python for data science",
        "Deep learning with PyTorch",
        "Python web development",
    ]
    
    for doc in docs:
        search_engine.add_document(doc)
    
    # First search
    print("\nSearch 1: 'Python programming'")
    results = search_engine.search("Python programming")
    for doc, score in results:
        print(f"  - {doc} (score: {score:.3f})")
    
    # Second search (query embedding cached)
    print("\nSearch 2: 'Python programming' (same query)")
    results = search_engine.search("Python programming")
    for doc, score in results:
        print(f"  - {doc} (score: {score:.3f})")
    print("  (Query embedding retrieved from cache)")
    
    # ========================================================================
    # 5. Handling embedding dimensions
    # ========================================================================
    print("\n" + "-"*80)
    print("5. HANDLING EMBEDDING DIMENSIONS")
    print("-"*80)
    
    dim_cache = create_memory_cache()
    
    text = "Embeddings are vector representations"
    
    # Different models have different dimensions
    models_dims = [
        ("text-embedding-ada-002", 1536),
        ("text-embedding-3-small", 1536),
        ("text-embedding-3-large", 3072),
    ]
    
    print(f"\nText: '{text}'")
    print("\nEmbedding with different models:")
    
    for model, dims in models_dims:
        key = generate_embedding_key(text, model=model)
        
        # Mock call with model-specific dimensions
        result = mock_embedding_api_call(text, model=model)
        result["dimensions"] = dims  # Override for demo
        
        dim_cache.set(key, result)
        print(f"  {model}: {dims} dimensions")
    
    # ========================================================================
    # 6. Cost optimization
    # ========================================================================
    print("\n" + "-"*80)
    print("6. COST OPTIMIZATION")
    print("-"*80)
    
    cost_cache = create_memory_cache()
    
    # Simulate FAQ system with repeated queries
    faqs = [
        "How do I reset my password?",
        "What are your business hours?",
        "How do I reset my password?",
        "Do you offer refunds?",
        "What are your business hours?",
        "How do I reset my password?",
        "How do I contact support?",
        "What are your business hours?",
    ]
    
    print(f"\nProcessing {len(faqs)} FAQ queries...")
    
    total_api_cost = 0.0
    total_saved = 0.0
    
    for i, faq in enumerate(faqs, 1):
        key = generate_embedding_key(faq, model="text-embedding-ada-002")
        cached = cost_cache.get(key)
        
        if cached is None:
            result = mock_embedding_api_call(faq)
            cost_cache.set(key, result)
            total_api_cost += result["cost"]
            print(f"  {i}. ✗ ${result['cost']:.6f} - {faq}")
        else:
            total_saved += cached["cost"]
            print(f"  {i}. ✓ Saved ${cached['cost']:.6f} - {faq}")
    
    print(f"\nCost Optimization:")
    print(f"  Total API cost:  ${total_api_cost:.6f}")
    print(f"  Total saved:     ${total_saved:.6f}")
    print(f"  Savings rate:    {(total_saved/(total_api_cost + total_saved)*100):.1f}%")
    print(f"  Unique queries:  {cost_cache.size()}")
    
    # ========================================================================
    # 7. Cache key normalization
    # ========================================================================
    print("\n" + "-"*80)
    print("7. CACHE KEY NORMALIZATION")
    print("-"*80)
    
    # Whitespace and case variations
    variations = [
        "  Machine Learning  ",  # Extra whitespace
        "machine learning",      # Different case
        "Machine Learning",      # Original
    ]
    
    print("\nText variations:")
    for text in variations:
        key = generate_embedding_key(text.strip().lower(), model="text-embedding-ada-002")
        print(f"  '{text}' → {key[:16]}...")
    
    print("\nTip: Normalize text (strip, lowercase) before generating keys")
    print("     This improves cache hit rate for similar queries")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("  • Cache embeddings to avoid redundant API calls")
    print("  • Use generate_embedding_key() for consistent keys")
    print("  • Different models need different cache entries")
    print("  • Batch operations benefit greatly from caching")
    print("  • Normalize text for better cache hit rates")
    print("  • Typical savings: 60-95% for repeated queries")
    print("  • Perfect for: RAG systems, semantic search, FAQ matching")


if __name__ == "__main__":
    main()
