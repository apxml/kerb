"""
Custom Embedder and Provider Classes Example
============================================

This example demonstrates using provider classes for more control over embeddings.

Main concepts:
- Using LocalEmbedder for hash-based embeddings
- Using SentenceTransformerEmbedder for local ML models
- Using OpenAIEmbedder for API-based embeddings
- Creating custom embedding pipelines
- Managing embedder state and configuration
"""

from kerb.embedding.providers import (
    LocalEmbedder,
    local_embed,
    # SentenceTransformerEmbedder,  # Requires: pip install sentence-transformers
    # sentence_transformer_embed,
    # OpenAIEmbedder,  # Requires: pip install openai
    # openai_embed,
)
from kerb.embedding import (
    cosine_similarity,
    embed_batch
)


def main():
    """Run custom embedder example."""
    
    print("="*80)
    print("CUSTOM EMBEDDER AND PROVIDER CLASSES EXAMPLE")
    print("="*80)
    
    # 1. LocalEmbedder class
    print("\n1. LOCAL EMBEDDER CLASS")
    print("-"*80)
    
    # Create embedder instance with custom dimensions
    embedder = LocalEmbedder(dimensions=512)
    
    print(f"Created LocalEmbedder with {embedder.dimensions} dimensions")
    
    text1 = "Using embedder classes for more control"
    text2 = "Classes provide stateful embedding generation"
    
    # Generate embeddings
    emb1 = embedder.embed(text1)
    emb2 = embedder.embed(text2)
    
    print(f"\nText 1: '{text1}'")
    print(f"Embedding dimension: {len(emb1)}")
    print(f"First 3 values: {[round(v, 4) for v in emb1[:3]]}")
    
    similarity = cosine_similarity(emb1, emb2)
    print(f"\nSimilarity between texts: {similarity:.4f}")
    
    # Batch embedding
    texts = [
        "First document",
        "Second document",
        "Third document"
    ]
    
    batch_embeddings = embedder.embed_batch(texts)
    print(f"\nBatch embedded {len(batch_embeddings)} texts")
    
    # 2. Direct function usage
    print("\n2. DIRECT FUNCTION USAGE")
    print("-"*80)
    
    # Use local_embed function directly
    text = "Direct function call for embeddings"
    embedding = local_embed(text, dimensions=256)
    
    print(f"Text: '{text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Generated using local_embed() function")
    
    # 3. Multiple embedder instances
    print("\n3. MULTIPLE EMBEDDER INSTANCES")
    print("-"*80)
    
    # Create different embedders for different purposes
    quick_embedder = LocalEmbedder(dimensions=128)  # Fast, lower dimension
    precise_embedder = LocalEmbedder(dimensions=1024)  # Slower, higher dimension
    
    text = "Testing with different embedder configurations"
    
    quick_emb = quick_embedder.embed(text)
    precise_emb = precise_embedder.embed(text)
    
    print(f"Text: '{text}'")
    print(f"\nQuick embedder: {len(quick_emb)} dimensions")
    print(f"Precise embedder: {len(precise_emb)} dimensions")
    print("\nUse quick for real-time, precise for accuracy")
    
    # 4. Custom embedding pipeline
    print("\n4. CUSTOM EMBEDDING PIPELINE")
    print("-"*80)
    
    class CustomEmbeddingPipeline:
        """Custom pipeline with preprocessing and multiple embedders."""

# %%
# Setup and Imports
# -----------------
        

# %%
#   Init  
# --------

        def __init__(self):
            self.embedder = LocalEmbedder(dimensions=384)
        

# %%
# Preprocess
# ----------

        def preprocess(self, text):
            """Preprocess text before embedding."""
            # Convert to lowercase
            text = text.lower()
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        
        def embed_with_preprocessing(self, text):
            """Embed with preprocessing."""
            processed = self.preprocess(text)
            return self.embedder.embed(processed)
        

# %%
# Embed Multi Representation
# --------------------------

        def embed_multi_representation(self, text):
            """Create multiple representations and combine."""
            # Original
            emb1 = self.embedder.embed(text)
            
            # Lowercase version
            emb2 = self.embedder.embed(text.lower())
            
            # Average the two
            combined = [(v1 + v2) / 2 for v1, v2 in zip(emb1, emb2)]
            return combined
    
    pipeline = CustomEmbeddingPipeline()
    
    text = "  This Text   Has   Irregular   Spacing  "
    
    # Without preprocessing
    normal_emb = local_embed(text)
    
    # With preprocessing
    processed_emb = pipeline.embed_with_preprocessing(text)
    
    print(f"Original text: '{text}'")
    print(f"Preprocessed: '{pipeline.preprocess(text)}'")
    print(f"\nBoth generate {len(normal_emb)}-dimensional embeddings")
    print("Preprocessing ensures consistent results")
    
    # 5. Embedder configuration
    print("\n5. EMBEDDER CONFIGURATION")
    print("-"*80)
    
    # Different configurations for different use cases
    configs = [
        ("Fast", LocalEmbedder(dimensions=64)),
        ("Balanced", LocalEmbedder(dimensions=256)),
        ("High-Quality", LocalEmbedder(dimensions=768))
    ]
    
    test_text = "Configuration affects quality and speed"
    
    print(f"Text: '{test_text}'")
    print("\nDifferent configurations:")
    
    for name, embedder in configs:
        emb = embedder.embed(test_text)
        print(f"  {name:15s}: {len(emb):4d} dimensions")
    
    # 6. Caching with embedder class
    print("\n6. CACHING WITH EMBEDDER CLASS")
    print("-"*80)
    
    class CachedEmbedder:
        """Embedder with built-in caching."""
        

# %%
#   Init  
# --------

        def __init__(self, dimensions=384):
            self.embedder = LocalEmbedder(dimensions=dimensions)
            self.cache = {}
        

# %%
# Embed
# -----

        def embed(self, text):
            """Embed with caching."""
            if text in self.cache:
                print(f"  Cache hit: '{text[:40]}'")
                return self.cache[text]
            
            print(f"  Computing: '{text[:40]}'")
            embedding = self.embedder.embed(text)
            self.cache[text] = embedding
            return embedding
        
        def clear_cache(self):
            """Clear the cache."""
            self.cache.clear()
        

# %%
# Cache Size
# ----------

        def cache_size(self):
            """Get cache size."""
            return len(self.cache)
    
    cached_embedder = CachedEmbedder()
    
    texts = [
        "First call to this text",
        "Second different text",
        "First call to this text",  # Duplicate - should hit cache
        "Third unique text",
        "Second different text"  # Duplicate - should hit cache
    ]
    
    print("Processing texts with caching:")
    for text in texts:
        cached_embedder.embed(text)
    
    print(f"\nCache contains {cached_embedder.cache_size()} unique embeddings")
    
    # 7. Specialized embedder wrapper
    print("\n7. SPECIALIZED EMBEDDER WRAPPER")
    print("-"*80)
    
    class DomainSpecificEmbedder:
        """Embedder optimized for specific domain."""
        

# %%
#   Init  
# --------

        def __init__(self, domain="general"):
            self.domain = domain
            self.embedder = LocalEmbedder(dimensions=384)
            
            # Domain-specific prefixes
            self.prefixes = {
                "code": "Code: ",
                "medical": "Medical: ",
                "legal": "Legal: ",
                "general": ""
            }
        

# %%
# Embed
# -----

        def embed(self, text):
            """Embed with domain context."""
            prefix = self.prefixes.get(self.domain, "")
            contextualized = f"{prefix}{text}"
            return self.embedder.embed(contextualized)
    
    text = "patient shows symptoms of condition"
    
    general_embedder = DomainSpecificEmbedder(domain="general")
    medical_embedder = DomainSpecificEmbedder(domain="medical")
    
    general_emb = general_embedder.embed(text)
    medical_emb = medical_embedder.embed(text)
    
    similarity = cosine_similarity(general_emb, medical_emb)
    
    print(f"Text: '{text}'")
    print(f"\nGeneral domain embedding dimension: {len(general_emb)}")
    print(f"Medical domain embedding dimension: {len(medical_emb)}")
    print(f"Similarity: {similarity:.4f}")
    print("\nDomain prefixes help differentiate contexts")
    
    # 8. Comparison with standard embed function
    print("\n8. PROVIDER CLASSES VS STANDARD FUNCTIONS")
    print("-"*80)
    
    text = "Comparing different approaches"
    
    # Using standard function
    emb_func = embed_batch([text])[0]
    
    # Using provider class
    embedder_obj = LocalEmbedder()
    emb_class = embedder_obj.embed(text)
    
    # Using direct provider function
    emb_direct = local_embed(text)
    
    print(f"Text: '{text}'")
    print(f"\nAll three approaches produce {len(emb_func)}-dimensional embeddings")
    print(f"Standard function: Convenient, good for most uses")
    print(f"Provider class: More control, stateful, reusable")
    print(f"Direct function: Explicit provider, clear backend")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKEY TAKEAWAYS:")
    print("- Provider classes offer more control than standard functions")
    print("- LocalEmbedder: Fast, no dependencies, good for testing")
    print("- SentenceTransformerEmbedder: High quality local ML models")
    print("- OpenAIEmbedder: Highest quality, requires API key")
    print("- Classes maintain state and configuration")
    print("- Easy to build custom pipelines with preprocessing")
    print("- Provider classes enable advanced patterns like caching")
    print("\nNOTE: To use SentenceTransformerEmbedder:")
    print("  pip install sentence-transformers")
    print("\nNOTE: To use OpenAIEmbedder:")
    print("  pip install openai")
    print("  Set OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    main()
