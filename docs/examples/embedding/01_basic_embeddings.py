"""
Basic Embedding Example
=======================

=======================

This example demonstrates how to generate embeddings for text using different models.

Main concepts:
- Using the embed() function with default (local) model
- Switching between different embedding models
- Understanding embedding dimensions
- Basic vector properties
"""

from kerb.embedding import (
    embed,
    EmbeddingModel,
    embedding_dimension,
    vector_magnitude,
    normalize_vector
)


def main():
    """Run basic embedding example."""
    
    print("="*80)
    print("BASIC EMBEDDING EXAMPLE")
    print("="*80)
    
    # 1. Default embedding (local/hash-based - no dependencies required)
    print("\n1. DEFAULT LOCAL EMBEDDING")
    print("-"*80)
    
    text = "Machine learning transforms data into insights"
    embedding = embed(text)
    
    print(f"Text: '{text}'")
    print(f"Embedding dimension: {embedding_dimension(embedding)}")
    print(f"Vector magnitude: {vector_magnitude(embedding):.6f}")
    print(f"First 5 values: {[round(v, 4) for v in embedding[:5]]}")
    
    # 2. Different models with EmbeddingModel enum
    print("\n2. EMBEDDING WITH DIFFERENT MODELS")
    print("-"*80)
    
    text2 = "Natural language processing enables AI to understand text"
    
    # Local model (default - 384 dimensions)
    local_emb = embed(text2, model=EmbeddingModel.LOCAL, dimensions=384)
    print(f"\nLocal Model (hash-based):")
    print(f"  Dimensions: {len(local_emb)}")
    print(f"  First 3 values: {[round(v, 4) for v in local_emb[:3]]}")
    
    # Note: For Sentence Transformers models, you need: pip install sentence-transformers
    # Example (commented to avoid dependency):
    # st_emb = embed(text2, model=EmbeddingModel.ALL_MINILM_L6_V2)
    # print(f"\nSentence Transformers (all-MiniLM-L6-v2):")
    # print(f"  Dimensions: {len(st_emb)}")
    
    # Note: For OpenAI models, you need: pip install openai
    # And set OPENAI_API_KEY environment variable
    # Example (commented to avoid API costs):
    # openai_emb = embed(text2, model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL)
    # print(f"\nOpenAI (text-embedding-3-small):")
    # print(f"  Dimensions: {len(openai_emb)}")
    
    # 3. Vector properties
    print("\n3. VECTOR PROPERTIES")
    print("-"*80)
    
    text3 = "Embeddings convert text to numerical vectors"
    vec = embed(text3)
    
    print(f"Original vector magnitude: {vector_magnitude(vec):.6f}")
    
    # Normalize to unit length
    normalized = normalize_vector(vec)
    print(f"Normalized vector magnitude: {vector_magnitude(normalized):.6f}")
    print(f"Sum of squared values: {sum(v*v for v in normalized):.6f}")
    
    # 4. Custom dimensions for local model
    print("\n4. CUSTOM DIMENSIONS")
    print("-"*80)
    
    text4 = "Vectors can have different dimensionalities"
    
    vec_128 = embed(text4, model=EmbeddingModel.LOCAL, dimensions=128)
    vec_256 = embed(text4, model=EmbeddingModel.LOCAL, dimensions=256)
    vec_512 = embed(text4, model=EmbeddingModel.LOCAL, dimensions=512)
    
    print(f"128-dim vector: {len(vec_128)} dimensions")
    print(f"256-dim vector: {len(vec_256)} dimensions")
    print(f"512-dim vector: {len(vec_512)} dimensions")
    
    # 5. Reproducibility
    print("\n5. REPRODUCIBILITY")
    print("-"*80)
    
    text5 = "Same input produces same output"
    
    emb1 = embed(text5)
    emb2 = embed(text5)
    
    print(f"First embedding: {[round(v, 4) for v in emb1[:3]]}")
    print(f"Second embedding: {[round(v, 4) for v in emb2[:3]]}")
    print(f"Are they identical? {emb1 == emb2}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKEY TAKEAWAYS:")
    print("- Use embed() for simple text-to-vector conversion")
    print("- EmbeddingModel enum provides type-safe model selection")
    print("- Local model requires no dependencies, good for testing")
    print("- Different models produce different dimensions")
    print("- Same input always produces same embedding (deterministic)")


if __name__ == "__main__":
    main()
