"""
Basic Text Chunking Example
===========================

This example demonstrates fundamental chunking approaches for LLM applications.

Main concepts:
- Simple character-based chunking
- Overlap between chunks for context preservation
- Quick utility functions for common tasks

Use cases:
- Splitting large documents for LLM processing
- Creating chunks with context overlap
- Basic text preprocessing
"""

from kerb.chunk import simple_chunker, overlap_chunker, chunk_text


def demonstrate_simple_chunking():
    """Show basic character-based chunking."""
    print("="*80)
    print("SIMPLE CHUNKING")
    print("="*80)
    
    # Sample text
    text = """

# %%
# Setup and Imports
# -----------------
    Artificial intelligence (AI) is transforming the technology landscape.
    Machine learning enables computers to learn from data without explicit programming.
    Natural language processing allows machines to understand and generate human language.
    Deep learning uses neural networks to solve complex problems.
    Large language models can perform various text tasks with remarkable accuracy.
    """.strip()
    
    print(f"\nOriginal text ({len(text)} chars):\n{text}\n")
    
    # Split into chunks of 100 characters
    chunks = simple_chunker(text, chunk_size=100)
    
    print(f"\nCreated {len(chunks)} chunks (size=100):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"  '{chunk}'")


def demonstrate_overlap_chunking():
    """Show chunking with overlap for context preservation."""
    print("\n" + "="*80)
    print("OVERLAP CHUNKING")
    print("="*80)
    
    text = """
    The transformer architecture revolutionized NLP in 2017.
    Self-attention mechanisms allow models to weigh input importance.
    BERT introduced bidirectional training for better understanding.
    GPT models demonstrate impressive text generation capabilities.
    Fine-tuning adapts pre-trained models to specific tasks.
    """.strip()
    
    print(f"\nText to chunk ({len(text)} chars):\n{text}\n")
    
    # Create chunks with 10% overlap
    chunks = overlap_chunker(text, chunk_size=80, overlap_ratio=0.15)
    
    print(f"\nCreated {len(chunks)} chunks (size=80, overlap=15%):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"  '{chunk}'")
        
        # Highlight overlap
        if i < len(chunks):
            overlap_chars = int(80 * 0.15)
            if len(chunk) >= overlap_chars:
                print(f"  Overlap with next: '...{chunk[-overlap_chars:]}'")



# %%
# Demonstrate Utility Function
# ----------------------------

def demonstrate_utility_function():
    """Show the convenient chunk_text utility."""
    print("\n" + "="*80)
    print("CHUNK_TEXT UTILITY")
    print("="*80)
    
    # Long article excerpt
    article = """
    Retrieval-Augmented Generation (RAG) combines retrieval and generation.
    It retrieves relevant documents from a knowledge base first.
    Then it uses those documents as context for the LLM.
    This approach reduces hallucinations significantly.
    RAG systems are widely used in production applications.
    They enable LLMs to access up-to-date information.
    Vector databases store document embeddings for efficient retrieval.
    Semantic search finds the most relevant chunks.
    """.strip()
    
    print(f"\nArticle ({len(article)} chars):\n{article}\n")
    
    # Quick chunking with utility function
    chunks = chunk_text(article, chunk_size=120, overlap=20)
    
    print(f"\nUtility function created {len(chunks)} chunks:")
    print(f"  - Chunk size: 120 characters")
    print(f"  - Overlap: 20 characters\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: '{chunk.strip()}'")


def compare_chunking_strategies():
    """Compare different basic chunking approaches."""
    print("\n" + "="*80)
    print("COMPARING STRATEGIES")
    print("="*80)
    
    text = "LLM applications require efficient text chunking for optimal performance."
    
    print(f"\nText: '{text}' ({len(text)} chars)\n")
    
    # Strategy 1: No overlap
    chunks_no_overlap = simple_chunker(text, chunk_size=30)
    print(f"No overlap (size=30): {len(chunks_no_overlap)} chunks")
    for i, c in enumerate(chunks_no_overlap, 1):
        print(f"  {i}: '{c}'")
    
    # Strategy 2: With overlap
    chunks_with_overlap = overlap_chunker(text, chunk_size=30, overlap_ratio=0.2)
    print(f"\nWith 20% overlap (size=30): {len(chunks_with_overlap)} chunks")
    for i, c in enumerate(chunks_with_overlap, 1):
        print(f"  {i}: '{c}'")
    
    # Strategy 3: Using chunk_text directly
    chunks_direct = chunk_text(text, chunk_size=30, overlap=10)
    print(f"\nDirect overlap (size=30, overlap=10): {len(chunks_direct)} chunks")
    for i, c in enumerate(chunks_direct, 1):
        print(f"  {i}: '{c}'")



# %%
# Main
# ----

def main():
    """Run basic chunking examples."""
    
    print("\n" + "="*80)
    print("BASIC TEXT CHUNKING EXAMPLES")
    print("="*80)
    print("\nThese examples show fundamental chunking approaches for LLM applications.\n")
    
    demonstrate_simple_chunking()
    demonstrate_overlap_chunking()
    demonstrate_utility_function()
    compare_chunking_strategies()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key Takeaways:
- simple_chunker: Basic fixed-size chunks, good for uniform processing
- overlap_chunker: Adds context overlap, better for maintaining coherence
- chunk_text: Quick utility for common chunking needs
- Overlap is crucial for RAG systems to preserve context across chunks
- Choose chunk size based on your LLM's context window and task needs
    """)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
