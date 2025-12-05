"""
Recursive Chunking Example
==========================

This example demonstrates semantic-aware text splitting using recursive chunkers.

Main concepts:
- Hierarchical separator-based splitting
- Maintaining semantic boundaries (paragraphs, sentences)
- Class-based vs functional interfaces
- Custom separator hierarchies

Use cases:
- Document chunking that respects structure
- Preserving paragraph/sentence integrity
- Customized splitting for specific document types
"""

from kerb.chunk import RecursiveChunker, recursive_chunker


def demonstrate_basic_recursive():
    """Show basic recursive chunking with default separators."""
    print("="*80)
    print("BASIC RECURSIVE CHUNKING")
    print("="*80)
    
    # Multi-paragraph text
    text = """

# %%
# Setup and Imports
# -----------------
Retrieval-Augmented Generation (RAG) is a powerful technique for LLM applications. It combines the benefits of retrieval systems with generative models.

The process works in several steps. First, documents are chunked and embedded. Then, relevant chunks are retrieved based on the query. Finally, the LLM generates a response using the retrieved context.

Vector databases play a crucial role in RAG systems. They enable efficient similarity search across millions of document chunks. Popular options include Pinecone, Weaviate, and Chroma.

Choosing the right chunk size is critical. Too small and you lose context. Too large and you may exceed token limits or include irrelevant information.
    """.strip()
    
    print(f"\nOriginal text ({len(text)} chars):\n{text}\n")
    
    # Use functional interface with default separators: ['\n\n', '\n', '. ', ' ', '']
    chunks = recursive_chunker(text, chunk_size=200)
    
    print(f"\nCreated {len(chunks)} chunks with recursive splitting (size=200):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"  {chunk}")


def demonstrate_class_based_recursive():
    """Show class-based recursive chunker for reusable configurations."""
    print("\n" + "="*80)
    print("CLASS-BASED RECURSIVE CHUNKER")
    print("="*80)
    
    text = """
Fine-tuning LLMs requires careful data preparation. Start with high-quality examples that represent your target task. Clean and format your dataset consistently.

Choose appropriate hyperparameters. Learning rate, batch size, and number of epochs significantly impact results. Monitor validation metrics to prevent overfitting.

Evaluation is essential. Test your fine-tuned model on held-out data. Compare performance against the base model to measure improvement.
    """.strip()
    
    print(f"\nText to chunk ({len(text)} chars):\n{text}\n")
    
    # Create reusable chunker instance
    chunker = RecursiveChunker(chunk_size=150)
    
    # Chunk the text
    chunks = chunker.chunk(text)
    
    print(f"\nRecursiveChunker created {len(chunks)} chunks (size=150):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"  {chunk}")
    
    # Show how separator hierarchy works
    print("\n\nDefault separator hierarchy:")
    print("  1. '\\n\\n' (paragraphs)")
    print("  2. '\\n' (lines)")
    print("  3. '. ' (sentences)")
    print("  4. ' ' (words)")
    print("  5. '' (characters)")



# %%
# Demonstrate Custom Separators
# -----------------------------

def demonstrate_custom_separators():
    """Show custom separator hierarchies for specific use cases."""
    print("\n" + "="*80)
    print("CUSTOM SEPARATOR HIERARCHIES")
    print("="*80)
    
    # JSON-like structured text
    structured_text = """
{
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 2000,
  "top_p": 0.9
}

{
  "model": "claude-3",
  "temperature": 0.5,
  "max_tokens": 1500,
  "top_p": 0.95
}

{
  "model": "llama-2",
  "temperature": 0.8,
  "max_tokens": 1800,
  "top_p": 0.85
}
    """.strip()
    
    print(f"\nStructured text ({len(structured_text)} chars):\n{structured_text}\n")
    
    # Custom separators for JSON-like content
    json_separators = ['\n\n', '\n', ',', ' ', '']
    chunker = RecursiveChunker(chunk_size=80, separators=json_separators)
    
    chunks = chunker.chunk(structured_text)
    
    print(f"\nCustom separators: {json_separators}")
    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")


def demonstrate_dialogue_chunking():
    """Show custom separators for dialogue/conversation data."""
    print("\n" + "="*80)
    print("DIALOGUE CHUNKING")
    print("="*80)
    
    dialogue = """
User: How do I implement RAG?
Assistant: RAG requires three main components: a vector database, an embedding model, and an LLM.

User: Which vector database should I use?
Assistant: Popular choices include Pinecone for managed solutions, or Chroma for local development. Choose based on your scale and budget.

User: What about chunking strategies?
Assistant: Use RecursiveChunker for general text, or SemanticChunker for sentence-based splitting. Chunk size typically ranges from 500-1000 characters.

User: How do I handle overlap?
Assistant: Use overlap_chunker or sentence_window_chunker. Overlap of 10-20% helps maintain context between chunks.
    """.strip()
    
    print(f"\nDialogue ({len(dialogue)} chars):\n{dialogue}\n")
    
    # Separator hierarchy that respects dialogue structure
    dialogue_separators = ['\n\n', '\nUser:', '\nAssistant:', '. ', ' ']
    chunker = RecursiveChunker(chunk_size=120, separators=dialogue_separators)
    
    chunks = chunker.chunk(dialogue)
    
    print(f"\nDialogue-aware separators: {dialogue_separators}")
    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:\n{chunk}\n")



# %%
# Demonstrate Vs Simple Chunking
# ------------------------------

def demonstrate_vs_simple_chunking():
    """Compare recursive vs simple chunking."""
    print("\n" + "="*80)
    print("RECURSIVE VS SIMPLE CHUNKING")
    print("="*80)
    
    text = """
Prompt engineering is crucial for LLM success.

Techniques include few-shot learning, chain-of-thought prompting, and role-based instructions.

Each approach has specific use cases and benefits.
    """.strip()
    
    print(f"\nText: {text}\n")
    print(f"Length: {len(text)} characters\n")
    
    # Simple chunking (breaks words/sentences)
    from kerb.chunk import simple_chunker
    simple_chunks = simple_chunker(text, chunk_size=60)
    
    print(f"Simple chunking ({len(simple_chunks)} chunks, size=60):")
    for i, chunk in enumerate(simple_chunks, 1):
        print(f"  {i}: '{chunk}'")
    
    # Recursive chunking (respects boundaries)
    recursive_chunks = recursive_chunker(text, chunk_size=60)
    
    print(f"\nRecursive chunking ({len(recursive_chunks)} chunks, size=60):")
    for i, chunk in enumerate(recursive_chunks, 1):
        print(f"  {i}: '{chunk}'")
    
    print("\nNotice how recursive chunking preserves sentence integrity!")


def main():
    """Run recursive chunking examples."""
    
    print("\n" + "="*80)
    print("RECURSIVE CHUNKING EXAMPLES")
    print("="*80)
    print("\nSemantic-aware text splitting for better LLM performance.\n")
    
    demonstrate_basic_recursive()
    demonstrate_class_based_recursive()
    demonstrate_custom_separators()
    demonstrate_dialogue_chunking()
    demonstrate_vs_simple_chunking()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key Takeaways:
- RecursiveChunker maintains semantic boundaries (paragraphs, sentences)
- Default separators: ['\\n\\n', '\\n', '. ', ' ', '']
- Customize separators for specific document types (JSON, dialogue, code)
- Class-based interface allows reusable chunker configurations
- Functional interface (recursive_chunker) is convenient for one-off use
- Much better than simple chunking for preserving meaning and context
    """)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
