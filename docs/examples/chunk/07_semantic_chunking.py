"""Semantic Chunking Example

This example demonstrates sentence-based semantic chunking.

Main concepts:
- Sentence-level chunking for semantic coherence
- SemanticChunker class
- Grouping related sentences
- Maintaining meaning across chunks

Use cases:
- Content with strong sentence-level semantics
- Question answering systems
- Semantic search applications
- Summary generation
"""

from kerb.chunk import SemanticChunker


def demonstrate_basic_semantic_chunking():
    """Show basic semantic chunking by sentences."""
    print("="*80)
    print("BASIC SEMANTIC CHUNKING")
    print("="*80)
    
    text = """
Transformers revolutionized natural language processing in 2017. The attention mechanism allows models to focus on relevant parts of the input. Self-attention computes relationships between all tokens in a sequence. Multi-head attention captures different types of relationships simultaneously. Positional encodings provide information about token order. These components combine to create powerful sequence models. BERT introduced bidirectional training for better understanding. GPT models demonstrated impressive generation capabilities. Modern LLMs build on these foundational architectures.
    """.strip()
    
    print(f"\nOriginal text ({len(text)} chars):\n{text}\n")
    
    # Chunk by grouping 3 sentences together
    chunker = SemanticChunker(sentences_per_chunk=3)
    chunks = chunker.chunk(text)
    
    print(f"\nSemanticChunker created {len(chunks)} chunks (sentences_per_chunk=3):")
    for i, chunk in enumerate(chunks, 1):
        sentences = chunk.split('. ')
        print(f"\nChunk {i} ({len(sentences)} sentences, {len(chunk)} chars):")
        print(f"  {chunk}")


def demonstrate_sentence_grouping():
    """Show different sentence grouping strategies."""
    print("\n" + "="*80)
    print("SENTENCE GROUPING STRATEGIES")
    print("="*80)
    
    text = """
RAG systems combine retrieval and generation. They retrieve relevant documents first. Then they use those documents as context. This reduces hallucinations significantly. The retrieval step uses vector similarity. Embeddings capture semantic meaning. Similar texts have similar embeddings. This enables semantic search. The generation step produces the final response.
    """.strip()
    
    print(f"\nText: {text}\n")
    
    # Strategy 1: Small groups (2 sentences)
    print("Strategy 1: Small groups (2 sentences per chunk)")
    chunker_2 = SemanticChunker(sentences_per_chunk=2)
    chunks_2 = chunker_2.chunk(text)
    print(f"  Chunks: {len(chunks_2)}")
    for i, chunk in enumerate(chunks_2, 1):
        print(f"  {i}. {chunk}")
    
    # Strategy 2: Medium groups (4 sentences)
    print("\nStrategy 2: Medium groups (4 sentences per chunk)")
    chunker_4 = SemanticChunker(sentences_per_chunk=4)
    chunks_4 = chunker_4.chunk(text)
    print(f"  Chunks: {len(chunks_4)}")
    for i, chunk in enumerate(chunks_4, 1):
        print(f"  {i}. {chunk}")
    
    # Strategy 3: Large groups (6 sentences)
    print("\nStrategy 3: Large groups (6 sentences per chunk)")
    chunker_6 = SemanticChunker(sentences_per_chunk=6)
    chunks_6 = chunker_6.chunk(text)
    print(f"  Chunks: {len(chunks_6)}")
    for i, chunk in enumerate(chunks_6, 1):
        print(f"  {i}. {chunk[:100]}...")


def demonstrate_qa_chunking():
    """Show semantic chunking for question answering."""
    print("\n" + "="*80)
    print("SEMANTIC CHUNKING FOR Q&A")
    print("="*80)
    
    knowledge_base = """
Fine-tuning adapts pre-trained models to specific tasks. It requires labeled training data in the correct format. The process updates model weights through backpropagation. Learning rate is a critical hyperparameter to tune. Too high and training becomes unstable. Too low and convergence is slow. Batch size affects memory usage and training dynamics. Larger batches provide more stable gradients. Smaller batches can lead to better generalization. Number of epochs determines training duration. Monitor validation loss to prevent overfitting. Early stopping helps avoid degradation.
    """.strip()
    
    print(f"Knowledge base:\n{knowledge_base[:150]}...\n")
    
    # Chunk for precise Q&A retrieval
    chunker = SemanticChunker(sentences_per_chunk=3)
    chunks = chunker.chunk(knowledge_base)
    
    print(f"Q&A chunks (3 sentences per chunk, {len(chunks)} chunks):\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Content: {chunk}")
        print(f"  Use case: Retrieve for questions about {chunk.split('.')[0].lower()}")
        print()


def demonstrate_coherence_preservation():
    """Show how semantic chunking preserves coherence."""
    print("\n" + "="*80)
    print("COHERENCE PRESERVATION")
    print("="*80)
    
    text = """
Vector databases are essential for RAG systems. They store high-dimensional embeddings efficiently. Similarity search retrieves relevant documents. Popular options include Pinecone and Weaviate. Pinecone offers managed cloud infrastructure. It scales automatically with demand. Weaviate provides both cloud and self-hosted options. It supports GraphQL for flexible queries.
    """.strip()
    
    print(f"Text: {text}\n")
    
    # Semantic chunking keeps related sentences together
    print("Semantic chunking (2 sentences per chunk):")
    semantic_chunker = SemanticChunker(sentences_per_chunk=2)
    semantic_chunks = semantic_chunker.chunk(text)
    
    for i, chunk in enumerate(semantic_chunks, 1):
        print(f"\nChunk {i}: {chunk}")
        print(f"  -> Related sentences stay together for coherent meaning")
    
    # Compare with simple chunking (breaks sentences)
    from kerb.chunk import simple_chunker
    print("\n\nSimple chunking (100 chars per chunk):")
    simple_chunks = simple_chunker(text, chunk_size=100)
    
    for i, chunk in enumerate(simple_chunks, 1):
        print(f"\nChunk {i}: '{chunk}'")
        if not chunk.endswith('.'):
            print(f"  -> Sentence broken mid-way, loses coherence")


def demonstrate_content_types():
    """Show semantic chunking for different content types."""
    print("\n" + "="*80)
    print("DIFFERENT CONTENT TYPES")
    print("="*80)
    
    # Technical content
    technical_text = """
Attention mechanisms compute weighted sums of value vectors. Query vectors determine which values to attend to. Key vectors are compared with queries for relevance. The dot product measures similarity between queries and keys. Softmax normalizes attention weights to sum to one. This allows the model to focus on relevant information.
    """.strip()
    
    print("Technical Content:")
    print(f"{technical_text[:100]}...\n")
    
    chunker = SemanticChunker(sentences_per_chunk=2)
    tech_chunks = chunker.chunk(technical_text)
    
    print(f"Technical chunks ({len(tech_chunks)} chunks):")
    for i, chunk in enumerate(tech_chunks, 1):
        print(f"  {i}. {chunk}")
    
    # Narrative content
    narrative_text = """
The AI research team faced a difficult challenge. Their model was overfitting the training data. Validation accuracy plateaued after just two epochs. They decided to try regularization techniques. Dropout layers were added to the network. This helped prevent overfitting. The model's generalization improved significantly.
    """.strip()
    
    print(f"\n\nNarrative Content:")
    print(f"{narrative_text[:100]}...\n")
    
    narrative_chunks = chunker.chunk(narrative_text)
    
    print(f"Narrative chunks ({len(narrative_chunks)} chunks):")
    for i, chunk in enumerate(narrative_chunks, 1):
        print(f"  {i}. {chunk}")


def demonstrate_semantic_search_prep():
    """Show preparing content for semantic search."""
    print("\n" + "="*80)
    print("SEMANTIC SEARCH PREPARATION")
    print("="*80)
    
    documents = [
        """
Embedding models convert text to vectors. They capture semantic meaning in high-dimensional space. Similar texts produce similar vectors. This enables similarity-based search. OpenAI's ada-002 is widely used. Sentence Transformers provide open-source alternatives.
        """.strip(),
        
        """
Prompt engineering optimizes LLM performance. Techniques include few-shot learning. Chain-of-thought improves reasoning. Role-based prompts guide behavior. Clear instructions enhance output quality. Iterative refinement is essential.
        """.strip()
    ]
    
    print(f"Processing {len(documents)} documents for semantic search\n")
    
    chunker = SemanticChunker(sentences_per_chunk=2)
    
    all_chunks = []
    for doc_id, doc in enumerate(documents, 1):
        chunks = chunker.chunk(doc)
        
        print(f"Document {doc_id}:")
        print(f"  Original: {doc[:60]}...")
        print(f"  Chunks created: {len(chunks)}")
        
        for chunk_id, chunk in enumerate(chunks, 1):
            all_chunks.append({
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'text': chunk,
                'preview': chunk[:50] + '...' if len(chunk) > 50 else chunk
            })
            print(f"    Chunk {chunk_id}: {chunk}")
        
        print()
    
    print(f"Total: {len(all_chunks)} searchable semantic units")
    print("\nEach chunk:")
    print("  - Contains complete sentences")
    print("  - Maintains semantic coherence")
    print("  - Ready for embedding")
    print("  - Optimized for retrieval")


def demonstrate_optimal_settings():
    """Show recommended settings for different scenarios."""
    print("\n" + "="*80)
    print("OPTIMAL SEMANTIC CHUNKING SETTINGS")
    print("="*80)
    
    sample_text = """
The transformer architecture uses self-attention. It processes sequences in parallel. This enables efficient training. Positional encodings add sequence information. Layer normalization stabilizes training. Residual connections help gradient flow. These components work together effectively.
    """.strip()
    
    print(f"Sample text: {sample_text}\n")
    
    # Scenario 1: Precise retrieval
    print("Scenario 1: PRECISE RETRIEVAL (Q&A systems)")
    print("  Goal: Find exact facts")
    print("  Setting: sentences_per_chunk=2")
    chunker_precise = SemanticChunker(sentences_per_chunk=2)
    precise_chunks = chunker_precise.chunk(sample_text)
    print(f"  Result: {len(precise_chunks)} focused chunks")
    for i, chunk in enumerate(precise_chunks, 1):
        print(f"    {i}. {chunk}")
    
    # Scenario 2: Contextual understanding
    print("\nScenario 2: CONTEXTUAL UNDERSTANDING (Summarization)")
    print("  Goal: Preserve broader context")
    print("  Setting: sentences_per_chunk=4")
    chunker_context = SemanticChunker(sentences_per_chunk=4)
    context_chunks = chunker_context.chunk(sample_text)
    print(f"  Result: {len(context_chunks)} contextual chunks")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"    {i}. {chunk[:80]}...")
    
    # Scenario 3: Balanced approach
    print("\nScenario 3: BALANCED (General RAG)")
    print("  Goal: Balance precision and context")
    print("  Setting: sentences_per_chunk=3")
    chunker_balanced = SemanticChunker(sentences_per_chunk=3)
    balanced_chunks = chunker_balanced.chunk(sample_text)
    print(f"  Result: {len(balanced_chunks)} balanced chunks")
    for i, chunk in enumerate(balanced_chunks, 1):
        print(f"    {i}. {chunk}")


def main():
    """Run semantic chunking examples."""
    
    print("\n" + "="*80)
    print("SEMANTIC CHUNKING EXAMPLES")
    print("="*80)
    print("\nSentence-based chunking for semantic coherence.\n")
    
    demonstrate_basic_semantic_chunking()
    demonstrate_sentence_grouping()
    demonstrate_qa_chunking()
    demonstrate_coherence_preservation()
    demonstrate_content_types()
    demonstrate_semantic_search_prep()
    demonstrate_optimal_settings()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key Takeaways:
- SemanticChunker groups sentences for semantic coherence
- Preserves complete sentences (no mid-sentence breaks)
- Configure sentences_per_chunk based on your use case
- 2 sentences: Precise retrieval and Q&A
- 3 sentences: Balanced approach for general RAG
- 4+ sentences: More context for summarization
- Better than character-based chunking for meaning preservation
- Ideal for semantic search and question answering
- Works well with technical and narrative content
    """)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
