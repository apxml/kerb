"""RAG-Specific Chunking Example

This example demonstrates chunking strategies optimized for RAG systems.

Main concepts:
- Sentence window chunking with overlap
- Context preservation across chunks
- Optimal chunk sizes for retrieval
- SentenceChunker class for sentence-based splitting

Use cases:
- Building RAG pipelines
- Document retrieval systems
- Question-answering applications
- Context-aware semantic search
"""

from kerb.chunk import sentence_window_chunker, SentenceChunker


def demonstrate_sentence_window():
    """Show sentence-based chunking with overlap for RAG."""
    print("="*80)
    print("SENTENCE WINDOW CHUNKING FOR RAG")
    print("="*80)
    
    # Knowledge base article
    article = """
Vector databases are essential for RAG systems. They store document embeddings and enable fast similarity search. Popular options include Pinecone, Weaviate, Qdrant, and Chroma. Each database has different trade-offs in terms of performance, scalability, and features. Pinecone offers a fully managed solution with excellent performance. Weaviate provides both cloud and self-hosted options. Qdrant emphasizes speed and efficiency. Chroma is designed for local development and prototyping. Choosing the right database depends on your specific requirements. Consider factors like query latency, storage capacity, and deployment preferences. Integration with your existing stack is also important. Most databases offer Python clients and REST APIs.
    """.strip()
    
    print(f"\nKnowledge base article ({len(article)} chars):\n{article}\n")
    
    # Create chunks with 3 sentences per window, 1 sentence overlap
    chunks = sentence_window_chunker(
        article,
        window_sentences=3,
        overlap_sentences=1
    )
    
    print(f"\nCreated {len(chunks)} chunks (window=3 sentences, overlap=1):")
    for i, chunk in enumerate(chunks, 1):
        sentences = chunk.split('. ')
        print(f"\nChunk {i} ({len(sentences)} sentences):")
        print(f"  {chunk}")


def demonstrate_sentence_chunker_class():
    """Show SentenceChunker class for reusable configurations."""
    print("\n" + "="*80)
    print("SENTENCE CHUNKER CLASS")
    print("="*80)
    
    text = """
Embedding models convert text into vector representations. These vectors capture semantic meaning in high-dimensional space. Similar texts produce similar vectors. This enables semantic search and similarity matching. Popular embedding models include OpenAI's text-embedding-ada-002. Sentence Transformers provide open-source alternatives. The choice of model affects retrieval quality. Larger models generally produce better embeddings. However, they also require more computational resources. Consider the trade-off between quality and cost.
    """.strip()
    
    print(f"\nText to chunk ({len(text)} chars):\n{text}\n")
    
    # Create reusable chunker with 4 sentences per window, 2 overlap
    chunker = SentenceChunker(
        window_sentences=4,
        overlap_sentences=2
    )
    
    chunks = chunker.chunk(text)
    
    print(f"\nSentenceChunker created {len(chunks)} chunks (window=4, overlap=2):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  {chunk}")


def demonstrate_overlap_benefits():
    """Show why overlap is critical for RAG systems."""
    print("\n" + "="*80)
    print("OVERLAP BENEFITS FOR RAG")
    print("="*80)
    
    text = """
Chunk size significantly impacts RAG performance. Small chunks (200-400 chars) provide precise retrieval. They match specific facts accurately. However, they may lack sufficient context. Large chunks (800-1200 chars) include more context. This helps the LLM generate better responses. But retrieval precision may decrease. The optimal size depends on your use case. Experiment with different sizes to find the best fit.
    """.strip()
    
    print(f"\nText: {text}\n")
    
    # No overlap - might miss context
    print("Without overlap (window=2, overlap=0):")
    chunks_no_overlap = sentence_window_chunker(text, window_sentences=2, overlap_sentences=0)
    for i, chunk in enumerate(chunks_no_overlap, 1):
        print(f"  Chunk {i}: {chunk}")
    
    # With overlap - preserves context across chunks
    print(f"\nWith overlap (window=2, overlap=1):")
    chunks_with_overlap = sentence_window_chunker(text, window_sentences=2, overlap_sentences=1)
    for i, chunk in enumerate(chunks_with_overlap, 1):
        print(f"  Chunk {i}: {chunk}")
        if i < len(chunks_with_overlap):
            # Find overlapping sentence
            current_sentences = chunk.split('. ')
            next_chunk_sentences = chunks_with_overlap[i].split('. ')
            print(f"    -> Overlaps with next chunk: '{current_sentences[-1]}'")


def demonstrate_rag_pipeline():
    """Simulate a simple RAG pipeline with sentence chunking."""
    print("\n" + "="*80)
    print("RAG PIPELINE SIMULATION")
    print("="*80)
    
    # Knowledge base documents
    documents = [
        """
        Fine-tuning adapts pre-trained models to specific tasks. It requires labeled training data. The process updates model weights through gradient descent. Learning rate and batch size are critical hyperparameters. Monitor validation loss to prevent overfitting. Fine-tuning is more data-efficient than training from scratch.
        """.strip(),
        """
        Prompt engineering optimizes LLM performance without fine-tuning. Techniques include few-shot learning and chain-of-thought prompting. System messages establish context and behavior. Few-shot examples demonstrate desired output format. Clear instructions improve response quality. Iterative refinement is essential for optimal results.
        """.strip(),
        """
        RAG combines retrieval and generation for better accuracy. It reduces hallucinations by grounding responses in retrieved documents. The retrieval step finds relevant context from a knowledge base. This context is added to the LLM prompt. The generation step produces the final response. RAG is particularly effective for knowledge-intensive tasks.
        """.strip()
    ]
    
    print("Knowledge Base Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"\nDoc {i}: {doc[:100]}...")
    
    # Chunk all documents with overlap for better retrieval
    chunker = SentenceChunker(window_sentences=2, overlap_sentences=1)
    
    all_chunks = []
    for doc_id, doc in enumerate(documents, 1):
        doc_chunks = chunker.chunk(doc)
        for chunk in doc_chunks:
            all_chunks.append({
                'doc_id': doc_id,
                'text': chunk,
                'length': len(chunk)
            })
    
    print(f"\n\nChunked into {len(all_chunks)} retrievable chunks:")
    for i, chunk_data in enumerate(all_chunks, 1):
        print(f"\nChunk {i} (from Doc {chunk_data['doc_id']}, {chunk_data['length']} chars):")
        print(f"  {chunk_data['text'][:100]}...")
    
    # Simulate retrieval (in real RAG, this would use embeddings and vector search)
    query = "How do I reduce hallucinations?"
    print(f"\n\nQuery: '{query}'")
    print("\nSimulated retrieval (keyword matching for demo):")
    
    relevant_chunks = [
        chunk for chunk in all_chunks 
        if 'hallucination' in chunk['text'].lower() or 'rag' in chunk['text'].lower()
    ]
    
    if relevant_chunks:
        print(f"\nRetrieved {len(relevant_chunks)} relevant chunks:")
        for chunk_data in relevant_chunks:
            print(f"\n  Doc {chunk_data['doc_id']}: {chunk_data['text']}")
    
    print("\n\nIn production, these chunks would be:")
    print("  1. Embedded using an embedding model")
    print("  2. Stored in a vector database")
    print("  3. Retrieved using semantic similarity search")
    print("  4. Passed as context to the LLM")


def demonstrate_optimal_configurations():
    """Show recommended configurations for different RAG scenarios."""
    print("\n" + "="*80)
    print("OPTIMAL RAG CONFIGURATIONS")
    print("="*80)
    
    sample_text = """
The attention mechanism is fundamental to transformers. It allows models to weigh the importance of different input tokens. Self-attention computes relationships between all token pairs. Multi-head attention uses multiple attention mechanisms in parallel. This captures different types of relationships. Positional encodings add sequence information. The combination enables powerful sequence modeling.
    """.strip()
    
    print(f"\nSample text: {sample_text}\n")
    
    # Configuration 1: Question Answering
    print("Configuration 1: QUESTION ANSWERING")
    print("  Goal: Precise fact retrieval")
    print("  Settings: window=2, overlap=1")
    qa_chunker = SentenceChunker(window_sentences=2, overlap_sentences=1)
    qa_chunks = qa_chunker.chunk(sample_text)
    print(f"  Result: {len(qa_chunks)} chunks")
    for i, chunk in enumerate(qa_chunks, 1):
        print(f"    {i}. {chunk[:80]}...")
    
    # Configuration 2: Summarization
    print("\nConfiguration 2: SUMMARIZATION")
    print("  Goal: Broader context")
    print("  Settings: window=4, overlap=2")
    summary_chunker = SentenceChunker(window_sentences=4, overlap_sentences=2)
    summary_chunks = summary_chunker.chunk(sample_text)
    print(f"  Result: {len(summary_chunks)} chunks")
    for i, chunk in enumerate(summary_chunks, 1):
        print(f"    {i}. {chunk[:80]}...")
    
    # Configuration 3: Chat/Dialogue
    print("\nConfiguration 3: CHAT/DIALOGUE")
    print("  Goal: Conversational context")
    print("  Settings: window=3, overlap=1")
    chat_chunker = SentenceChunker(window_sentences=3, overlap_sentences=1)
    chat_chunks = chat_chunker.chunk(sample_text)
    print(f"  Result: {len(chat_chunks)} chunks")
    for i, chunk in enumerate(chat_chunks, 1):
        print(f"    {i}. {chunk[:80]}...")


def main():
    """Run RAG-specific chunking examples."""
    
    print("\n" + "="*80)
    print("RAG-SPECIFIC CHUNKING EXAMPLES")
    print("="*80)
    print("\nOptimized chunking strategies for retrieval-augmented generation.\n")
    
    demonstrate_sentence_window()
    demonstrate_sentence_chunker_class()
    demonstrate_overlap_benefits()
    demonstrate_rag_pipeline()
    demonstrate_optimal_configurations()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key Takeaways:
- sentence_window_chunker splits text by sentences with overlap
- Overlap is CRITICAL for RAG to preserve context across chunks
- Typical settings: 2-4 sentences per window, 1-2 sentences overlap
- SentenceChunker class allows reusable RAG configurations
- Smaller windows = more precise retrieval
- Larger windows = more context for generation
- Optimize window size based on your specific use case
- Always test retrieval quality with your actual data
    """)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
