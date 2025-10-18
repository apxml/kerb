"""Batch Processing and Streaming Example

This example demonstrates efficient batch processing and streaming for large datasets.

Main concepts:
- Processing multiple texts efficiently with embed_batch()
- Memory-efficient streaming with embed_batch_stream()
- Handling large document collections
- Performance optimization techniques
"""

from kerb.embedding import (
    embed,
    embed_batch,
    embed_batch_stream,
    embedding_dimension
)
import time


def main():
    """Run batch processing example."""
    
    print("="*80)
    print("BATCH PROCESSING AND STREAMING EXAMPLE")
    print("="*80)
    
    # 1. Basic batch processing
    print("\n1. BASIC BATCH PROCESSING")
    print("-"*80)
    
    texts = [
        "First document about Python programming",
        "Second document about machine learning",
        "Third document about data science",
        "Fourth document about web development",
        "Fifth document about cloud computing"
    ]
    
    print(f"Processing {len(texts)} documents in batch...")
    
    start_time = time.time()
    embeddings = embed_batch(texts)
    batch_time = time.time() - start_time
    
    print(f"Generated {len(embeddings)} embeddings in {batch_time:.4f}s")
    print(f"Each embedding has {len(embeddings[0])} dimensions")
    
    # 2. Batch vs individual processing
    print("\n2. BATCH VS INDIVIDUAL COMPARISON")
    print("-"*80)
    
    sample_texts = [
        "Natural language processing",
        "Computer vision applications",
        "Reinforcement learning algorithms",
        "Deep neural networks",
        "Transfer learning methods"
    ]
    
    # Individual processing
    print("Processing individually...")
    start_time = time.time()
    individual_embeddings = [embed(text) for text in sample_texts]
    individual_time = time.time() - start_time
    
    # Batch processing
    print("Processing in batch...")
    start_time = time.time()
    batch_embeddings = embed_batch(sample_texts)
    batch_time = time.time() - start_time
    
    print(f"\nIndividual processing: {individual_time:.4f}s")
    print(f"Batch processing: {batch_time:.4f}s")
    print(f"Speedup: {individual_time/batch_time:.2f}x")
    
    # 3. Streaming for large datasets
    print("\n3. STREAMING FOR LARGE DATASETS")
    print("-"*80)
    
    # Simulate a large dataset
    large_dataset = [
        f"Document {i}: This is document number {i} about various topics"
        for i in range(100)
    ]
    
    print(f"Processing {len(large_dataset)} documents with streaming...")
    
    # Process and save embeddings one at a time (memory efficient)
    processed_count = 0
    embeddings_store = {}
    
    for idx, embedding in embed_batch_stream(large_dataset, batch_size=20):
        embeddings_store[idx] = embedding
        processed_count += 1
        
        # Show progress every 25 documents
        if processed_count % 25 == 0:
            print(f"  Processed {processed_count}/{len(large_dataset)} documents")
    
    print(f"Completed: {len(embeddings_store)} embeddings stored")
    
    # 4. Batch size optimization
    print("\n4. BATCH SIZE OPTIMIZATION")
    print("-"*80)
    
    test_texts = [f"Test document {i}" for i in range(50)]
    
    batch_sizes = [5, 10, 25, 50]
    
    print(f"Testing different batch sizes on {len(test_texts)} documents:")
    
    for batch_size in batch_sizes:
        start_time = time.time()
        embeddings = embed_batch(test_texts, batch_size=batch_size)
        elapsed = time.time() - start_time
        
        print(f"  Batch size {batch_size:3d}: {elapsed:.4f}s")
    
    # 5. Processing document chunks
    print("\n5. PROCESSING DOCUMENT CHUNKS")
    print("-"*80)
    
    # Simulate long documents that need chunking
    long_documents = [
        "Part A: Introduction to the topic. " * 10,
        "Part B: Main content and details. " * 10,
        "Part C: Conclusion and summary. " * 10
    ]
    
    print(f"Processing {len(long_documents)} documents...")
    
    # Chunk and embed
    all_chunks = []
    chunk_metadata = []
    
    for doc_id, doc in enumerate(long_documents):
        # Split into sentences (simple split for demo)
        chunks = [s.strip() for s in doc.split('.') if s.strip()]
        all_chunks.extend(chunks)
        
        # Track which document each chunk belongs to
        for chunk in chunks:
            chunk_metadata.append({'doc_id': doc_id, 'text': chunk})
    
    print(f"Created {len(all_chunks)} chunks")
    
    # Batch process all chunks
    chunk_embeddings = embed_batch(all_chunks, batch_size=10)
    print(f"Generated {len(chunk_embeddings)} chunk embeddings")
    
    # 6. Real-world scenario: Incremental indexing
    print("\n6. INCREMENTAL INDEXING")
    print("-"*80)
    
    class IncrementalIndex:
        """Index that processes documents incrementally."""
        
        def __init__(self):
            self.documents = []
            self.embeddings = []
        
        def add_batch(self, new_documents):
            """Add a batch of documents to the index."""
            print(f"  Adding {len(new_documents)} documents...")
            
            # Generate embeddings for new documents
            new_embeddings = embed_batch(new_documents, batch_size=10)
            
            # Add to index
            self.documents.extend(new_documents)
            self.embeddings.extend(new_embeddings)
            
            print(f"  Index now contains {len(self.documents)} documents")
        
        def get_stats(self):
            """Get index statistics."""
            return {
                'total_docs': len(self.documents),
                'embedding_dim': len(self.embeddings[0]) if self.embeddings else 0
            }
    
    # Create index and add documents incrementally
    index = IncrementalIndex()
    
    # First batch
    batch1 = [f"Initial document {i}" for i in range(10)]
    index.add_batch(batch1)
    
    # Second batch
    batch2 = [f"New document {i}" for i in range(10, 20)]
    index.add_batch(batch2)
    
    # Third batch
    batch3 = [f"Latest document {i}" for i in range(20, 30)]
    index.add_batch(batch3)
    
    stats = index.get_stats()
    print(f"\nFinal index statistics:")
    print(f"  Total documents: {stats['total_docs']}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKEY TAKEAWAYS:")
    print("- Use embed_batch() for better performance on multiple texts")
    print("- Use embed_batch_stream() for memory-efficient processing")
    print("- Adjust batch_size based on your use case and memory constraints")
    print("- Streaming is ideal for large datasets or real-time processing")
    print("- Incremental indexing allows building indexes over time")


if __name__ == "__main__":
    main()
