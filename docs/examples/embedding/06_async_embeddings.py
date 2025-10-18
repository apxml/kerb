"""Async Embeddings Example

This example demonstrates asynchronous embedding operations for concurrent processing.

Main concepts:
- Using embed_async() for single async embeddings
- Processing multiple texts with embed_batch_async()
- Streaming large datasets asynchronously
- Concurrent processing for better performance
- Handling async operations properly
"""

import asyncio
import time
from kerb.embedding import (
    embed,
    embed_async,
    embed_batch_async,
    embed_batch_stream_async,
    cosine_similarity
)


async def example_1_basic_async():
    """Example 1: Basic async embedding."""
    print("\n1. BASIC ASYNC EMBEDDING")
    print("-"*80)
    
    text = "Asynchronous operations improve performance"
    
    print(f"Text: '{text}'")
    print("Generating embedding asynchronously...")
    
    # Note: For local model, this runs in thread pool
    # Use the default local model to avoid API key requirement
    from kerb.embedding import EmbeddingModel
    embedding = await embed_async(text, model=EmbeddingModel.LOCAL)
    
    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"First 3 values: {[round(v, 4) for v in embedding[:3]]}")


async def example_2_concurrent_requests():
    """Example 2: Concurrent async requests."""
    print("\n2. CONCURRENT ASYNC REQUESTS")
    print("-"*80)
    
    texts = [
        "First document about machine learning",
        "Second document about data science",
        "Third document about artificial intelligence",
        "Fourth document about neural networks",
        "Fifth document about deep learning"
    ]
    
    print(f"Processing {len(texts)} texts concurrently...")
    
    from kerb.embedding import EmbeddingModel
    
    # Launch all embedding tasks concurrently
    start_time = time.time()
    tasks = [embed_async(text, model=EmbeddingModel.LOCAL) for text in texts]
    embeddings = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.4f}s")
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Each embedding: {len(embeddings[0])} dimensions")


async def example_3_batch_async():
    """Example 3: Batch async processing."""
    print("\n3. BATCH ASYNC PROCESSING")
    print("-"*80)
    
    documents = [
        f"Document {i}: Content about topic {i % 3}"
        for i in range(20)
    ]
    
    print(f"Processing {len(documents)} documents in async batches...")
    
    from kerb.embedding import EmbeddingModel
    
    start_time = time.time()
    
    # Use local model by default to avoid API key requirement
    embeddings = await embed_batch_async(
        documents, 
        model=EmbeddingModel.LOCAL,
        batch_size=5, 
        max_concurrent=3
    )
    
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.4f}s")
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Average time per document: {elapsed/len(documents):.4f}s")


async def example_4_streaming_async():
    """Example 4: Async streaming for large datasets."""
    print("\n4. ASYNC STREAMING")
    print("-"*80)
    
    # Simulate large dataset
    large_dataset = [
        f"Article {i}: Content about various topics in field {i % 5}"
        for i in range(50)
    ]
    
    print(f"Streaming {len(large_dataset)} documents asynchronously...")
    
    from kerb.embedding import EmbeddingModel
    
    processed_count = 0
    embeddings_by_index = {}
    
    start_time = time.time()
    
    # Stream embeddings as they're generated
    async for idx, embedding in embed_batch_stream_async(
        large_dataset,
        model=EmbeddingModel.LOCAL,
        batch_size=10
    ):
        embeddings_by_index[idx] = embedding
        processed_count += 1
        
        # Show progress every 10 documents
        if processed_count % 10 == 0:
            print(f"  Processed {processed_count}/{len(large_dataset)} documents")
    
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.4f}s")
    print(f"Stored {len(embeddings_by_index)} embeddings")


async def example_5_parallel_search():
    """Example 5: Parallel semantic search."""
    print("\n5. PARALLEL SEMANTIC SEARCH")
    print("-"*80)
    
    from kerb.embedding import EmbeddingModel
    
    # Document collection
    documents = [
        "Python programming basics",
        "Machine learning fundamentals",
        "Web development with JavaScript",
        "Data analysis techniques",
        "Cloud computing platforms",
        "Database design principles",
        "Mobile app development",
        "Cybersecurity best practices"
    ]
    
    # Multiple queries to search
    queries = [
        "programming languages",
        "AI and ML",
        "web technologies"
    ]
    
    print(f"Searching {len(queries)} queries across {len(documents)} documents...")
    
    # Embed all documents first
    doc_embeddings = await embed_batch_async(documents, model=EmbeddingModel.LOCAL)
    
    # Process all queries in parallel
    async def search_query(query, doc_embeddings, documents):
        query_emb = await embed_async(query, model=EmbeddingModel.LOCAL)
        
        # Find best match
        similarities = [
            cosine_similarity(query_emb, doc_emb)
            for doc_emb in doc_embeddings
        ]
        
        best_idx = similarities.index(max(similarities))
        return {
            'query': query,
            'best_match': documents[best_idx],
            'score': similarities[best_idx]
        }
    
    # Run all searches concurrently
    search_tasks = [
        search_query(query, doc_embeddings, documents)
        for query in queries
    ]
    
    results = await asyncio.gather(*search_tasks)
    
    print("\nResults:")
    for result in results:
        print(f"\nQuery: '{result['query']}'")
        print(f"  Best match: {result['best_match']}")
        print(f"  Score: {result['score']:.4f}")


async def example_6_rate_limiting():
    """Example 6: Rate limiting with semaphore."""
    print("\n6. RATE LIMITING WITH SEMAPHORE")
    print("-"*80)
    
    from kerb.embedding import EmbeddingModel
    
    texts = [f"Text {i} for processing" for i in range(15)]
    
    # Limit to 3 concurrent operations
    semaphore = asyncio.Semaphore(3)
    
    async def rate_limited_embed(text, sem, idx):
        async with sem:
            print(f"  Processing text {idx}...")
            embedding = await embed_async(text, model=EmbeddingModel.LOCAL)
            await asyncio.sleep(0.1)  # Simulate processing time
            return embedding
    
    print(f"Processing {len(texts)} texts with max 3 concurrent operations...")
    
    tasks = [
        rate_limited_embed(text, semaphore, i)
        for i, text in enumerate(texts)
    ]
    
    embeddings = await asyncio.gather(*tasks)
    
    print(f"Completed: {len(embeddings)} embeddings generated")


async def example_7_error_handling():
    """Example 7: Error handling in async operations."""
    print("\n7. ERROR HANDLING")
    print("-"*80)
    
    from kerb.embedding import EmbeddingModel
    
    texts = [
        "Valid text 1",
        "Valid text 2",
        "",  # Empty text might cause issues
        "Valid text 3"
    ]
    
    print("Processing texts with error handling...")
    
    async def safe_embed(text, idx):
        try:
            if not text:
                print(f"  Warning: Empty text at index {idx}, using default")
                text = "default text"
            
            embedding = await embed_async(text, model=EmbeddingModel.LOCAL)
            return (idx, embedding, None)
        except Exception as e:
            print(f"  Error at index {idx}: {e}")
            return (idx, None, str(e))
    
    tasks = [safe_embed(text, i) for i, text in enumerate(texts)]
    results = await asyncio.gather(*tasks)
    
    successful = sum(1 for _, emb, err in results if emb is not None)
    failed = sum(1 for _, emb, err in results if emb is None)
    
    print(f"\nResults:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


def main():
    """Run async embeddings example."""
    
    print("="*80)
    print("ASYNC EMBEDDINGS EXAMPLE")
    print("="*80)
    
    # Run all async examples
    asyncio.run(example_1_basic_async())
    asyncio.run(example_2_concurrent_requests())
    asyncio.run(example_3_batch_async())
    asyncio.run(example_4_streaming_async())
    asyncio.run(example_5_parallel_search())
    asyncio.run(example_6_rate_limiting())
    asyncio.run(example_7_error_handling())
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKEY TAKEAWAYS:")
    print("- Use embed_async() for non-blocking embedding generation")
    print("- asyncio.gather() runs multiple operations concurrently")
    print("- embed_batch_async() handles batching automatically")
    print("- Streaming async is memory-efficient for large datasets")
    print("- Semaphores control concurrency and rate limiting")
    print("- Always handle errors in async operations")
    print("- Async is most beneficial with API-based models")


if __name__ == "__main__":
    main()
