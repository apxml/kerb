"""RAG Context Management Example

This example demonstrates context management for Retrieval-Augmented
Generation (RAG) systems, a critical LLM use case.

Main concepts:
- Managing retrieved document context
- Combining query context with retrieved content
- Source attribution and metadata tracking
- Multi-source context merging
- Dynamic context selection based on relevance
"""

from kerb.context import (
    ContextItem,
    ContextWindow,
    create_context_window,
    truncate_context_window,
    TruncationStrategy,
    priority_by_relevance,
    deduplicate_context,
    merge_context_windows,
    optimize_context_for_query,
)
from kerb.tokenizer import count_tokens


def create_document_corpus():
    """Create a sample document corpus for RAG."""
    documents = [
        {
            "id": "doc1",
            "content": "FastAPI is a modern Python web framework for building APIs with automatic OpenAPI documentation.",
            "source": "FastAPI Documentation",
            "relevance_score": 0.95
        },
        {
            "id": "doc2",
            "content": "FastAPI supports async/await syntax enabling high-performance concurrent request handling.",
            "source": "FastAPI Guide",
            "relevance_score": 0.88
        },
        {
            "id": "doc3",
            "content": "Pydantic models in FastAPI provide automatic data validation and serialization.",
            "source": "FastAPI Tutorial",
            "relevance_score": 0.82
        },
        {
            "id": "doc4",
            "content": "Django is a full-featured web framework with built-in admin interface and ORM.",
            "source": "Django Documentation",
            "relevance_score": 0.45
        },
        {
            "id": "doc5",
            "content": "FastAPI dependency injection system enables clean and testable code architecture.",
            "source": "FastAPI Advanced",
            "relevance_score": 0.78
        },
    ]
    return documents


def main():
    """Run RAG context management example."""
    
    print("="*80)
    print("RAG CONTEXT MANAGEMENT EXAMPLE")
    print("="*80)
    
    # Example 1: Basic RAG context assembly
    print("\n1. BASIC RAG CONTEXT ASSEMBLY")
    print("-"*80)
    print("Use case: Combine query with retrieved documents")
    
    query = "How do I build APIs with FastAPI?"
    documents = create_document_corpus()
    
    # Create context items from documents
    doc_items = []
    for doc in documents:
        item = ContextItem(
            content=doc["content"],
            token_count=count_tokens(doc["content"]),
            priority=doc["relevance_score"],
            metadata={
                "doc_id": doc["id"],
                "source": doc["source"],
                "type": "retrieved_doc"
            }
        )
        doc_items.append(item)
    
    # Add query context
    query_item = ContextItem(
        content=f"User query: {query}",
        token_count=count_tokens(f"User query: {query}"),
        priority=1.0,
        metadata={"type": "query"}
    )
    
    # Combine into RAG context
    rag_items = [query_item] + doc_items
    rag_window = create_context_window(rag_items, max_tokens=300)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(doc_items)} documents")
    print(f"Total context: {rag_window.current_tokens} tokens")
    
    print("\nContext items by relevance:")
    for item in sorted(rag_items[1:], key=lambda x: x.priority, reverse=True):
        source = item.metadata.get("source", "unknown")
        print(f"  [{source}] Relevance: {item.priority:.2f}")
        print(f"    {item.content[:70]}...")
    
    # Example 2: Token-limited RAG context
    print("\n2. TOKEN-LIMITED RAG CONTEXT")
    print("-"*80)
    print("Use case: Fit retrieved documents within token budget")
    
    token_budget = 150
    print(f"\nToken budget: {token_budget}")
    print(f"Original context: {rag_window.current_tokens} tokens")
    
    # Truncate using priority (keep most relevant)
    limited_window = truncate_context_window(
        rag_window,
        token_budget,
        TruncationStrategy.PRIORITY
    )
    
    print(f"After truncation: {limited_window.current_tokens} tokens")
    print(f"Kept {len(limited_window.items)} items")
    
    print("\nSelected documents:")
    for item in limited_window.items:
        if item.metadata.get("type") == "retrieved_doc":
            source = item.metadata.get("source", "unknown")
            print(f"  [{source}] {item.content[:60]}...")
    
    # Example 3: Multi-source RAG
    print("\n3. MULTI-SOURCE RAG")
    print("-"*80)
    print("Use case: Combine information from different knowledge sources")
    
    # Create different knowledge sources
    documentation = create_context_window([
        ContextItem(
            content="FastAPI path operations use Python decorators like @app.get()",
            token_count=count_tokens("FastAPI path operations use Python decorators like @app.get()"),
            priority=0.9,
            metadata={"source_type": "documentation"}
        ),
    ])
    
    code_examples = create_context_window([
        ContextItem(
            content="Example: @app.post('/users/') async def create_user(user: User): return user",
            token_count=count_tokens("Example: @app.post('/users/') async def create_user(user: User): return user"),
            priority=0.85,
            metadata={"source_type": "code"}
        ),
    ])
    
    community_qa = create_context_window([
        ContextItem(
            content="Q: Best practices for FastAPI? A: Use async for I/O operations, leverage Pydantic models.",
            token_count=count_tokens("Q: Best practices for FastAPI? A: Use async for I/O operations, leverage Pydantic models."),
            priority=0.7,
            metadata={"source_type": "community"}
        ),
    ])
    
    # Merge sources
    multi_source = merge_context_windows(
        [documentation, code_examples, community_qa],
        max_tokens=200
    )
    
    print(f"Merged {len(multi_source.items)} items from 3 sources")
    print("\nSources included:")
    for item in multi_source.items:
        source_type = item.metadata.get("source_type", "unknown")
        print(f"  [{source_type}] {item.content[:60]}...")
    
    # Example 4: Source attribution tracking
    print("\n4. SOURCE ATTRIBUTION TRACKING")
    print("-"*80)
    print("Use case: Track which sources contributed to the response")
    
    attributed_docs = [
        ContextItem(
            content="FastAPI supports WebSocket connections for real-time features.",
            token_count=count_tokens("FastAPI supports WebSocket connections for real-time features."),
            priority=0.88,
            metadata={
                "source": "FastAPI WebSocket Guide",
                "url": "https://fastapi.tiangolo.com/advanced/websockets/",
                "page": 42,
                "timestamp": "2024-01-15"
            }
        ),
        ContextItem(
            content="Background tasks in FastAPI run after returning response.",
            token_count=count_tokens("Background tasks in FastAPI run after returning response."),
            priority=0.75,
            metadata={
                "source": "FastAPI Background Tasks",
                "url": "https://fastapi.tiangolo.com/tutorial/background-tasks/",
                "page": 38,
                "timestamp": "2024-01-10"
            }
        ),
    ]
    
    attributed_window = create_context_window(attributed_docs)
    
    print("\nDocuments with full attribution:")
    for item in attributed_window.items:
        meta = item.metadata
        print(f"\nSource: {meta.get('source', 'Unknown')}")
        print(f"  URL: {meta.get('url', 'N/A')}")
        print(f"  Page: {meta.get('page', 'N/A')}")
        print(f"  Date: {meta.get('timestamp', 'N/A')}")
        print(f"  Content: {item.content}")
    
    # Example 5: Query-optimized RAG context
    print("\n5. QUERY-OPTIMIZED RAG CONTEXT")
    print("-"*80)
    print("Use case: Dynamically select most relevant documents for query")
    
    # Large document set
    large_corpus = []
    topics = [
        ("routing", "FastAPI routing uses decorators for path operations"),
        ("validation", "Pydantic models provide automatic request validation"),
        ("async", "Async functions enable concurrent request handling"),
        ("dependency", "Dependency injection manages shared resources"),
        ("security", "FastAPI includes security utilities for OAuth2 and JWT"),
        ("testing", "TestClient enables easy API endpoint testing"),
        ("deployment", "FastAPI apps deploy with uvicorn ASGI server"),
        ("middleware", "Middleware processes requests before reaching endpoints"),
    ]
    
    for topic, content in topics:
        large_corpus.append(ContextItem(
            content=content,
            token_count=count_tokens(content),
            priority=0.7,
            metadata={"topic": topic}
        ))
    
    corpus_window = create_context_window(large_corpus)
    
    # Different queries
    queries = [
        "How do I validate request data?",
        "What about async performance?",
        "How to deploy FastAPI?",
    ]
    
    print(f"Document corpus: {len(large_corpus)} documents")
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Optimize context for this specific query
        optimized = optimize_context_for_query(
            corpus_window,
            query,
            max_tokens=50
        )
        
        print(f"Selected {len(optimized.items)} most relevant documents:")
        for item in optimized.items:
            topic = item.metadata.get("topic", "unknown")
            print(f"  [{topic}] {item.content}")
    
    # Example 6: Deduplication in RAG
    print("\n6. DEDUPLICATION IN RAG")
    print("-"*80)
    print("Use case: Remove similar documents from retrieval results")
    
    # Simulated retrieval with duplicates
    retrieval_results = [
        ContextItem(
            content="FastAPI provides automatic API documentation with Swagger UI.",
            token_count=count_tokens("FastAPI provides automatic API documentation with Swagger UI."),
            priority=0.92,
            metadata={"source": "doc_a"}
        ),
        ContextItem(
            content="Use Pydantic models for request and response schemas in FastAPI.",
            token_count=count_tokens("Use Pydantic models for request and response schemas in FastAPI."),
            priority=0.87,
            metadata={"source": "doc_b"}
        ),
        ContextItem(
            content="FastAPI offers automatic API documentation through Swagger UI.",
            token_count=count_tokens("FastAPI offers automatic API documentation through Swagger UI."),
            priority=0.90,
            metadata={"source": "doc_c"}
        ),
        ContextItem(
            content="Dependency injection in FastAPI manages shared resources efficiently.",
            token_count=count_tokens("Dependency injection in FastAPI manages shared resources efficiently."),
            priority=0.85,
            metadata={"source": "doc_d"}
        ),
    ]
    
    retrieval_window = create_context_window(retrieval_results)
    
    print(f"\nRetrieved documents: {len(retrieval_window.items)}")
    for item in retrieval_window.items:
        source = item.metadata.get("source", "unknown")
        print(f"  [{source}] {item.content[:50]}...")
    
    # Deduplicate
    deduped_items = deduplicate_context(retrieval_window.items, similarity_threshold=0.85)
    deduped = create_context_window(deduped_items)
    
    print(f"\nAfter deduplication: {len(deduped.items)} unique documents")
    for item in deduped.items:
        source = item.metadata.get("source", "unknown")
        print(f"  [{source}] {item.content}")
    
    # Example 7: RAG context template
    print("\n7. RAG CONTEXT TEMPLATE")
    print("-"*80)
    print("Use case: Structured template for RAG prompts")
    
    def format_rag_prompt(query: str, documents: list, max_docs: int = 3) -> str:
        """Format RAG context into structured prompt."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("CONTEXT INFORMATION")
        lines.append("=" * 60)
        
        # Documents
        lines.append(f"\nRelevant documents (showing top {min(len(documents), max_docs)}):\n")
        
        for i, doc in enumerate(documents[:max_docs], 1):
            source = doc.metadata.get("source", "Unknown")
            lines.append(f"Document {i} (Source: {source}):")
            lines.append(f"{doc.content}")
            lines.append("")
        
        # Query
        lines.append("=" * 60)
        lines.append("QUERY")
        lines.append("=" * 60)
        lines.append(f"\n{query}\n")
        
        # Instruction
        lines.append("=" * 60)
        lines.append("INSTRUCTION")
        lines.append("=" * 60)
        lines.append("\nUsing the context documents above, provide a comprehensive")
        lines.append("answer to the query. Cite sources where appropriate.")
        
        return "\n".join(lines)
    
    sample_query = "How do I use dependency injection in FastAPI?"
    sample_docs = [
        ContextItem(
            content="FastAPI dependency injection uses Depends() to inject dependencies into path operations.",
            token_count=50,
            metadata={"source": "FastAPI Dependencies Guide"}
        ),
        ContextItem(
            content="Dependencies can be functions, classes, or other callables that provide reusable logic.",
            token_count=50,
            metadata={"source": "FastAPI Advanced Features"}
        ),
    ]
    
    formatted_prompt = format_rag_prompt(sample_query, sample_docs)
    
    print("\nFormatted RAG prompt:")
    print(formatted_prompt)
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
