"""
Complete RAG Pipeline Example
=============================

This example demonstrates a production-ready RAG pipeline.

Main concepts:
- Building an end-to-end RAG system
- Combining query processing, search, re-ranking, and context management
- Handling different query types
- Performance optimization strategies
- Production best practices
"""

from kerb.retrieval import (
    Document,
    rewrite_query,
    expand_query,
    generate_sub_queries,
    keyword_search,
    semantic_search,
    hybrid_search,
    rerank_results,
    reciprocal_rank_fusion,
    compress_context,
    filter_results,
    diversify_results,
    results_to_context
)
from kerb.embedding import embed, embed_batch


class RAGPipeline:
    """Production-ready RAG pipeline."""
    
    def __init__(self, documents, max_context_tokens=2000):
        """Initialize the RAG pipeline.

# %%
# Setup and Imports
# -----------------
        
        Args:
            documents: List of Document objects
            max_context_tokens: Maximum tokens for context
        """
        self.documents = documents
        self.max_context_tokens = max_context_tokens
        
        # Pre-compute embeddings
        print("Initializing RAG pipeline...")
        print(f"  Documents: {len(documents)}")
        print(f"  Max context tokens: {max_context_tokens}")
        print("  Computing embeddings...")
        
        doc_texts = [doc.content for doc in documents]
        self.doc_embeddings = embed_batch(doc_texts)
        
        print(f"  Ready! ({len(self.doc_embeddings)} embeddings)")
    
    def retrieve(self, query, strategy="hybrid", top_k=10, rerank_method="relevance"):
        """Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            strategy: Search strategy ('keyword', 'semantic', 'hybrid')
            top_k: Number of initial results
            rerank_method: Re-ranking method
            
        Returns:
            List of SearchResult objects
        """
        # Query optimization
        optimized_query = rewrite_query(query, style="clear")
        
        # Search
        if strategy == "keyword":
            results = keyword_search(optimized_query, self.documents, top_k=top_k)
        
        elif strategy == "semantic":
            query_embedding = embed(optimized_query)
            results = semantic_search(
                query_embedding=query_embedding,
                documents=self.documents,
                document_embeddings=self.doc_embeddings,
                top_k=top_k
            )
        
        elif strategy == "hybrid":
            query_embedding = embed(optimized_query)
            results = hybrid_search(
                query=optimized_query,
                query_embedding=query_embedding,
                documents=self.documents,
                document_embeddings=self.doc_embeddings,
                keyword_weight=0.4,
                semantic_weight=0.6,
                top_k=top_k
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Re-rank
        if rerank_method and results:
            results = rerank_results(query, results, method=rerank_method)
        
        return results
    
    def get_context(self, query, strategy="hybrid", apply_diversity=True):
        """Get optimized context for LLM.
        
        Args:
            query: User query string
            strategy: Search strategy
            apply_diversity: Whether to diversify results
            
        Returns:
            String containing formatted context
        """
        # Retrieve
        results = self.retrieve(query, strategy=strategy, top_k=12)
        
        # Filter by quality
        results = filter_results(
            results,
            min_score=0.2,
            dedup_threshold=0.9
        )
        
        # Apply diversity if requested
        if apply_diversity:
            results = diversify_results(results, max_results=8, diversity_factor=0.4)
        
        # Compress to fit context window
        results = compress_context(query, results, max_tokens=self.max_context_tokens)
        
        # Format for LLM
        context = results_to_context(results)
        
        return context, results
    
    def multi_query_retrieve(self, query, max_queries=3):
        """Use multiple query variations for comprehensive retrieval.
        
        Args:
            query: User query string
            max_queries: Maximum sub-queries to generate
            
        Returns:
            Fused search results
        """
        # Generate query variations
        sub_queries = generate_sub_queries(query, max_queries=max_queries)
        
        # Retrieve for each query
        all_results = []
        for sq in sub_queries:
            results = self.retrieve(sq, strategy="hybrid", top_k=8)
            all_results.append(results)
        
        # Fuse results
        fused = reciprocal_rank_fusion(all_results, k=60)
        
        return fused[:10]


def create_knowledge_base():
    """Create a sample knowledge base."""
    return [
        Document(
            id="py-basics",
            content="Python is a high-level, interpreted programming language. It emphasizes code "
                   "readability and uses significant whitespace. Python supports multiple programming "
                   "paradigms including procedural, object-oriented, and functional programming.",
            metadata={"category": "python", "topic": "basics", "difficulty": "beginner"}
        ),
        Document(
            id="py-async",
            content="Asynchronous programming in Python is achieved using async/await syntax and the "
                   "asyncio library. This allows writing concurrent code that can handle I/O operations "
                   "efficiently without blocking. Key concepts include event loops, coroutines, and tasks.",
            metadata={"category": "python", "topic": "async", "difficulty": "intermediate"}
        ),
        Document(
            id="py-web",
            content="Python web frameworks like FastAPI and Flask make it easy to build web applications "
                   "and APIs. FastAPI supports async operations natively and provides automatic API "
                   "documentation. Django is a full-featured framework for larger applications.",
            metadata={"category": "python", "topic": "web", "difficulty": "intermediate"}
        ),
        Document(
            id="ml-intro",
            content="Machine learning enables systems to learn from data without explicit programming. "
                   "Supervised learning uses labeled data, unsupervised learning finds patterns in "
                   "unlabeled data, and reinforcement learning learns through trial and error.",
            metadata={"category": "ml", "topic": "basics", "difficulty": "beginner"}
        ),
        Document(
            id="ml-dl",
            content="Deep learning uses neural networks with multiple layers to learn hierarchical "
                   "representations. Popular architectures include CNNs for images, RNNs for sequences, "
                   "and Transformers for language tasks. Libraries like PyTorch and TensorFlow simplify implementation.",
            metadata={"category": "ml", "topic": "deep-learning", "difficulty": "advanced"}
        ),
        Document(
            id="nlp-basics",
            content="Natural Language Processing (NLP) enables computers to understand and generate "
                   "human language. Common tasks include text classification, named entity recognition, "
                   "sentiment analysis, and machine translation. Modern NLP relies heavily on deep learning.",
            metadata={"category": "ml", "topic": "nlp", "difficulty": "intermediate"}
        ),
        Document(
            id="nlp-transformers",
            content="Transformer models revolutionized NLP with self-attention mechanisms. They can "
                   "process entire sequences in parallel, unlike RNNs. Models like BERT, GPT, and T5 "
                   "are built on the Transformer architecture and achieve state-of-the-art results.",
            metadata={"category": "ml", "topic": "nlp", "difficulty": "advanced"}
        ),
        Document(
            id="api-rest",
            content="REST APIs use HTTP methods (GET, POST, PUT, DELETE) to perform operations on "
                   "resources. They are stateless, cacheable, and follow a client-server architecture. "
                   "RESTful design principles make APIs scalable and easy to understand.",
            metadata={"category": "web", "topic": "api", "difficulty": "beginner"}
        ),
        Document(
            id="api-graphql",
            content="GraphQL is a query language for APIs that allows clients to request exactly the "
                   "data they need. Unlike REST, GraphQL uses a single endpoint and a type system to "
                   "define the API schema. It reduces over-fetching and under-fetching of data.",
            metadata={"category": "web", "topic": "api", "difficulty": "intermediate"}
        ),
        Document(
            id="docker-basics",
            content="Docker containers package applications with their dependencies, ensuring consistency "
                   "across environments. Containers are lightweight, start quickly, and share the host OS "
                   "kernel. Dockerfiles define how to build container images.",
            metadata={"category": "devops", "topic": "containers", "difficulty": "beginner"}
        ),
        Document(
            id="k8s-basics",
            content="Kubernetes orchestrates containerized applications at scale. It handles deployment, "
                   "scaling, load balancing, and self-healing. Key concepts include Pods, Services, "
                   "Deployments, and ConfigMaps. Kubernetes enables declarative configuration.",
            metadata={"category": "devops", "topic": "orchestration", "difficulty": "advanced"}
        ),
        Document(
            id="db-sql",
            content="SQL databases use structured schemas and tables to store data. They provide ACID "
                   "guarantees and support complex queries with JOINs. Popular SQL databases include "
                   "PostgreSQL, MySQL, and SQLite. They're ideal for structured, relational data.",
            metadata={"category": "database", "topic": "sql", "difficulty": "beginner"}
        ),
    ]



# %%
# Main
# ----

def main():
    """Run complete RAG pipeline examples."""
    
    print("="*80)
    print("COMPLETE RAG PIPELINE")
    print("="*80)
    
    # Create knowledge base and pipeline
    kb = create_knowledge_base()
    rag = RAGPipeline(kb, max_context_tokens=1000)
    
    print()
    
    
    # 1. Basic RAG Query
    print("\n1. BASIC RAG QUERY")
    print("-"*80)
    
    query = "How does async programming work in Python?"
    print(f"Query: '{query}'\n")
    
    context, results = rag.get_context(query, strategy="hybrid")
    
    print(f"Retrieved {len(results)} documents:")
    for r in results[:3]:
        print(f"  {r.rank}. {r.document.id} (score: {r.score:.3f})")
        print(f"     {r.document.content[:80]}...")
    
    print(f"\nGenerated context ({len(context)} chars):")
    print(context[:300] + "...")
    
    
    # 2. Multi-Query Retrieval
    print("\n\n2. MULTI-QUERY RETRIEVAL")
    print("-"*80)
    
    complex_query = "What are the differences between REST and GraphQL APIs?"
    print(f"Complex query: '{complex_query}'\n")
    
    print("Breaking down into sub-queries:")
    sub_queries = generate_sub_queries(complex_query, max_queries=3)
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")
    
    fused_results = rag.multi_query_retrieve(complex_query, max_queries=3)
    
    print(f"\nFused results ({len(fused_results)} documents):")
    for r in fused_results[:5]:
        print(f"  {r.rank}. {r.document.id} (score: {r.score:.3f})")
        print(f"     Topic: {r.document.metadata.get('topic')}")
    
    
    # 3. Strategy Comparison
    print("\n\n3. SEARCH STRATEGY COMPARISON")
    print("-"*80)
    
    query = "machine learning neural networks"
    print(f"Query: '{query}'\n")
    
    strategies = ["keyword", "semantic", "hybrid"]
    
    for strategy in strategies:
        results = rag.retrieve(query, strategy=strategy, top_k=5)
        print(f"  {strategy.upper():10} strategy:")
        print(f"    Top results: {[r.document.id for r in results[:3]]}")
    
    
    # 4. Diversity Control
    print("\n\n4. DIVERSITY CONTROL")
    print("-"*80)
    
    query = "programming languages"
    print(f"Query: '{query}'\n")
    
    print("Without diversity:")
    context_no_div, results_no_div = rag.get_context(query, apply_diversity=False)
    print(f"  Documents: {[r.document.id for r in results_no_div[:5]]}")
    print(f"  Topics: {[r.document.metadata.get('topic') for r in results_no_div[:5]]}")
    
    print("\nWith diversity:")
    context_div, results_div = rag.get_context(query, apply_diversity=True)
    print(f"  Documents: {[r.document.id for r in results_div[:5]]}")
    print(f"  Topics: {[r.document.metadata.get('topic') for r in results_div[:5]]}")
    
    
    # 5. Context Window Management
    print("\n\n5. CONTEXT WINDOW MANAGEMENT")
    print("-"*80)
    
    query = "web development with Python"
    
    token_limits = [500, 1000, 2000]
    
    print(f"Query: '{query}'\n")
    
    for limit in token_limits:
        rag_temp = RAGPipeline(kb, max_context_tokens=limit)
        context, results = rag_temp.get_context(query)
        actual_tokens = len(context) // 4
        
        print(f"  Token limit: {limit:4} | Actual: ~{actual_tokens:4} | Docs: {len(results)}")
    
    
    # 6. Production RAG Workflow
    print("\n\n6. PRODUCTION RAG WORKFLOW")
    print("-"*80)
    
    def rag_qa_system(user_query):
        """Complete RAG-based Q&A system."""
        print(f"User query: '{user_query}'")
        
        # Step 1: Query optimization
        optimized = rewrite_query(user_query, style="clear")
        print(f"  1. Optimized: '{optimized}'")
        
        # Step 2: Retrieval
        context, results = rag.get_context(optimized, strategy="hybrid", apply_diversity=True)
        print(f"  2. Retrieved: {len(results)} documents")
        
        # Step 3: Context formatting
        context_tokens = len(context) // 4
        print(f"  3. Context: ~{context_tokens} tokens")
        
        # Step 4: Prepare for LLM
        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {user_query}

Answer:"""
        
        prompt_tokens = len(prompt) // 4
        print(f"  4. Full prompt: ~{prompt_tokens} tokens")
        
        return {
            'query': user_query,
            'optimized_query': optimized,
            'results': results,
            'context': context,
            'prompt': prompt
        }
    
    print("\nExample workflow:\n")
    response = rag_qa_system("How do I build a web API?")
    
    print(f"\nRetrieved documents:")
    for r in response['results'][:3]:
        print(f"  - {r.document.id}: {r.document.metadata.get('topic')}")
    
    
    # 7. Performance Tips
    print("\n\n7. PRODUCTION PERFORMANCE TIPS")
    print("-"*80)
    
    tips = [
        "Pre-compute and cache document embeddings",
        "Use hybrid search for best accuracy/recall trade-off",
        "Apply re-ranking only on top-k results (not all documents)",
        "Implement caching for frequent queries",
        "Use async operations for parallel retrieval",
        "Monitor token usage to stay within limits",
        "Batch embed operations when possible",
        "Consider approximate nearest neighbor search for large collections"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")
    
    
    # 8. Quality Metrics
    print("\n\n8. RAG QUALITY CONSIDERATIONS")
    print("-"*80)
    
    print("\nKey metrics to monitor:")
    print("  - Retrieval precision: Are retrieved docs relevant?")
    print("  - Retrieval recall: Are all relevant docs found?")
    print("  - Context relevance: Is context useful for answering?")
    print("  - Answer accuracy: Does LLM produce correct answers?")
    print("  - Latency: Time from query to response")
    print("  - Token efficiency: Context fits within limits")
    
    print("\nOptimization strategies:")
    print("  - A/B test different retrieval strategies")
    print("  - Tune keyword/semantic weights")
    print("  - Adjust diversity factors")
    print("  - Experiment with re-ranking methods")
    print("  - Fine-tune embedding models on your domain")
    
    
    print("\n" + "="*80)
    print("Complete RAG pipeline demonstration finished!")
    print("="*80)


if __name__ == "__main__":
    main()
