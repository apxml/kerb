"""Result Formatting Example

This example demonstrates formatting search results for LLM consumption.

Main concepts:
- Converting results to context strings
- Different formatting styles for different use cases
- Optimizing format for token efficiency
- Adding metadata and citations
"""

from kerb.retrieval import (
    Document,
    SearchResult,
    keyword_search,
    format_results,
    results_to_context
)


def create_sample_documents():
    """Create sample documents with rich metadata."""
    return [
        Document(
            id="doc1",
            content="Python is a high-level programming language known for simplicity and readability.",
            metadata={
                "title": "Introduction to Python",
                "author": "Alice Smith",
                "date": "2024-01-15",
                "source": "python-guide.com",
                "category": "programming"
            }
        ),
        Document(
            id="doc2",
            content="Asynchronous programming in Python uses async/await syntax for concurrent operations.",
            metadata={
                "title": "Async Python Guide",
                "author": "Bob Jones",
                "date": "2024-03-20",
                "source": "async-python.dev",
                "category": "programming"
            }
        ),
        Document(
            id="doc3",
            content="FastAPI is a modern web framework for building APIs with Python and type hints.",
            metadata={
                "title": "FastAPI Tutorial",
                "author": "Alice Smith",
                "date": "2024-02-10",
                "source": "fastapi-tutorial.com",
                "category": "web"
            }
        ),
        Document(
            id="doc4",
            content="Machine learning models learn patterns from data to make predictions on new data.",
            metadata={
                "title": "ML Fundamentals",
                "author": "Charlie Brown",
                "date": "2024-01-05",
                "source": "ml-basics.org",
                "category": "ai"
            }
        ),
    ]


def main():
    """Run result formatting examples."""
    
    print("="*80)
    print("RESULT FORMATTING FOR LLM CONSUMPTION")
    print("="*80)
    
    documents = create_sample_documents()
    query = "python async programming"
    results = keyword_search(query, documents, top_k=4)
    
    print(f"\nQuery: '{query}'")
    print(f"Results: {len(results)} documents\n")
    
    
    # 1. Basic Result Formatting
    print("\n1. BASIC RESULT FORMATTING")
    print("-"*80)
    
    formatted = format_results(results, format_style="simple")
    print("SIMPLE style (simple list):")
    print(formatted)
    
    print("\n" + "-"*40)
    formatted = format_results(results, format_style="simple")  # Use simple since no numbered option
    print("\nSIMPLE style (with rankings):")
    print(formatted)
    
    print("\n" + "-"*40)
    formatted = format_results(results, format_style="detailed", include_metadata=True)
    print("\nDETAILED style (with metadata):")
    print(formatted)
    
    
    # 2. Context String Generation
    print("\n\n2. CONTEXT STRING GENERATION")
    print("-"*80)
    print("Convert results to context strings for LLM prompts.\n")
    
    context = results_to_context(results)
    
    print("Generated context:")
    print(context)
    
    print(f"\nContext length: {len(context)} characters (~{len(context) // 4} tokens)")
    
    
    # 3. Custom Formatting
    print("\n\n3. CUSTOM FORMATTING TEMPLATES")
    print("-"*80)
    
    def format_with_citations(results):
        """Format results with citation numbers."""
        output = []
        for i, result in enumerate(results, 1):
            doc = result.document
            title = doc.metadata.get('title', 'Untitled')
            author = doc.metadata.get('author', 'Unknown')
            
            output.append(f"[{i}] {title} by {author}")
            output.append(f"    {doc.content}")
            output.append("")
        
        return "\n".join(output)
    
    print("Custom citation format:")
    print(format_with_citations(results))
    
    
    # 4. Compact Format for Token Efficiency
    print("\n\n4. COMPACT FORMAT (Token Efficient)")
    print("-"*80)
    
    def compact_format(results, max_length=100):
        """Ultra-compact format to save tokens."""
        output = []
        for r in results:
            # Truncate content
            content = r.document.content
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            # Minimal formatting
            output.append(f"- {content}")
        
        return "\n".join(output)
    
    print("Compact format (max 100 chars per result):")
    compact = compact_format(results, max_length=100)
    print(compact)
    print(f"\nLength: {len(compact)} chars (~{len(compact) // 4} tokens)")
    
    
    # 5. Structured Format with Metadata
    print("\n\n5. STRUCTURED FORMAT WITH METADATA")
    print("-"*80)
    
    def structured_format(results):
        """Format with structured metadata."""
        output = ["Retrieved Documents:\n"]
        
        for i, result in enumerate(results, 1):
            doc = result.document
            
            output.append(f"Document {i}:")
            output.append(f"  Title: {doc.metadata.get('title', 'N/A')}")
            output.append(f"  Source: {doc.metadata.get('source', 'N/A')}")
            output.append(f"  Date: {doc.metadata.get('date', 'N/A')}")
            output.append(f"  Relevance: {result.score:.3f}")
            output.append(f"  Content: {doc.content}")
            output.append("")
        
        return "\n".join(output)
    
    print(structured_format(results))
    
    
    # 6. Markdown Format
    print("\n\n6. MARKDOWN FORMAT")
    print("-"*80)
    
    def markdown_format(results):
        """Format results as markdown."""
        output = ["# Search Results\n"]
        
        for i, result in enumerate(results, 1):
            doc = result.document
            title = doc.metadata.get('title', 'Document')
            author = doc.metadata.get('author', 'Unknown')
            source = doc.metadata.get('source', 'N/A')
            
            output.append(f"## {i}. {title}")
            output.append(f"**Author:** {author} | **Source:** {source}")
            output.append(f"\n{doc.content}\n")
            output.append("---\n")
        
        return "\n".join(output)
    
    print(markdown_format(results))
    
    
    # 7. JSON-like Format
    print("\n\n7. JSON-LIKE STRUCTURED FORMAT")
    print("-"*80)
    
    def json_like_format(results):
        """Format results in a JSON-like structure."""
        output = ["Documents: ["]
        
        for i, result in enumerate(results):
            doc = result.document
            
            output.append("  {")
            output.append(f'    "id": "{doc.id}",')
            output.append(f'    "title": "{doc.metadata.get("title", "N/A")}",')
            output.append(f'    "content": "{doc.content}",')
            output.append(f'    "relevance": {result.score:.3f}')
            
            if i < len(results) - 1:
                output.append("  },")
            else:
                output.append("  }")
        
        output.append("]")
        
        return "\n".join(output)
    
    print(json_like_format(results))
    
    
    # 8. Context for Different LLM Use Cases
    print("\n\n8. CONTEXT FOR DIFFERENT LLM USE CASES")
    print("-"*80)
    
    # Q&A format
    print("Q&A Format:")
    qa_context = f"""Answer the following question using the provided context.

Context:
{results_to_context(results)}

Question: {query}

Answer:"""
    print(qa_context)
    
    print("\n" + "-"*40)
    
    # Summarization format
    print("\nSummarization Format:")
    summary_context = f"""Summarize the key information from these documents:

{results_to_context(results)}

Summary:"""
    print(summary_context)
    
    print("\n" + "-"*40)
    
    # Comparison format
    print("\nComparison Format:")
    comparison_context = f"""Compare and contrast the information from these sources:

{results_to_context(results)}

Comparison:"""
    print(comparison_context)
    
    
    # 9. Token-Aware Formatting
    print("\n\n9. TOKEN-AWARE FORMATTING")
    print("-"*80)
    
    def token_aware_format(results, max_tokens=200):
        """Format results within a token budget."""
        output = []
        current_tokens = 0
        
        for result in results:
            content = result.document.content
            content_tokens = len(content) // 4
            
            if current_tokens + content_tokens > max_tokens:
                # Truncate to fit
                remaining = max_tokens - current_tokens
                chars_to_include = remaining * 4
                if chars_to_include > 20:  # Only include if meaningful
                    content = content[:chars_to_include] + "..."
                    output.append(f"- {content}")
                break
            else:
                output.append(f"- {content}")
                current_tokens += content_tokens
        
        return "\n".join(output), current_tokens
    
    for budget in [100, 200, 400]:
        formatted, used = token_aware_format(results, max_tokens=budget)
        print(f"\nBudget: {budget} tokens, Used: ~{used} tokens")
        print(formatted[:150] + "..." if len(formatted) > 150 else formatted)
    
    
    # 10. Production Formatting Pipeline
    print("\n\n10. PRODUCTION FORMATTING PIPELINE")
    print("-"*80)
    
    def format_for_llm(results, query_text, max_tokens=2000, include_citations=True):
        """Production-ready formatting for LLM context."""
        
        # Header
        output = [f"Retrieved information for: '{query_text}'\n"]
        
        # Format each result
        current_tokens = len(" ".join(output)) // 4
        
        for i, result in enumerate(results, 1):
            doc = result.document
            
            # Build result entry
            if include_citations:
                title = doc.metadata.get('title', f'Document {i}')
                entry = f"[{i}] {title}\n{doc.content}\n"
            else:
                entry = f"{doc.content}\n"
            
            entry_tokens = len(entry) // 4
            
            # Check if we have room
            if current_tokens + entry_tokens > max_tokens:
                # Try to fit a truncated version
                remaining = max_tokens - current_tokens
                chars = remaining * 4
                if chars > 50:
                    truncated = doc.content[:chars] + "..."
                    if include_citations:
                        title = doc.metadata.get('title', f'Document {i}')
                        entry = f"[{i}] {title}\n{truncated}\n"
                    else:
                        entry = f"{truncated}\n"
                    output.append(entry)
                break
            else:
                output.append(entry)
                current_tokens += entry_tokens
        
        # Add citation guide if needed
        if include_citations:
            output.append("\nWhen answering, cite sources using [number] format.")
        
        return "".join(output)
    
    print("Production format example:\n")
    prod_context = format_for_llm(results, query, max_tokens=500, include_citations=True)
    print(prod_context)
    print(f"\nEstimated tokens: ~{len(prod_context) // 4}")
    
    
    # 11. Best Practices
    print("\n\n11. FORMATTING BEST PRACTICES")
    print("-"*80)
    
    practices = [
        "Include document IDs or numbers for citation tracking",
        "Add metadata (source, date) for credibility",
        "Keep format consistent across all results",
        "Use clear separators between documents",
        "Monitor token usage to stay within limits",
        "Consider truncating content while preserving meaning",
        "Add instructions for the LLM when needed",
        "Use compact formats for large result sets",
        "Include relevance scores when helpful",
        "Test different formats for your specific use case"
    ]
    
    for i, practice in enumerate(practices, 1):
        print(f"  {i:2}. {practice}")
    
    
    print("\n" + "="*80)
    print("Result formatting demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    main()
