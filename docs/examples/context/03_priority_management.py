"""
Priority-Based Context Management Example
=========================================

This example demonstrates how to use priorities to intelligently manage
context when dealing with token limits - essential for LLM applications.

Main concepts:
- Assigning priorities to context items
- Priority-based context selection
- Using priority functions (recency, relevance, diversity)
- Combining multiple priority criteria
- Dynamic priority adjustment
"""

from kerb.context import (
    ContextItem,
    ContextWindow,
    create_context_window,
    truncate_context_window,
    TruncationStrategy,
    assign_priorities,
    priority_by_recency,
    priority_by_relevance,
    priority_by_diversity,
)
from kerb.tokenizer import count_tokens
import time


def main():
    """Run priority management example."""
    
    print("="*80)
    print("PRIORITY-BASED CONTEXT MANAGEMENT EXAMPLE")
    print("="*80)
    
    # Example 1: Basic priority assignment
    print("\n1. BASIC PRIORITY ASSIGNMENT")
    print("-"*80)
    
    items = [
        ContextItem(content="System: You are a Python expert.", priority=1.0, token_count=10),
        ContextItem(content="User: How do I use decorators?", priority=0.9, token_count=10),
        ContextItem(content="Assistant: Decorators are functions...", priority=0.7, token_count=10),
        ContextItem(content="User: What about class decorators?", priority=0.9, token_count=10),
        ContextItem(content="Assistant: Class decorators work similarly...", priority=0.6, token_count=10),
    ]
    
    window = create_context_window(items, max_tokens=30)
    print(f"Created window with {len(window.items)} items ({window.current_tokens} tokens)")
    print(f"Token limit: 30 tokens")
    
    # Truncate using priority
    truncated = truncate_context_window(window, 30, TruncationStrategy.PRIORITY)
    print(f"\nAfter priority-based truncation: {len(truncated.items)} items kept")
    for item in sorted(truncated.items, key=lambda x: x.priority, reverse=True):
        print(f"  Priority {item.priority:.1f}: {item.content[:50]}...")
    
    # Example 2: Priority by recency
    print("\n2. PRIORITY BY RECENCY")
    print("-"*80)
    print("Use case: Favor recent interactions in conversation")
    
    # Create items with timestamps
    current_time = time.time()
    conversation = []
    
    for i, msg in enumerate([
        "User: Tell me about Python",
        "Assistant: Python is a high-level language...",
        "User: What about async programming?",
        "Assistant: Async programming in Python...",
        "User: Show me an example",
        "Assistant: Here's an async example...",
        "User: How do I handle errors?",
    ]):
        item = ContextItem(
            content=msg,
            priority=0.5,  # Base priority
            token_count=count_tokens(msg),
            timestamp=current_time - (len(conversation) - i) * 60  # Minutes ago
        )
        conversation.append(item)
    
    print(f"Created {len(conversation)} messages with timestamps")
    
    # Apply recency-based priorities
    recency_items = priority_by_recency(conversation)
    
    print("\nPriorities after recency weighting:")
    for item in recency_items:
        minutes_ago = (current_time - item.timestamp) / 60
        print(f"  {minutes_ago:.0f}m ago - Priority {item.priority:.2f}: {item.content[:50]}...")
    
    # Example 3: Priority by relevance to query
    print("\n3. PRIORITY BY RELEVANCE TO QUERY")
    print("-"*80)
    print("Use case: Retrieve most relevant context for a specific query")
    
    knowledge_base = [
        ContextItem(
            content="Python decorators modify function behavior without changing code",
            token_count=count_tokens("Python decorators modify function behavior without changing code"),
            metadata={"topic": "decorators"}
        ),
        ContextItem(
            content="List comprehensions provide concise way to create lists in Python",
            token_count=count_tokens("List comprehensions provide concise way to create lists in Python"),
            metadata={"topic": "comprehensions"}
        ),
        ContextItem(
            content="Async/await enables concurrent programming in Python",
            token_count=count_tokens("Async/await enables concurrent programming in Python"),
            metadata={"topic": "async"}
        ),
        ContextItem(
            content="Generator functions use yield to produce values lazily",
            token_count=count_tokens("Generator functions use yield to produce values lazily"),
            metadata={"topic": "generators"}
        ),
        ContextItem(
            content="Context managers handle resource setup and cleanup",
            token_count=count_tokens("Context managers handle resource setup and cleanup"),
            metadata={"topic": "context_managers"}
        ),
    ]
    
    query = "How do I use decorators in Python?"
    print(f"\nQuery: {query}")
    
    # Apply relevance-based priorities
    relevant_items = priority_by_relevance(knowledge_base, query)
    
    print("\nItems ranked by relevance:")
    for item in sorted(relevant_items, key=lambda x: x.priority, reverse=True):
        topic = item.metadata.get("topic", "unknown")
        print(f"  Priority {item.priority:.2f} [{topic}]: {item.content}")
    
    # Example 4: Priority by diversity
    print("\n4. PRIORITY BY DIVERSITY")
    print("-"*80)
    print("Use case: Ensure diverse information in context")
    
    documents = [
        ContextItem(
            content="Python is great for web development with Django and Flask",
            token_count=count_tokens("Python is great for web development with Django and Flask"),
            metadata={"category": "web"}
        ),
        ContextItem(
            content="Python web frameworks include FastAPI for modern APIs",
            token_count=count_tokens("Python web frameworks include FastAPI for modern APIs"),
            metadata={"category": "web"}
        ),
        ContextItem(
            content="NumPy and Pandas excel at data analysis in Python",
            token_count=count_tokens("NumPy and Pandas excel at data analysis in Python"),
            metadata={"category": "data"}
        ),
        ContextItem(
            content="TensorFlow and PyTorch enable machine learning in Python",
            token_count=count_tokens("TensorFlow and PyTorch enable machine learning in Python"),
            metadata={"category": "ml"}
        ),
        ContextItem(
            content="Python automation scripts simplify DevOps tasks",
            token_count=count_tokens("Python automation scripts simplify DevOps tasks"),
            metadata={"category": "devops"}
        ),
        ContextItem(
            content="Flask is a lightweight web framework for Python",
            token_count=count_tokens("Flask is a lightweight web framework for Python"),
            metadata={"category": "web"}
        ),
    ]
    
    print(f"Documents by category:")
    for doc in documents:
        cat = doc.metadata.get("category", "unknown")
        print(f"  [{cat}] {doc.content}")
    
    # Apply diversity-based priorities
    diverse_items = priority_by_diversity(documents)
    
    print("\nAfter diversity prioritization:")
    for item in sorted(diverse_items, key=lambda x: x.priority, reverse=True):
        cat = item.metadata.get("category", "unknown")
        print(f"  Priority {item.priority:.2f} [{cat}]: {item.content}")
    
    # Example 5: Custom priority function
    print("\n5. CUSTOM PRIORITY FUNCTION")
    print("-"*80)
    print("Use case: Domain-specific priority calculation")
    
    def code_review_priority(item: ContextItem) -> float:
        """Custom priority for code review context."""

# %%
# Setup and Imports
# -----------------
        base_priority = 0.5
        
        # Boost priority based on item type
        item_type = item.metadata.get("type", "")
        if item_type == "security":
            base_priority += 0.5
        elif item_type == "bug":
            base_priority += 0.4
        elif item_type == "performance":
            base_priority += 0.3
        elif item_type == "style":
            base_priority += 0.1
        
        # Boost if marked as critical
        if item.metadata.get("critical", False):
            base_priority += 0.3
        
        return min(base_priority, 1.0)
    
    review_items = [
        ContextItem(
            content="SQL injection vulnerability in user input handling",
            token_count=20,
            metadata={"type": "security", "critical": True}
        ),
        ContextItem(
            content="Missing error handling in API endpoint",
            token_count=20,
            metadata={"type": "bug", "critical": False}
        ),
        ContextItem(
            content="Database query could be optimized with index",
            token_count=20,
            metadata={"type": "performance", "critical": False}
        ),
        ContextItem(
            content="Variable naming doesn't follow PEP 8 conventions",
            token_count=20,
            metadata={"type": "style", "critical": False}
        ),
        ContextItem(
            content="Race condition in concurrent user session handling",
            token_count=20,
            metadata={"type": "bug", "critical": True}
        ),
    ]
    
    # Apply custom priority function
    prioritized_review = assign_priorities(review_items, code_review_priority)
    
    print("Code review items by priority:")
    for item in sorted(prioritized_review, key=lambda x: x.priority, reverse=True):
        item_type = item.metadata.get("type", "unknown")
        critical = " [CRITICAL]" if item.metadata.get("critical") else ""
        print(f"  Priority {item.priority:.2f} [{item_type}{critical}]: {item.content}")
    
    # Example 6: Combining multiple priority signals
    print("\n6. COMBINING PRIORITY SIGNALS")
    print("-"*80)
    print("Use case: Multi-factor priority calculation for RAG")
    
    rag_documents = [
        ContextItem(
            content="Authentication implementation guide with code examples",
            token_count=count_tokens("Authentication implementation guide with code examples"),
            timestamp=current_time - 100,
            metadata={"source_quality": 0.9, "user_rating": 4.5}
        ),
        ContextItem(
            content="Quick tips for Python authentication",
            token_count=count_tokens("Quick tips for Python authentication"),
            timestamp=current_time - 50,
            metadata={"source_quality": 0.6, "user_rating": 3.8}
        ),
        ContextItem(
            content="Comprehensive security best practices for auth systems",
            token_count=count_tokens("Comprehensive security best practices for auth systems"),
            timestamp=current_time - 200,
            metadata={"source_quality": 1.0, "user_rating": 5.0}
        ),
        ContextItem(
            content="Database design for user management",
            token_count=count_tokens("Database design for user management"),
            timestamp=current_time - 150,
            metadata={"source_quality": 0.8, "user_rating": 4.2}
        ),
    ]
    
    query = "How to implement secure authentication?"
    print(f"Query: {query}")
    
    # Apply relevance
    docs_with_relevance = priority_by_relevance(rag_documents, query)
    
    # Apply recency
    docs_with_recency = priority_by_recency(docs_with_relevance)
    
    # Combine with source quality
    for doc in docs_with_recency:
        quality = doc.metadata.get("source_quality", 0.5)
        rating = doc.metadata.get("user_rating", 3.0) / 5.0
        # Weighted combination: 50% relevance+recency, 30% quality, 20% rating
        doc.priority = doc.priority * 0.5 + quality * 0.3 + rating * 0.2
    
    print("\nFinal priorities (relevance + recency + quality + rating):")
    for doc in sorted(docs_with_recency, key=lambda x: x.priority, reverse=True):
        quality = doc.metadata.get("source_quality", 0)
        rating = doc.metadata.get("user_rating", 0)
        print(f"  Priority {doc.priority:.2f} (Q:{quality:.1f}, R:{rating:.1f}): {doc.content}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
