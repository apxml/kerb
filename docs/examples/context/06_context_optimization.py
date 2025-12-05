"""
Context Optimization Example
============================

This example demonstrates optimization techniques for improving
context quality and relevance in LLM applications.

Main concepts:
- Deduplication to remove redundant information
- Reordering for logical flow and coherence
- Merging multiple context windows
- Query-specific context optimization
- Context quality improvement strategies
"""

from kerb.context import (
    ContextItem,
    ContextWindow,
    create_context_window,
    deduplicate_context,
    reorder_context,
    merge_context_windows,
    optimize_context_for_query,
)
from kerb.tokenizer import count_tokens


def main():
    """Run context optimization example."""
    
    print("="*80)
    print("CONTEXT OPTIMIZATION EXAMPLE")
    print("="*80)
    
    # Example 1: Deduplication
    print("\n1. CONTEXT DEDUPLICATION")
    print("-"*80)
    print("Use case: Remove redundant information from context")
    
    # Create context with duplicates and near-duplicates
    duplicate_items = [
        ContextItem(
            content="Python is a high-level programming language.",
            token_count=count_tokens("Python is a high-level programming language."),
            metadata={"source": "doc1"}
        ),
        ContextItem(
            content="Python is used for web development and data science.",
            token_count=count_tokens("Python is used for web development and data science."),
            metadata={"source": "doc2"}
        ),
        ContextItem(
            content="Python is a high-level programming language.",
            token_count=count_tokens("Python is a high-level programming language."),
            metadata={"source": "doc3"}
        ),
        ContextItem(
            content="Machine learning frameworks in Python include TensorFlow.",
            token_count=count_tokens("Machine learning frameworks in Python include TensorFlow."),
            metadata={"source": "doc4"}
        ),
        ContextItem(
            content="Python is a high level programming language.",  # Slight variation
            token_count=count_tokens("Python is a high level programming language."),
            metadata={"source": "doc5"}
        ),
    ]
    
    dup_window = create_context_window(duplicate_items)
    print(f"\nOriginal context: {len(dup_window.items)} items")
    for i, item in enumerate(dup_window.items, 1):
        source = item.metadata.get("source", "unknown")
        print(f"  {i}. [{source}] {item.content}")
    
    # Deduplicate
    dedup_items = deduplicate_context(dup_window.items, similarity_threshold=0.9)
    dedup_window = create_context_window(dedup_items)
    
    print(f"\nAfter deduplication: {len(dedup_window.items)} items")
    for i, item in enumerate(dedup_window.items, 1):
        source = item.metadata.get("source", "unknown")
        print(f"  {i}. [{source}] {item.content}")
    
    savings = dup_window.current_tokens - dedup_window.current_tokens
    print(f"\nToken savings: {savings} tokens ({savings/dup_window.current_tokens:.1%})")
    
    # Example 2: Context reordering
    print("\n2. CONTEXT REORDERING")
    print("-"*80)
    print("Use case: Arrange context for optimal LLM understanding")
    
    # Create unordered context
    unordered = [
        ContextItem(
            content="Step 3: Test the implementation",
            priority=0.7,
            token_count=count_tokens("Step 3: Test the implementation"),
            metadata={"order": 3, "type": "instruction"}
        ),
        ContextItem(
            content="Step 1: Set up the environment",
            priority=0.9,
            token_count=count_tokens("Step 1: Set up the environment"),
            metadata={"order": 1, "type": "instruction"}
        ),
        ContextItem(
            content="Background: Project uses Flask framework",
            priority=1.0,
            token_count=count_tokens("Background: Project uses Flask framework"),
            metadata={"order": 0, "type": "context"}
        ),
        ContextItem(
            content="Step 2: Write the code",
            priority=0.8,
            token_count=count_tokens("Step 2: Write the code"),
            metadata={"order": 2, "type": "instruction"}
        ),
    ]
    
    unordered_window = create_context_window(unordered)
    print("\nOriginal order:")
    for item in unordered_window.items:
        print(f"  - {item.content}")
    
    # Reorder by priority
    priority_ordered = reorder_context(unordered_window.items, strategy="priority")
    print("\nOrdered by priority (highest first):")
    for item in priority_ordered:
        print(f"  - Priority {item.priority:.1f}: {item.content}")
    
    # Reorder by chronological order
    chrono_ordered = reorder_context(unordered_window.items, strategy="chronological")
    print("\nOrdered chronologically:")
    for item in chrono_ordered:
        print(f"  - {item.content}")
    
    # Example 3: Merging context windows
    print("\n3. MERGING CONTEXT WINDOWS")
    print("-"*80)
    print("Use case: Combine context from multiple sources")
    
    # Create separate context windows
    system_context = create_context_window([
        ContextItem(
            content="System: You are a helpful coding assistant",
            priority=1.0,
            token_count=count_tokens("System: You are a helpful coding assistant"),
            metadata={"source": "system"}
        ),
    ])
    
    user_history = create_context_window([
        ContextItem(
            content="User prefers Python 3.10+ features",
            priority=0.8,
            token_count=count_tokens("User prefers Python 3.10+ features"),
            metadata={"source": "preferences"}
        ),
        ContextItem(
            content="User's previous project used FastAPI",
            priority=0.7,
            token_count=count_tokens("User's previous project used FastAPI"),
            metadata={"source": "history"}
        ),
    ])
    
    current_task = create_context_window([
        ContextItem(
            content="Current task: Implement REST API endpoint",
            priority=0.9,
            token_count=count_tokens("Current task: Implement REST API endpoint"),
            metadata={"source": "task"}
        ),
    ])
    
    print("Separate contexts:")
    print(f"  System context: {len(system_context.items)} items")
    print(f"  User history: {len(user_history.items)} items")
    print(f"  Current task: {len(current_task.items)} items")
    
    # Merge all contexts
    merged = merge_context_windows(
        [system_context, user_history, current_task],
        max_tokens=200
    )
    
    print(f"\nMerged context: {len(merged.items)} items, {merged.current_tokens} tokens")
    for item in merged.items:
        source = item.metadata.get("source", "unknown")
        print(f"  [{source}] {item.content}")
    
    # Example 4: Query-specific optimization
    print("\n4. QUERY-SPECIFIC OPTIMIZATION")
    print("-"*80)
    print("Use case: Optimize context for a specific query")
    
    knowledge_items = [
        ContextItem(
            content="FastAPI is a modern Python web framework for building APIs",
            token_count=count_tokens("FastAPI is a modern Python web framework for building APIs"),
            metadata={"topic": "fastapi"}
        ),
        ContextItem(
            content="Django is a full-featured web framework with ORM",
            token_count=count_tokens("Django is a full-featured web framework with ORM"),
            metadata={"topic": "django"}
        ),
        ContextItem(
            content="Flask is a lightweight web framework for Python",
            token_count=count_tokens("Flask is a lightweight web framework for Python"),
            metadata={"topic": "flask"}
        ),
        ContextItem(
            content="FastAPI provides automatic API documentation with Swagger",
            token_count=count_tokens("FastAPI provides automatic API documentation with Swagger"),
            metadata={"topic": "fastapi"}
        ),
        ContextItem(
            content="Pydantic models enable data validation in FastAPI",
            token_count=count_tokens("Pydantic models enable data validation in FastAPI"),
            metadata={"topic": "fastapi"}
        ),
        ContextItem(
            content="SQLAlchemy is an ORM for database operations",
            token_count=count_tokens("SQLAlchemy is an ORM for database operations"),
            metadata={"topic": "database"}
        ),
    ]
    
    knowledge_window = create_context_window(knowledge_items)
    query = "How do I create a REST API with FastAPI?"
    
    print(f"\nQuery: {query}")
    print(f"Available context: {len(knowledge_window.items)} items")
    
    # Optimize for query
    optimized = optimize_context_for_query(
        knowledge_window,
        query,
        max_tokens=100
    )
    
    print(f"\nOptimized context: {len(optimized.items)} items selected")
    for item in optimized.items:
        topic = item.metadata.get("topic", "unknown")
        print(f"  [{topic}] {item.content}")
    
    # Example 5: Multi-stage optimization
    print("\n5. MULTI-STAGE OPTIMIZATION")
    print("-"*80)
    print("Use case: Apply multiple optimization techniques in sequence")
    
    # Create noisy context with duplicates and low-relevance items
    noisy_context = [
        ContextItem(content="Python web development best practices", token_count=10, priority=0.8),
        ContextItem(content="Python web development best practices", token_count=10, priority=0.8),
        ContextItem(content="JavaScript frontend frameworks overview", token_count=10, priority=0.5),
        ContextItem(content="Python API development with FastAPI", token_count=10, priority=0.9),
        ContextItem(content="Database optimization techniques", token_count=10, priority=0.6),
        ContextItem(content="Python web development best practices", token_count=10, priority=0.8),
        ContextItem(content="FastAPI dependency injection system", token_count=10, priority=0.9),
        ContextItem(content="CSS styling frameworks comparison", token_count=10, priority=0.4),
        ContextItem(content="Python async programming patterns", token_count=10, priority=0.7),
        ContextItem(content="FastAPI testing strategies", token_count=10, priority=0.9),
    ]
    
    noisy_window = create_context_window(noisy_context)
    query = "FastAPI development guide"
    
    print(f"Starting context: {len(noisy_window.items)} items, {noisy_window.current_tokens} tokens")
    
    # Stage 1: Deduplicate
    stage1_items = deduplicate_context(noisy_window.items, similarity_threshold=0.95)
    stage1 = create_context_window(stage1_items)
    print(f"\nStage 1 (Deduplication): {len(stage1.items)} items, {stage1.current_tokens} tokens")
    
    # Stage 2: Optimize for query
    stage2 = optimize_context_for_query(stage1, query, max_tokens=60)
    print(f"Stage 2 (Query optimization): {len(stage2.items)} items, {stage2.current_tokens} tokens")
    
    # Stage 3: Reorder by priority
    stage3_items = reorder_context(stage2.items, strategy="priority")
    stage3 = create_context_window(stage3_items)
    print(f"Stage 3 (Priority reordering): {len(stage3.items)} items")
    
    print("\nFinal optimized context:")
    for i, item in enumerate(stage3.items, 1):
        print(f"  {i}. Priority {item.priority:.1f}: {item.content}")
    
    # Example 6: Context quality metrics
    print("\n6. CONTEXT QUALITY METRICS")
    print("-"*80)
    
    def calculate_quality_metrics(window: ContextWindow, query: str = "") -> dict:
        """Calculate quality metrics for context window."""

# %%
# Setup and Imports
# -----------------
        metrics = {
            "total_items": len(window.items),
            "total_tokens": window.current_tokens,
            "avg_priority": sum(item.priority for item in window.items) / len(window.items) if window.items else 0,
            "high_priority_ratio": sum(1 for item in window.items if item.priority >= 0.8) / len(window.items) if window.items else 0,
        }
        
        # Check for duplicates
        unique_content = set(item.content for item in window.items)
        metrics["duplication_ratio"] = 1 - (len(unique_content) / len(window.items)) if window.items else 0
        
        return metrics
    
    # Compare before and after optimization
    before_metrics = calculate_quality_metrics(noisy_window, query)
    after_metrics = calculate_quality_metrics(stage3, query)
    
    print("\nQuality Metrics Comparison:")
    print("\n  Metric                | Before | After  | Change")
    print("  " + "-"*60)
    
    for metric in before_metrics:
        before = before_metrics[metric]
        after = after_metrics[metric]
        
        if isinstance(before, float):
            change = after - before
            print(f"  {metric:20} | {before:6.2f} | {after:6.2f} | {change:+6.2f}")
        else:
            change = after - before
            print(f"  {metric:20} | {before:6} | {after:6} | {change:+6}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
