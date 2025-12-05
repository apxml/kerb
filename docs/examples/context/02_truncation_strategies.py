"""
Context Truncation Strategies Example
=====================================

This example demonstrates different strategies for truncating context windows
when they exceed token limits - a critical capability for LLM applications.

Main concepts:
- Understanding truncation strategies (FIRST, LAST, MIDDLE, PRIORITY)
- Applying truncation to fit within token budgets
- Comparing strategy outcomes
- Choosing appropriate strategies for different use cases
"""

from kerb.context import (
    ContextItem,
    ContextWindow,
    create_context_window,
    truncate_context_window,
    TruncationStrategy,
)
from kerb.tokenizer import count_tokens


def create_sample_conversation() -> list[ContextItem]:
    """Create a sample conversation for demonstration."""
    messages = [
        ("System", "You are a helpful coding assistant.", 1.0),
        ("User", "I need help with Python async/await.", 0.9),
        ("Assistant", "I'll help you understand Python async/await. It's used for concurrent programming...", 0.8),
        ("User", "Can you show an example with aiohttp?", 0.9),
        ("Assistant", "Here's an example using aiohttp for async HTTP requests: import aiohttp...", 0.7),
        ("User", "How do I handle errors in async code?", 1.0),
        ("Assistant", "Error handling in async code uses try/except blocks with async functions...", 0.6),
        ("User", "What about timeouts?", 0.9),
        ("Assistant", "You can use asyncio.wait_for() to add timeouts to async operations...", 0.5),
        ("User", "Show me a complete example with error handling and timeouts.", 1.0),
    ]
    
    items = []
    for i, (role, content, priority) in enumerate(messages):
        full_content = f"{role}: {content}"
        items.append(ContextItem(
            content=full_content,
            priority=priority,
            token_count=count_tokens(full_content),
            metadata={"role": role, "turn": i}
        ))
    
    return items


def main():
    """Run truncation strategies example."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("CONTEXT TRUNCATION STRATEGIES EXAMPLE")
    print("="*80)
    
    # Create a conversation that exceeds token limit
    conversation_items = create_sample_conversation()
    total_tokens = sum(item.token_count for item in conversation_items)
    
    print(f"\nOriginal conversation: {len(conversation_items)} messages")
    print(f"Total tokens: {total_tokens}")
    
    # Set a token limit that requires truncation
    max_tokens = 100
    print(f"\nToken limit for this example: {max_tokens}")
    print(f"Must reduce by ~{total_tokens - max_tokens} tokens")
    
    # Strategy 1: FIRST - Keep earliest messages
    print("\n1. TRUNCATION STRATEGY: FIRST (Keep earliest)")
    print("-"*80)
    print("Use case: Preserve system prompts and initial context")
    
    window_first = create_context_window(conversation_items, max_tokens=max_tokens)
    truncated_first = truncate_context_window(window_first, max_tokens, TruncationStrategy.FIRST)
    
    print(f"\nResult: {len(truncated_first.items)} items kept ({truncated_first.current_tokens} tokens)")
    print("Kept messages:")
    for item in truncated_first.items:
        role = item.metadata.get("role", "Unknown")
        preview = item.content[:60] + "..." if len(item.content) > 60 else item.content
        print(f"  {role}: {preview}")
    
    # Strategy 2: LAST - Keep most recent messages
    print("\n2. TRUNCATION STRATEGY: LAST (Keep most recent)")
    print("-"*80)
    print("Use case: Maintain recent conversation context")
    
    window_last = create_context_window(conversation_items, max_tokens=max_tokens)
    truncated_last = truncate_context_window(window_last, max_tokens, TruncationStrategy.LAST)
    
    print(f"\nResult: {len(truncated_last.items)} items kept ({truncated_last.current_tokens} tokens)")
    print("Kept messages:")
    for item in truncated_last.items:
        role = item.metadata.get("role", "Unknown")
        preview = item.content[:60] + "..." if len(item.content) > 60 else item.content
        print(f"  {role}: {preview}")
    
    # Strategy 3: MIDDLE - Keep beginning and end
    print("\n3. TRUNCATION STRATEGY: MIDDLE (Keep beginning and end)")
    print("-"*80)
    print("Use case: Preserve system context and recent messages")
    
    window_middle = create_context_window(conversation_items, max_tokens=max_tokens)
    truncated_middle = truncate_context_window(window_middle, max_tokens, TruncationStrategy.MIDDLE)
    
    print(f"\nResult: {len(truncated_middle.items)} items kept ({truncated_middle.current_tokens} tokens)")
    print("Kept messages:")
    for item in truncated_middle.items:
        role = item.metadata.get("role", "Unknown")
        preview = item.content[:60] + "..." if len(item.content) > 60 else item.content
        print(f"  {role}: {preview}")
    
    # Strategy 4: PRIORITY - Keep highest priority items
    print("\n4. TRUNCATION STRATEGY: PRIORITY (Keep highest priority)")
    print("-"*80)
    print("Use case: Keep most important context regardless of position")
    
    window_priority = create_context_window(conversation_items, max_tokens=max_tokens)
    truncated_priority = truncate_context_window(window_priority, max_tokens, TruncationStrategy.PRIORITY)
    
    print(f"\nResult: {len(truncated_priority.items)} items kept ({truncated_priority.current_tokens} tokens)")
    print("Kept messages (by priority):")
    for item in sorted(truncated_priority.items, key=lambda x: x.priority, reverse=True):
        role = item.metadata.get("role", "Unknown")
        preview = item.content[:60] + "..." if len(item.content) > 60 else item.content
        print(f"  Priority {item.priority:.1f} - {role}: {preview}")
    
    # Example 5: Comparing strategies for code review context
    print("\n5. CODE REVIEW SCENARIO")
    print("-"*80)
    
    code_review_items = [
        ContextItem(
            content="Task: Review pull request for authentication module",
            priority=1.0,
            token_count=count_tokens("Task: Review pull request for authentication module"),
            metadata={"type": "task"}
        ),
        ContextItem(
            content="Project context: Large e-commerce platform using microservices",
            priority=0.9,
            token_count=count_tokens("Project context: Large e-commerce platform using microservices"),
            metadata={"type": "context"}
        ),
        ContextItem(
            content="Security requirements: OWASP compliance, JWT tokens, rate limiting",
            priority=1.0,
            token_count=count_tokens("Security requirements: OWASP compliance, JWT tokens, rate limiting"),
            metadata={"type": "requirements"}
        ),
        ContextItem(
            content="Code changes summary: Modified login endpoint, added password validation",
            priority=0.8,
            token_count=count_tokens("Code changes summary: Modified login endpoint, added password validation"),
            metadata={"type": "changes"}
        ),
        ContextItem(
            content="Previous review comments: Consider using bcrypt, add rate limiting",
            priority=0.7,
            token_count=count_tokens("Previous review comments: Consider using bcrypt, add rate limiting"),
            metadata={"type": "history"}
        ),
        ContextItem(
            content="Test coverage: 85% for auth module, missing edge case tests",
            priority=0.9,
            token_count=count_tokens("Test coverage: 85% for auth module, missing edge case tests"),
            metadata={"type": "testing"}
        ),
    ]
    
    review_tokens = sum(item.token_count for item in code_review_items)
    review_limit = 50
    
    print(f"Code review context: {review_tokens} tokens, limit: {review_limit} tokens")
    print("\nWith PRIORITY strategy (recommended for code review):")
    
    review_window = create_context_window(code_review_items, max_tokens=review_limit)
    truncated_review = truncate_context_window(review_window, review_limit, TruncationStrategy.PRIORITY)
    
    print(f"Kept {len(truncated_review.items)} most important items:")
    for item in sorted(truncated_review.items, key=lambda x: x.priority, reverse=True):
        ctx_type = item.metadata.get("type", "unknown")
        print(f"  [{ctx_type}] Priority {item.priority:.1f}: {item.content[:70]}...")
    
    # Example 6: Strategy recommendations
    print("\n6. STRATEGY SELECTION GUIDE")
    print("-"*80)
    
    recommendations = [
        ("FIRST", "System prompts, initial instructions", "Chatbots, assistants"),
        ("LAST", "Recent conversation, latest updates", "Continuous conversations"),
        ("MIDDLE", "System + recent context", "Long conversations with setup"),
        ("PRIORITY", "Critical information only", "Complex multi-source context"),
    ]
    
    print("\nStrategy | Best For | Common Use Cases")
    print("-" * 80)
    for strategy, best_for, use_cases in recommendations:
        print(f"{strategy:10} | {best_for:30} | {use_cases}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
