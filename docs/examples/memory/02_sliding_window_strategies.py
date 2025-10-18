"""Sliding Window Strategies Example

This example demonstrates different window strategies for managing conversation context
in LLM applications with token limits.

Main concepts:
- Sliding window for recent messages
- Token-limited window for API constraints
- Alternating user/assistant pairs
- Custom window strategies
"""

from kerb.memory import (
    create_sliding_window,
    create_token_limited_window,
    create_alternating_window,
    ConversationBuffer
)
from kerb.core.types import Message


def simple_token_counter(text: str) -> int:
    """Simple token estimation: ~1 token per 0.75 words."""
    return int(len(text.split()) / 0.75)


def main():
    """Run sliding window strategies example."""
    
    print("="*80)
    print("SLIDING WINDOW STRATEGIES EXAMPLE")
    print("="*80)
    
    # Create sample conversation
    messages = [
        Message("system", "You are a helpful AI assistant for code reviews."),
        Message("user", "I need help reviewing my Python code for performance issues."),
        Message("assistant", "I'd be happy to help! Please share the code you'd like me to review."),
        Message("user", "Here's a function that processes a large dataset using nested loops."),
        Message("assistant", "Nested loops can be inefficient. Consider using list comprehensions or numpy for better performance."),
        Message("user", "What about using pandas for data processing?"),
        Message("assistant", "Pandas is excellent for tabular data. It uses vectorized operations which are much faster than Python loops."),
        Message("user", "Can you show me an example of vectorization?"),
        Message("assistant", "Sure! Instead of 'for i in range(len(df)): df.loc[i, 'new'] = df.loc[i, 'a'] + df.loc[i, 'b']', use 'df['new'] = df['a'] + df['b']'."),
        Message("user", "That's much cleaner! What about memory optimization?"),
        Message("assistant", "For memory optimization, use appropriate dtypes (e.g., int32 instead of int64), process data in chunks, and use generators for large datasets."),
        Message("user", "How do I profile memory usage in Python?"),
        Message("assistant", "Use the 'memory_profiler' package. Decorate functions with @profile and run 'python -m memory_profiler script.py'."),
    ]
    
    print(f"\nTotal messages in conversation: {len(messages)}")
    
    # Strategy 1: Simple sliding window
    print("\n" + "-"*80)
    print("STRATEGY 1: SIMPLE SLIDING WINDOW")
    print("-"*80)
    
    window = create_sliding_window(messages, window_size=5, include_system=False)
    print(f"\nLast 5 messages (excluding system):")
    for msg in window:
        print(f"  {msg.role}: {msg.content[:70]}...")
    
    # Strategy 2: Token-limited window
    print("\n" + "-"*80)
    print("STRATEGY 2: TOKEN-LIMITED WINDOW")
    print("-"*80)
    
    # For API constraints (e.g., 4K context limit, reserve 2K for completion)
    max_tokens = 300  # Reduced for demo
    token_window = create_token_limited_window(
        messages,
        max_tokens=max_tokens,
        token_estimator=simple_token_counter
    )
    
    total_tokens = sum(simple_token_counter(m.content) for m in token_window)
    print(f"\nMessages fitting in {max_tokens} tokens: {len(token_window)}")
    print(f"Actual token count: {total_tokens}")
    print("\nIncluded messages:")
    for i, msg in enumerate(token_window, 1):
        tokens = simple_token_counter(msg.content)
        print(f"  [{i}] {msg.role} ({tokens} tokens): {msg.content[:50]}...")
    
    # Strategy 3: Alternating pairs
    print("\n" + "-"*80)
    print("STRATEGY 3: ALTERNATING USER/ASSISTANT PAIRS")
    print("-"*80)
    
    pairs_window = create_alternating_window(messages, pairs=3)
    print(f"\nLast 3 user/assistant pairs:")
    for i, msg in enumerate(pairs_window, 1):
        print(f"  [{i}] {msg.role}: {msg.content[:70]}...")
    
    # Strategy 4: Custom window with ConversationBuffer
    print("\n" + "-"*80)
    print("STRATEGY 4: CUSTOM WINDOW WITH BUFFER")
    print("-"*80)
    
    buffer = ConversationBuffer(window_size=6)
    for msg in messages:
        buffer.add_message(msg.role, msg.content)
    
    # Get context with token limit
    context_with_limit = buffer.get_context(max_tokens=250, include_summary=False)
    print(f"\nContext formatted for LLM (max 250 tokens):")
    print(f"{context_with_limit[:300]}...")
    
    # Strategy 5: Hybrid approach - system + recent pairs
    print("\n" + "-"*80)
    print("STRATEGY 5: HYBRID (SYSTEM + RECENT PAIRS)")
    print("-"*80)
    
    # Keep system message + recent pairs for context
    system_messages = [m for m in messages if m.role == "system"]
    recent_pairs = create_alternating_window(messages, pairs=2)
    hybrid_window = system_messages + recent_pairs
    
    print(f"\nHybrid window ({len(hybrid_window)} messages):")
    for msg in hybrid_window:
        print(f"  {msg.role}: {msg.content[:70]}...")
    
    # Strategy 6: Priority-based window
    print("\n" + "-"*80)
    print("STRATEGY 6: PRIORITY-BASED WINDOW")
    print("-"*80)
    
    # Prioritize: system messages, questions (containing '?'), recent messages
    priority_window = []
    
    # Always include system messages
    priority_window.extend([m for m in messages if m.role == "system"])
    
    # Add messages with questions
    questions = [m for m in messages if "?" in m.content and m not in priority_window]
    priority_window.extend(questions[:3])  # Top 3 questions
    
    # Fill remaining with recent messages
    recent = [m for m in messages[-5:] if m not in priority_window]
    priority_window.extend(recent)
    
    print(f"\nPriority window ({len(priority_window)} messages):")
    print("  [System messages + Important questions + Recent context]")
    for i, msg in enumerate(priority_window, 1):
        marker = "[Q]" if "?" in msg.content else ""
        print(f"  [{i}] {msg.role} {marker}: {msg.content[:60]}...")
    
    # Real-world scenario: Preparing context for LLM API call
    print("\n" + "-"*80)
    print("REAL-WORLD: PREPARING CONTEXT FOR LLM API")
    print("-"*80)
    
    # Scenario: GPT-4 with 8K context, reserve 2K for completion
    available_tokens = 6000
    
    # Method 1: Use token-limited window
    api_context = create_token_limited_window(
        messages,
        max_tokens=available_tokens,
        token_estimator=simple_token_counter
    )
    
    estimated_tokens = sum(simple_token_counter(m.content) for m in api_context)
    print(f"\nPreparing context for API call:")
    print(f"  Available tokens: {available_tokens}")
    print(f"  Messages included: {len(api_context)}")
    print(f"  Estimated tokens: {estimated_tokens}")
    print(f"  Reserved for completion: {available_tokens - estimated_tokens}")
    
    # Format for API
    formatted_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in api_context
    ]
    print(f"\nFormatted {len(formatted_messages)} messages for API call")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
