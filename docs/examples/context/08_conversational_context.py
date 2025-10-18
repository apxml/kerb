"""Conversational Context Management Example

This example demonstrates managing context in multi-turn conversations
with LLMs while respecting token limits.

Main concepts:
- Managing conversation history
- Sliding window for recent messages
- System prompt preservation
- Context rotation strategies
- Memory-efficient long conversations
"""

from kerb.context import (
    ContextItem,
    ContextWindow,
    create_context_window,
    truncate_context_window,
    TruncationStrategy,
    create_sliding_window,
)
from kerb.tokenizer import count_tokens
import time


def main():
    """Run conversational context management example."""
    
    print("="*80)
    print("CONVERSATIONAL CONTEXT MANAGEMENT EXAMPLE")
    print("="*80)
    
    # Example 1: Basic conversation context
    print("\n1. BASIC CONVERSATION CONTEXT")
    print("-"*80)
    print("Use case: Simple chat history management")
    
    # Create conversation history
    conversation = []
    
    def add_message(role: str, content: str, priority: float = 1.0):
        """Add message to conversation."""
        msg = ContextItem(
            content=f"{role}: {content}",
            token_count=count_tokens(f"{role}: {content}"),
            priority=priority,
            timestamp=time.time(),
            metadata={"role": role, "turn": len(conversation) // 2}
        )
        conversation.append(msg)
        return msg
    
    # Simulate conversation
    add_message("System", "You are a helpful coding assistant", priority=1.0)
    add_message("User", "How do I read a file in Python?")
    add_message("Assistant", "You can use the open() function: with open('file.txt', 'r') as f: content = f.read()")
    add_message("User", "What about writing to a file?")
    add_message("Assistant", "Use 'w' mode: with open('file.txt', 'w') as f: f.write('content')")
    
    conv_window = create_context_window(conversation, max_tokens=200)
    
    print(f"Conversation history: {len(conversation)} messages, {conv_window.current_tokens} tokens")
    print("\nMessages:")
    for item in conversation:
        role = item.metadata.get("role", "Unknown")
        content = item.content.split(": ", 1)[1] if ": " in item.content else item.content
        preview = content[:60] + "..." if len(content) > 60 else content
        print(f"  [{role}] {preview}")
    
    # Example 2: Sliding window for long conversations
    print("\n2. SLIDING WINDOW FOR LONG CONVERSATIONS")
    print("-"*80)
    print("Use case: Keep only recent messages in context")
    
    # Create longer conversation
    long_conversation = []
    
    # System message (always keep)
    system_msg = ContextItem(
        content="System: You are a Python expert assistant",
        token_count=count_tokens("System: You are a Python expert assistant"),
        priority=1.0,
        metadata={"role": "system", "always_keep": True}
    )
    long_conversation.append(system_msg)
    
    # Add 10 exchanges
    topics = [
        ("lists", "How do I sort a list?", "Use the sorted() function or list.sort() method"),
        ("dicts", "How do I iterate over a dictionary?", "Use items(): for key, value in dict.items()"),
        ("functions", "How do I define a function?", "Use def keyword: def function_name(params):"),
        ("classes", "How do I create a class?", "Use class keyword: class ClassName:"),
        ("exceptions", "How do I handle errors?", "Use try/except blocks"),
        ("modules", "How do I import a module?", "Use import statement: import module_name"),
        ("async", "What is async/await?", "Async/await enables concurrent programming"),
        ("decorators", "How do I use decorators?", "Decorators modify function behavior: @decorator"),
        ("generators", "What are generators?", "Generators yield values lazily using yield"),
        ("comprehensions", "What are list comprehensions?", "Concise syntax: [x for x in iterable]"),
    ]
    
    for i, (topic, question, answer) in enumerate(topics):
        long_conversation.append(ContextItem(
            content=f"User: {question}",
            token_count=count_tokens(f"User: {question}"),
            priority=0.8,
            timestamp=time.time() - (len(topics) - i) * 10,
            metadata={"role": "user", "topic": topic, "turn": i}
        ))
        long_conversation.append(ContextItem(
            content=f"Assistant: {answer}",
            token_count=count_tokens(f"Assistant: {answer}"),
            priority=0.7,
            timestamp=time.time() - (len(topics) - i) * 10 + 1,
            metadata={"role": "assistant", "topic": topic, "turn": i}
        ))
    
    total_tokens = sum(item.token_count for item in long_conversation)
    print(f"\nFull conversation: {len(long_conversation)} messages, {total_tokens} tokens")
    
    # Create sliding window keeping last 5 exchanges (10 messages) + system
    recent_window_size = 11  # 1 system + 10 messages (5 exchanges)
    recent_messages = [long_conversation[0]] + long_conversation[-10:]  # System + last 10
    
    recent_window = create_context_window(recent_messages)
    print(f"Recent window: {len(recent_window.items)} messages, {recent_window.current_tokens} tokens")
    print("\nIncluded topics:")
    topics_included = set()
    for item in recent_window.items[1:]:  # Skip system message
        topic = item.metadata.get("topic", "unknown")
        if topic != "unknown":
            topics_included.add(topic)
    print(f"  {', '.join(sorted(topics_included))}")
    
    # Example 3: System prompt preservation
    print("\n3. SYSTEM PROMPT PRESERVATION")
    print("-"*80)
    print("Use case: Always keep system prompt regardless of truncation")
    
    # Create conversation exceeding limit
    messages_with_system = [
        ContextItem(
            content="System: You are an expert in web security",
            token_count=count_tokens("System: You are an expert in web security"),
            priority=1.0,
            metadata={"role": "system", "always_keep": True}
        ),
    ]
    
    # Add many user messages
    for i in range(8):
        messages_with_system.append(ContextItem(
            content=f"User: Question {i+1} about security best practices",
            token_count=count_tokens(f"User: Question {i+1} about security best practices"),
            priority=0.8,
            metadata={"role": "user"}
        ))
        messages_with_system.append(ContextItem(
            content=f"Assistant: Answer {i+1} with security recommendations",
            token_count=count_tokens(f"Assistant: Answer {i+1} with security recommendations"),
            priority=0.7,
            metadata={"role": "assistant"}
        ))
    
    full_window = create_context_window(messages_with_system, max_tokens=100)
    
    print(f"\nBefore truncation: {len(full_window.items)} messages, {full_window.current_tokens} tokens")
    
    # Separate system prompt
    system_prompt = [item for item in full_window.items if item.metadata.get("always_keep")]
    other_messages = [item for item in full_window.items if not item.metadata.get("always_keep")]
    
    # Truncate only non-system messages
    if system_prompt and other_messages:
        other_window = create_context_window(other_messages)
        system_tokens = sum(item.token_count for item in system_prompt)
        remaining_budget = max(20, 100 - system_tokens)  # Ensure at least 20 tokens for other messages
        
        truncated_others = truncate_context_window(
            other_window,
            max_tokens=remaining_budget,
            strategy=TruncationStrategy.LAST
        )
        
        # Combine back
        final_items = system_prompt + truncated_others.items
        final_window = create_context_window(final_items)
        
        print(f"After truncation: {len(final_window.items)} messages, {final_window.current_tokens} tokens")
        print("\nPreserved system prompt and recent messages")
        print(f"System: {system_prompt[0].content}")
        print(f"+ {len(truncated_others.items)} recent messages")
    else:
        print("No system prompt found or no other messages to truncate")
    
    # Example 4: Context rotation strategies
    print("\n4. CONTEXT ROTATION STRATEGIES")
    print("-"*80)
    print("Use case: Different strategies for managing conversation history")
    
    def demonstrate_strategy(strategy_name: str, description: str, keep_system: bool = True):
        """Demonstrate a rotation strategy."""
        print(f"\n{strategy_name}:")
        print(f"  {description}")
        
        # Create sample conversation
        items = [ContextItem(
            content="System: Assistant",
            token_count=5,
            priority=1.0,
            metadata={"role": "system"}
        )]
        
        for i in range(6):
            items.append(ContextItem(
                content=f"User message {i+1}",
                token_count=10,
                priority=0.8 - i * 0.05,
                metadata={"role": "user", "index": i}
            ))
            items.append(ContextItem(
                content=f"Assistant response {i+1}",
                token_count=15,
                priority=0.7 - i * 0.05,
                metadata={"role": "assistant", "index": i}
            ))
        
        window = create_context_window(items, max_tokens=80)
        
        if strategy_name == "FIFO (Keep Recent)":
            result = truncate_context_window(window, 80, TruncationStrategy.LAST)
        elif strategy_name == "Priority-Based":
            result = truncate_context_window(window, 80, TruncationStrategy.PRIORITY)
        else:
            result = window
        
        user_count = sum(1 for item in result.items if item.metadata.get("role") == "user")
        asst_count = sum(1 for item in result.items if item.metadata.get("role") == "assistant")
        print(f"  Kept: {user_count} user + {asst_count} assistant messages")
    
    demonstrate_strategy(
        "FIFO (Keep Recent)",
        "Drop oldest messages, keep most recent"
    )
    
    demonstrate_strategy(
        "Priority-Based",
        "Keep highest priority messages regardless of position"
    )
    
    # Example 5: Memory-efficient long conversations
    print("\n5. MEMORY-EFFICIENT LONG CONVERSATIONS")
    print("-"*80)
    print("Use case: Summarize old context to maintain long-term memory")
    
    # Simulate very long conversation
    very_long_conv = []
    
    # System message
    very_long_conv.append(ContextItem(
        content="System: AI coding assistant",
        token_count=10,
        priority=1.0,
        metadata={"role": "system"}
    ))
    
    # Old context (to be summarized)
    old_topics = ["variables", "loops", "functions", "classes", "modules"]
    for topic in old_topics:
        very_long_conv.append(ContextItem(
            content=f"Discussed {topic} in Python",
            token_count=15,
            priority=0.5,
            metadata={"role": "summary", "topic": topic, "is_summary": True}
        ))
    
    # Recent context (keep in full)
    recent_exchanges = [
        ("How do I use decorators?", "Decorators use @ syntax to modify functions"),
        ("Show me an example", "Here's an example: @decorator def function(): pass"),
        ("Can decorators take arguments?", "Yes, use decorator factories that return decorators"),
    ]
    
    for question, answer in recent_exchanges:
        very_long_conv.append(ContextItem(
            content=f"User: {question}",
            token_count=count_tokens(f"User: {question}"),
            priority=0.9,
            metadata={"role": "user"}
        ))
        very_long_conv.append(ContextItem(
            content=f"Assistant: {answer}",
            token_count=count_tokens(f"Assistant: {answer}"),
            priority=0.8,
            metadata={"role": "assistant"}
        ))
    
    efficient_window = create_context_window(very_long_conv)
    
    print(f"\nEfficient context structure:")
    print(f"  System: 1 message")
    print(f"  Summaries: {sum(1 for i in very_long_conv if i.metadata.get('is_summary', False))} items")
    print(f"  Recent: {len(recent_exchanges) * 2} messages")
    print(f"  Total tokens: {efficient_window.current_tokens}")
    
    print("\nContext breakdown:")
    print("  [System] AI coding assistant")
    print("  [Summaries] Previous topics: variables, loops, functions, classes, modules")
    print("  [Recent] Full conversation about decorators")
    
    # Example 6: Context budget allocation
    print("\n6. CONTEXT BUDGET ALLOCATION")
    print("-"*80)
    print("Use case: Allocate token budget across context types")
    
    total_budget = 200
    allocations = {
        "system": 0.10,      # 10% for system prompt
        "history": 0.30,     # 30% for conversation history
        "knowledge": 0.40,   # 40% for retrieved knowledge
        "current": 0.20,     # 20% for current query/response
    }
    
    print(f"\nTotal token budget: {total_budget}")
    print("\nAllocations:")
    for category, percentage in allocations.items():
        tokens = int(total_budget * percentage)
        print(f"  {category:12}: {percentage:5.0%} ({tokens:3} tokens)")
    
    # Create components
    components = {
        "system": [ContextItem(content="System prompt", token_count=20, priority=1.0)],
        "history": [ContextItem(content=f"Message {i}", token_count=15, priority=0.8) 
                   for i in range(5)],
        "knowledge": [ContextItem(content=f"Knowledge {i}", token_count=20, priority=0.9) 
                     for i in range(5)],
        "current": [ContextItem(content="Current query", token_count=40, priority=1.0)],
    }
    
    # Fit each component to budget
    fitted_components = {}
    for category, items in components.items():
        budget = int(total_budget * allocations[category])
        window = create_context_window(items, max_tokens=budget)
        fitted = truncate_context_window(window, budget, TruncationStrategy.PRIORITY)
        fitted_components[category] = fitted
        
    print("\nActual usage after fitting:")
    for category, window in fitted_components.items():
        budget = int(total_budget * allocations[category])
        print(f"  {category:12}: {window.current_tokens:3}/{budget:3} tokens "
              f"({len(window.items)} items)")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
