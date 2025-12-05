"""
LLM Context Optimization Example
================================

This example demonstrates how to optimize conversation context for LLM API calls
with token limits and intelligent memory selection.

Main concepts:
- Token budget management for API calls
- Smart context window selection
- Balancing historical context vs. recent messages
- Priority-based message inclusion
- Context compression strategies
"""

from kerb.memory import (
    ConversationBuffer,
    create_token_limited_window,
    create_sliding_window
)
from kerb.memory.patterns import get_relevant_memory
from kerb.core.types import Message


def estimate_tokens(text: str) -> int:
    """Estimate token count (approximation: ~1 token per 0.75 words)."""
    return int(len(text.split()) / 0.75)


def main():
    """Run LLM context optimization example."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("LLM CONTEXT OPTIMIZATION EXAMPLE")
    print("="*80)
    
    # Simulate a long conversation
    buffer = ConversationBuffer(max_messages=200)
    
    conversation = [
        ("system", "You are an expert Python developer helping with code optimization."),
        ("user", "I have a slow Python script that processes CSV files. Can you help?"),
        ("assistant", "I'd be happy to help! What does the script do and what's causing the slowness?"),
        ("user", "It reads a 10GB CSV file and filters rows based on multiple conditions."),
        ("assistant", "For large CSV files, consider using pandas with chunksize parameter or Dask for parallel processing."),
        ("user", "How does chunking work in pandas?"),
        ("assistant", "Use pd.read_csv('file.csv', chunksize=10000) to process 10,000 rows at a time. This reduces memory usage significantly."),
        ("user", "What about using multiprocessing?"),
        ("assistant", "Great idea! Use multiprocessing.Pool to parallelize chunk processing. Each process handles a portion of the data independently."),
        ("user", "Can you show me a code example?"),
        ("assistant", "Here's a basic example using chunking: for chunk in pd.read_csv('data.csv', chunksize=10000): filtered = chunk[chunk['value'] > 100]; process(filtered)"),
        ("user", "How do I optimize the filtering conditions?"),
        ("assistant", "Use vectorized operations instead of apply(). For example: df[df['col'] > 100] is faster than df[df.apply(lambda x: x['col'] > 100)]"),
        ("user", "What about memory usage optimization?"),
        ("assistant", "Specify dtypes when reading CSV to reduce memory. Use category dtype for columns with limited unique values. Consider using pyarrow engine."),
        ("user", "Should I use Dask or stick with pandas?"),
        ("assistant", "Dask is great when data doesn't fit in memory. If pandas chunking works and data fits in RAM, stick with pandas for simplicity."),
        ("user", "How do I profile my script's performance?"),
        ("assistant", "Use cProfile for CPU profiling and memory_profiler for memory. Run: python -m cProfile -s cumtime script.py"),
        ("user", "What's the best format for storing processed data?"),
        ("assistant", "Parquet format is efficient and compressed. Use df.to_parquet('output.parquet') instead of CSV for better performance."),
    ]
    
    for role, content in conversation:
        buffer.add_message(role, content)
    
    print(f"Created conversation with {len(buffer.messages)} messages")
    
    # Calculate total tokens
    total_tokens = sum(estimate_tokens(m.content) for m in buffer.messages)
    print(f"Total conversation tokens: ~{total_tokens}")
    
    # Scenario 1: API with 4K token limit
    print("\n" + "-"*80)
    print("SCENARIO 1: API WITH 4K TOKEN LIMIT")
    print("-"*80)
    
    # Example: GPT-3.5 with 4K context
    total_limit = 4096
    completion_budget = 500  # Reserve for response
    system_message_tokens = estimate_tokens(buffer.messages[0].content) if buffer.messages[0].role == "system" else 0
    available_for_context = total_limit - completion_budget - system_message_tokens
    
    print(f"\nToken budget:")
    print(f"  Total limit: {total_limit}")
    print(f"  Reserved for completion: {completion_budget}")
    print(f"  System message: {system_message_tokens}")
    print(f"  Available for context: {available_for_context}")
    
    # Get messages within token limit
    context_messages = create_token_limited_window(
        [m for m in buffer.messages if m.role != "system"],
        max_tokens=available_for_context,
        token_estimator=estimate_tokens
    )
    
    context_tokens = sum(estimate_tokens(m.content) for m in context_messages)
    print(f"\nOptimized context:")
    print(f"  Messages included: {len(context_messages)}")
    print(f"  Context tokens: {context_tokens}")
    print(f"  Remaining for completion: {total_limit - system_message_tokens - context_tokens}")
    
    # Scenario 2: Priority-based context selection
    print("\n" + "-"*80)
    print("SCENARIO 2: PRIORITY-BASED SELECTION")
    print("-"*80)
    

# %%
# Get Message Priority
# --------------------

    def get_message_priority(msg: Message) -> int:
        """Assign priority to messages."""
        priority = 0
        
        # System messages: highest priority
        if msg.role == "system":
            priority = 100
        
        # User questions: high priority
        elif msg.role == "user" and "?" in msg.content:
            priority = 80
        
        # Code examples: high priority
        elif "def " in msg.content or "import " in msg.content or "for " in msg.content:
            priority = 70
        
        # Recent messages: medium priority
        elif msg in buffer.messages[-5:]:
            priority = 60
        
        # Other messages: lower priority
        else:
            priority = 30
        
        return priority
    
    # Sort by priority and select within token budget
    prioritized = sorted(buffer.messages, key=get_message_priority, reverse=True)
    
    selected_messages = []
    token_count = 0
    
    for msg in prioritized:
        msg_tokens = estimate_tokens(msg.content)
        if token_count + msg_tokens <= available_for_context:
            selected_messages.append(msg)
            token_count += msg_tokens
    
    # Sort selected messages chronologically
    selected_messages.sort(key=lambda m: buffer.messages.index(m))
    
    print(f"\nPriority-based selection:")
    print(f"  Messages selected: {len(selected_messages)}")
    print(f"  Total tokens: {token_count}")
    
    # Show selected messages with priority
    print(f"\nSelected messages (in order):")
    for i, msg in enumerate(selected_messages[:5], 1):
        priority = get_message_priority(msg)
        print(f"  [{i}] Priority {priority}: {msg.role}: {msg.content[:60]}...")
    
    # Scenario 3: Hybrid approach - relevance + recency
    print("\n" + "-"*80)
    print("SCENARIO 3: HYBRID (RELEVANCE + RECENCY)")
    print("-"*80)
    
    # User asks a new question
    new_query = "How do I handle errors when processing chunks?"
    
    # Split budget: 70% for relevant context, 30% for recent messages
    relevant_budget = int(available_for_context * 0.7)
    recent_budget = int(available_for_context * 0.3)
    
    # Get relevant historical messages
    relevant_messages = get_relevant_memory(new_query, buffer, top_k=10)
    relevant_context = create_token_limited_window(
        relevant_messages,
        max_tokens=relevant_budget,
        token_estimator=estimate_tokens
    )
    
    # Get recent messages (excluding those already in relevant context)
    recent_candidates = [m for m in buffer.messages[-10:] if m not in relevant_context]
    recent_context = create_token_limited_window(
        recent_candidates,
        max_tokens=recent_budget,
        token_estimator=estimate_tokens
    )
    
    # Combine contexts
    hybrid_context = relevant_context + recent_context
    hybrid_tokens = sum(estimate_tokens(m.content) for m in hybrid_context)
    
    print(f"\nNew query: '{new_query}'")
    print(f"\nHybrid context allocation:")
    print(f"  Relevant messages: {len(relevant_context)} (~{sum(estimate_tokens(m.content) for m in relevant_context)} tokens)")
    print(f"  Recent messages: {len(recent_context)} (~{sum(estimate_tokens(m.content) for m in recent_context)} tokens)")
    print(f"  Total: {len(hybrid_context)} messages (~{hybrid_tokens} tokens)")
    
    # Scenario 4: Context compression
    print("\n" + "-"*80)
    print("SCENARIO 4: CONTEXT COMPRESSION")
    print("-"*80)
    
    # Use summaries to compress old context
    compressed_buffer = ConversationBuffer(
        max_messages=8,
        enable_summaries=True
    )
    
    # Add all messages (triggers summarization)
    for msg in buffer.messages:
        compressed_buffer.add_message(msg.role, msg.content)
    
    # Build compressed context
    compressed_context = compressed_buffer.get_context(include_summary=True)
    compressed_tokens = estimate_tokens(compressed_context)
    
    print(f"\nCompression results:")
    print(f"  Original messages: {len(buffer.messages)}")
    print(f"  Original tokens: ~{total_tokens}")
    print(f"  Compressed messages: {len(compressed_buffer.messages)}")
    print(f"  Summaries: {len(compressed_buffer.summaries)}")
    print(f"  Compressed tokens: ~{compressed_tokens}")
    print(f"  Compression ratio: {(1 - compressed_tokens/total_tokens)*100:.1f}%")
    
    # Scenario 5: Multi-model optimization
    print("\n" + "-"*80)
    print("SCENARIO 5: MULTI-MODEL OPTIMIZATION")
    print("-"*80)
    
    # Different models have different token limits
    models = {
        "gpt-3.5-turbo": {"context_limit": 4096, "completion_reserve": 500},
        "gpt-4": {"context_limit": 8192, "completion_reserve": 1000},
        "claude-instant": {"context_limit": 9000, "completion_reserve": 1000},
        "claude-2": {"context_limit": 100000, "completion_reserve": 2000},
    }
    
    print("\nContext optimization for different models:")
    
    for model_name, limits in models.items():
        available = limits["context_limit"] - limits["completion_reserve"] - system_message_tokens
        
        optimized = create_token_limited_window(
            [m for m in buffer.messages if m.role != "system"],
            max_tokens=available,
            token_estimator=estimate_tokens
        )
        
        optimized_tokens = sum(estimate_tokens(m.content) for m in optimized)
        
        print(f"\n  {model_name}:")
        print(f"    Context limit: {limits['context_limit']}")
        print(f"    Available: {available}")
        print(f"    Messages fit: {len(optimized)}/{len(buffer.messages)}")
        print(f"    Tokens used: {optimized_tokens}")
    
    # Real-world: Complete API call preparation
    print("\n" + "-"*80)
    print("REAL-WORLD: API CALL PREPARATION")
    print("-"*80)
    
    def prepare_api_context(
        buffer: ConversationBuffer,
        query: str,
        model_limit: int = 4096,
        completion_budget: int = 500,
        strategy: str = "hybrid"
    ):
        """Prepare optimized context for API call."""
        
        # Get system message
        system_msg = next((m for m in buffer.messages if m.role == "system"), None)
        system_tokens = estimate_tokens(system_msg.content) if system_msg else 0
        
        # Calculate available budget
        available = model_limit - completion_budget - system_tokens
        
        # Build context based on strategy
        if strategy == "recent":
            context = create_token_limited_window(
                [m for m in buffer.messages if m.role != "system"],
                max_tokens=available,
                token_estimator=estimate_tokens
            )
        elif strategy == "hybrid":
            # 70% relevant, 30% recent
            relevant_budget = int(available * 0.7)
            recent_budget = int(available * 0.3)
            
            relevant = get_relevant_memory(query, buffer, top_k=10)
            relevant_ctx = create_token_limited_window(relevant, max_tokens=relevant_budget, token_estimator=estimate_tokens)
            
            recent_candidates = [m for m in buffer.messages[-10:] if m not in relevant_ctx and m.role != "system"]
            recent_ctx = create_token_limited_window(recent_candidates, max_tokens=recent_budget, token_estimator=estimate_tokens)
            
            context = relevant_ctx + recent_ctx
        else:
            context = buffer.messages
        
        # Format for API
        api_messages = []
        if system_msg:
            api_messages.append({"role": "system", "content": system_msg.content})
        
        for msg in context:
            if msg.role != "system":
                api_messages.append({"role": msg.role, "content": msg.content})
        
        return api_messages
    
    # Prepare for API call
    api_context = prepare_api_context(
        buffer,
        query=new_query,
        model_limit=4096,
        completion_budget=500,
        strategy="hybrid"
    )
    
    total_api_tokens = sum(estimate_tokens(m["content"]) for m in api_context)
    
    print(f"\nPrepared API context:")
    print(f"  Model: gpt-3.5-turbo (4096 tokens)")
    print(f"  Query: '{new_query}'")
    print(f"  Messages: {len(api_context)}")
    print(f"  Total tokens: ~{total_api_tokens}")
    print(f"  Reserved for completion: 500")
    print(f"  Safety margin: {4096 - total_api_tokens - 500} tokens")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
