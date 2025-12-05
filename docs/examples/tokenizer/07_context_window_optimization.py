"""
Context Window Optimization Example
===================================

This example demonstrates how to optimize content to fit within model context
windows, which is critical for LLM applications working with long documents,
maintaining conversation history, or processing large amounts of data.

Main concepts:
- Managing content within context limits
- Prioritizing important information
- Sliding window techniques
- Conversation history pruning
"""

from kerb.tokenizer import (
    count_tokens,
    count_tokens_for_messages,
    truncate_to_token_limit,
    Tokenizer
)
from typing import List, Dict, Tuple


def main():
    """Run context window optimization examples."""
    
    print("="*80)
    print("CONTEXT WINDOW OPTIMIZATION EXAMPLE")
    print("="*80)
    
    # Example 1: Fitting long documents into context window
    print("\n" + "-"*80)
    print("EXAMPLE 1: Document Summarization Within Context Limits")
    print("-"*80)
    
    # Simulate a long document
    long_document = """

# %%
# Setup and Imports
# -----------------
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that focuses on building
    systems that can learn from data and improve their performance over time without
    being explicitly programmed.
    
    History
    The field of machine learning has its roots in the 1950s when Arthur Samuel
    developed a program that could learn to play checkers. Since then, the field
    has evolved dramatically with advances in computing power and data availability.
    
    Types of Machine Learning
    There are three main types: supervised learning, unsupervised learning, and
    reinforcement learning. Each has its own use cases and applications.
    
    Applications
    Machine learning is used in countless applications today, from recommendation
    systems to autonomous vehicles, medical diagnosis to fraud detection.
    
    Future Trends
    The future of machine learning includes advances in deep learning, transfer
    learning, and AutoML technologies that make ML more accessible.
    """ * 5  # Repeat to make it longer
    
    context_limit = 200
    system_prompt = "Summarize the following document in a clear, concise manner."
    max_completion_tokens = 100
    
    print(f"Context window: {context_limit} tokens")
    print(f"Reserved for completion: {max_completion_tokens} tokens")
    
    system_tokens = count_tokens(system_prompt, tokenizer=Tokenizer.CL100K_BASE)
    available_for_document = context_limit - max_completion_tokens - system_tokens - 10  # 10 for overhead
    
    print(f"System prompt tokens: {system_tokens}")
    print(f"Available for document: {available_for_document} tokens")
    
    doc_tokens = count_tokens(long_document, tokenizer=Tokenizer.CL100K_BASE)
    print(f"\nOriginal document: {doc_tokens} tokens")
    
    if doc_tokens > available_for_document:
        print(f"Document exceeds available space by {doc_tokens - available_for_document} tokens")
        print("\nTruncating document...")
        
        truncated_doc = truncate_to_token_limit(
            long_document,
            max_tokens=available_for_document,
            tokenizer=Tokenizer.CL100K_BASE,
            ellipsis="...[document truncated]"
        )
        
        truncated_tokens = count_tokens(truncated_doc, tokenizer=Tokenizer.CL100K_BASE)
        print(f"Truncated to: {truncated_tokens} tokens")
        print(f"Fits in context: {system_tokens + truncated_tokens + max_completion_tokens < context_limit}")
    
    # Example 2: Sliding window for long conversations
    print("\n" + "-"*80)
    print("EXAMPLE 2: Sliding Window for Long Conversations")
    print("-"*80)
    
    # Simulate a long conversation
    full_conversation = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "How do I create a list in Python?"},
        {"role": "assistant", "content": "Use square brackets: my_list = [1, 2, 3]"},
        {"role": "user", "content": "How do I add items?"},
        {"role": "assistant", "content": "Use append(): my_list.append(4)"},
        {"role": "user", "content": "What about removing items?"},
        {"role": "assistant", "content": "Use remove() or pop(): my_list.remove(2) or my_list.pop()"},
        {"role": "user", "content": "How do I sort a list?"},
        {"role": "assistant", "content": "Use sorted() or .sort(): sorted_list = sorted(my_list)"},
        {"role": "user", "content": "What's list comprehension?"},
        {"role": "assistant", "content": "It's a concise way to create lists: [x*2 for x in range(5)]"},
    ]
    
    context_limit = 200
    print(f"Context limit: {context_limit} tokens")
    print(f"Full conversation: {len(full_conversation)} messages\n")
    
    full_tokens = count_tokens_for_messages(full_conversation, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Full conversation tokens: {full_tokens}")
    
    if full_tokens > context_limit:
        print(f"Exceeds limit by {full_tokens - context_limit} tokens")
        print("\nApplying sliding window strategy...")
        
        # Keep system message + most recent messages
        system_msg = full_conversation[0]
        remaining_messages = full_conversation[1:]
        
        # Binary search for maximum messages that fit
        for window_size in range(len(remaining_messages), 0, -1):
            window = [system_msg] + remaining_messages[-window_size:]
            window_tokens = count_tokens_for_messages(window, tokenizer=Tokenizer.CL100K_BASE)
            
            if window_tokens <= context_limit:
                print(f"\nOptimal window: {len(window)} messages ({window_tokens} tokens)")
                print(f"Kept messages: system + last {window_size} exchanges")
                print(f"Dropped messages: {len(full_conversation) - len(window)}")
                break
    
    # Example 3: Prioritizing important context
    print("\n" + "-"*80)
    print("EXAMPLE 3: Prioritizing Important Context")
    print("-"*80)
    
    # Different types of context with priorities
    context_pieces = [
        ("critical", "User ID: 12345, Session: active, Auth: verified"),
        ("important", "Recent purchase history: 3 items, Total: $150.50"),
        ("useful", "User preferences: Theme=dark, Language=en, Notifications=on"),
        ("optional", "Browser: Chrome, OS: macOS, Screen: 1920x1080"),
        ("optional", "Last login: 2024-10-14 10:30:00"),
        ("critical", "Current task: Process refund request"),
        ("important", "Order #67890 - Status: Delivered on 2024-10-10"),
        ("useful", "Customer tier: Premium, Member since: 2023-05"),
    ]
    
    context_limit = 100
    print(f"Context limit: {context_limit} tokens")
    print(f"Context pieces: {len(context_pieces)}\n")
    
    # Sort by priority
    priority_order = {"critical": 0, "important": 1, "useful": 2, "optional": 3}
    sorted_context = sorted(context_pieces, key=lambda x: priority_order[x[0]])
    
    # Add context pieces by priority until limit reached
    selected_context = []
    total_tokens = 0
    
    for priority, content in sorted_context:
        tokens = count_tokens(content, tokenizer=Tokenizer.CL100K_BASE)
        
        if total_tokens + tokens <= context_limit:
            selected_context.append((priority, content))
            total_tokens += tokens
        else:
            print(f"Stopped at {total_tokens} tokens (limit: {context_limit})")
            break
    
    print("Selected context pieces:")
    for priority, content in selected_context:
        tokens = count_tokens(content, tokenizer=Tokenizer.CL100K_BASE)
        print(f"  [{priority:10s}] ({tokens:2d} tokens) {content}")
    
    print(f"\nTotal: {total_tokens}/{context_limit} tokens")
    print(f"Included: {len(selected_context)}/{len(context_pieces)} pieces")
    
    # Example 4: Dynamic context adjustment
    print("\n" + "-"*80)
    print("EXAMPLE 4: Dynamic Context Adjustment")
    print("-"*80)
    
    def optimize_context_allocation(
        system_prompt: str,
        user_query: str,
        background_info: str,
        context_limit: int,
        min_completion_tokens: int
    ) -> Dict:
        """Dynamically allocate tokens between components."""
        
        system_tokens = count_tokens(system_prompt, tokenizer=Tokenizer.CL100K_BASE)
        query_tokens = count_tokens(user_query, tokenizer=Tokenizer.CL100K_BASE)
        info_tokens = count_tokens(background_info, tokenizer=Tokenizer.CL100K_BASE)
        
        # Calculate available space
        available = context_limit - min_completion_tokens
        required_base = system_tokens + query_tokens
        available_for_info = available - required_base
        
        if available_for_info < 0:
            return {
                "fits": False,
                "reason": "System prompt + query exceed available space"
            }
        
        # Adjust background info if needed
        if info_tokens > available_for_info:
            adjusted_info = truncate_to_token_limit(
                background_info,
                max_tokens=available_for_info,
                tokenizer=Tokenizer.CL100K_BASE
            )
            adjusted_info_tokens = count_tokens(adjusted_info, tokenizer=Tokenizer.CL100K_BASE)
        else:
            adjusted_info = background_info
            adjusted_info_tokens = info_tokens
        
        total_input = system_tokens + query_tokens + adjusted_info_tokens
        
        return {
            "fits": True,
            "system_tokens": system_tokens,
            "query_tokens": query_tokens,
            "background_tokens": adjusted_info_tokens,
            "total_input_tokens": total_input,
            "reserved_completion_tokens": min_completion_tokens,
            "total_tokens": total_input + min_completion_tokens,
            "background_truncated": info_tokens > available_for_info,
            "adjusted_background": adjusted_info if info_tokens > available_for_info else None
        }
    
    system = "You are a customer service AI assistant."
    query = "What's the status of my recent order?"
    background = (
        "Customer: John Doe. Account since 2023. Premium member. "
        "Recent orders: #001 (delivered), #002 (shipped), #003 (processing). "
        "Support history: 2 tickets resolved. Satisfaction rating: 5/5. "
        "Preferences: Email notifications, Express shipping. "
    ) * 3
    
    context_limit = 300
    min_completion = 100
    
    print(f"Context limit: {context_limit} tokens")
    print(f"Min completion tokens: {min_completion}")
    print(f"\nOriginal components:")
    print(f"  System: {count_tokens(system, tokenizer=Tokenizer.CL100K_BASE)} tokens")
    print(f"  Query: {count_tokens(query, tokenizer=Tokenizer.CL100K_BASE)} tokens")
    print(f"  Background: {count_tokens(background, tokenizer=Tokenizer.CL100K_BASE)} tokens")
    
    result = optimize_context_allocation(system, query, background, context_limit, min_completion)
    
    if result["fits"]:
        print(f"\nOptimized allocation:")
        print(f"  System: {result['system_tokens']} tokens")
        print(f"  Query: {result['query_tokens']} tokens")
        print(f"  Background: {result['background_tokens']} tokens")
        print(f"  Total input: {result['total_input_tokens']} tokens")
        print(f"  Reserved for completion: {result['reserved_completion_tokens']} tokens")
        print(f"  Total: {result['total_tokens']} tokens")
        
        if result["background_truncated"]:
            print(f"\nBackground was truncated to fit context window")
    else:
        print(f"\nERROR: {result['reason']}")
    
    # Example 5: Multi-document context management
    print("\n" + "-"*80)
    print("EXAMPLE 5: Multi-Document Context Management")
    print("-"*80)
    
    documents = {
        "doc1": "Product specifications: The device features a 6.5-inch display, 128GB storage, and 5G connectivity.",
        "doc2": "User manual excerpt: To activate the device, press and hold the power button for 3 seconds.",
        "doc3": "Warranty information: This product includes a 2-year limited warranty covering manufacturing defects.",
        "doc4": "Troubleshooting guide: If the device won't turn on, try charging it for at least 30 minutes.",
        "doc5": "Safety instructions: Do not expose the device to extreme temperatures or moisture.",
    }
    
    query = "How do I turn on my device and what warranty does it have?"
    context_limit = 150
    
    print(f"Query: {query}")
    print(f"Available documents: {len(documents)}")
    print(f"Context limit: {context_limit} tokens\n")
    
    # Simple relevance scoring (in practice, use embeddings or keyword matching)
    # For this example, we'll prioritize docs with keywords from the query
    keywords = ["turn on", "warranty", "activate", "power"]
    
    doc_scores = []
    for doc_id, content in documents.items():
        score = sum(1 for keyword in keywords if keyword in content.lower())
        tokens = count_tokens(content, tokenizer=Tokenizer.CL100K_BASE)
        doc_scores.append((score, doc_id, content, tokens))
    
    # Sort by relevance score (descending)
    doc_scores.sort(reverse=True)
    
    # Select documents until context limit
    query_tokens = count_tokens(query, tokenizer=Tokenizer.CL100K_BASE)
    available_for_docs = context_limit - query_tokens - 20  # Reserve 20 for formatting
    
    selected_docs = []
    used_tokens = 0
    
    print("Document selection by relevance:")
    for score, doc_id, content, tokens in doc_scores:
        if used_tokens + tokens <= available_for_docs:
            selected_docs.append(doc_id)
            used_tokens += tokens
            print(f"  SELECTED {doc_id} (relevance: {score}, tokens: {tokens})")
            print(f"    {content[:60]}...")
        else:
            print(f"  SKIPPED {doc_id} (relevance: {score}, tokens: {tokens}) - Would exceed limit")
    
    print(f"\nContext summary:")
    print(f"  Query tokens: {query_tokens}")
    print(f"  Document tokens: {used_tokens}")
    print(f"  Total: {query_tokens + used_tokens}/{context_limit}")
    print(f"  Documents included: {len(selected_docs)}/{len(documents)}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
