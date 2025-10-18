"""Sliding Window Context Example

This example demonstrates sliding window techniques for processing
long documents and maintaining context continuity in LLM applications.

Main concepts:
- Fixed-size sliding windows
- Token-based sliding windows
- Overlapping windows for continuity
- Adaptive windows combining priority and recency
- Processing long documents in chunks
"""

from kerb.context import (
    ContextItem,
    ContextWindow,
    create_sliding_window,
    create_token_sliding_window,
    create_adaptive_window,
)
from kerb.tokenizer import count_tokens


def create_long_document():
    """Create a sample long document split into paragraphs."""
    paragraphs = [
        "Introduction: Machine learning has revolutionized software development.",
        "Neural networks form the foundation of modern deep learning systems.",
        "Training requires large datasets and significant computational resources.",
        "Supervised learning uses labeled data to train predictive models.",
        "Unsupervised learning finds patterns in unlabeled data automatically.",
        "Reinforcement learning agents learn through trial and error interactions.",
        "Transfer learning leverages pre-trained models for new tasks efficiently.",
        "Model evaluation uses metrics like accuracy, precision, and recall.",
        "Overfitting occurs when models memorize training data instead of generalizing.",
        "Regularization techniques help prevent overfitting and improve generalization.",
        "Cross-validation provides robust estimates of model performance.",
        "Hyperparameter tuning optimizes model configuration for best results.",
        "Ensemble methods combine multiple models for better predictions.",
        "Feature engineering transforms raw data into useful model inputs.",
        "Conclusion: Continuous learning and experimentation drive ML success.",
    ]
    
    items = []
    for i, para in enumerate(paragraphs):
        items.append(ContextItem(
            content=para,
            token_count=count_tokens(para),
            priority=1.0,
            metadata={"paragraph": i, "section": i // 5}
        ))
    
    return items


def main():
    """Run sliding window example."""
    
    print("="*80)
    print("SLIDING WINDOW CONTEXT EXAMPLE")
    print("="*80)
    
    # Create long document
    document = create_long_document()
    total_tokens = sum(item.token_count for item in document)
    
    print(f"\nDocument: {len(document)} paragraphs, {total_tokens} tokens total")
    
    # Example 1: Fixed-size sliding windows
    print("\n1. FIXED-SIZE SLIDING WINDOWS")
    print("-"*80)
    print("Use case: Process document in equal-sized chunks")
    
    windows = create_sliding_window(document, window_size=3, step_size=3)
    
    print(f"\nCreated {len(windows)} non-overlapping windows (size=3)")
    for i, window in enumerate(windows, 1):
        print(f"\nWindow {i} ({window.current_tokens} tokens):")
        for item in window.items:
            print(f"  - {item.content[:60]}...")
    
    # Example 2: Overlapping sliding windows
    print("\n2. OVERLAPPING SLIDING WINDOWS")
    print("-"*80)
    print("Use case: Maintain context continuity between chunks")
    
    overlap_windows = create_sliding_window(document, window_size=4, step_size=2)
    
    print(f"\nCreated {len(overlap_windows)} overlapping windows")
    print("(window_size=4, step_size=2)")
    
    for i, window in enumerate(overlap_windows[:3], 1):  # Show first 3
        print(f"\nWindow {i}:")
        paragraphs = [item.metadata.get("paragraph", "?") for item in window.items]
        print(f"  Paragraphs: {paragraphs}")
        print(f"  Content: {window.items[0].content[:50]}...")
    
    if len(overlap_windows) > 3:
        print(f"\n... and {len(overlap_windows) - 3} more windows")
    
    # Example 3: Token-based sliding windows
    print("\n3. TOKEN-BASED SLIDING WINDOWS")
    print("-"*80)
    print("Use case: Create windows that fit within specific token limits")
    
    token_windows = create_token_sliding_window(
        document,
        max_tokens=100,
        overlap_tokens=20
    )
    
    print(f"\nCreated {len(token_windows)} windows (max_tokens=100, overlap=20)")
    
    for i, window in enumerate(token_windows, 1):
        print(f"\nWindow {i}: {window.current_tokens} tokens, {len(window.items)} items")
        first_para = window.items[0].metadata.get("paragraph", "?")
        last_para = window.items[-1].metadata.get("paragraph", "?")
        print(f"  Paragraphs {first_para}-{last_para}")
    
    # Example 4: Processing long documents with summarization
    print("\n4. PROCESSING WITH SUMMARIZATION")
    print("-"*80)
    print("Use case: Summarize document sections progressively")
    
    # Create windows for processing
    processing_windows = create_sliding_window(document, window_size=5, step_size=5)
    
    print(f"\nProcessing {len(processing_windows)} sections:")
    
    summaries = []
    for i, window in enumerate(processing_windows, 1):
        section_content = " ".join(item.content for item in window.items)
        
        # Simulate summary (in real use, pass to LLM)
        summary = f"Section {i}: {window.items[0].content.split(':')[0]}"
        summaries.append(summary)
        
        print(f"\nSection {i}:")
        print(f"  Original: {window.current_tokens} tokens")
        print(f"  Summary: {summary}")
    
    print("\nFinal document summary:")
    for summary in summaries:
        print(f"  - {summary}")
    
    # Example 5: Adaptive windows
    print("\n5. ADAPTIVE WINDOWS")
    print("-"*80)
    print("Use case: Create windows balancing recency and priority")
    
    # Mark some items as high priority
    for item in document:
        # Introduction and conclusion are high priority
        para_num = item.metadata.get("paragraph", 0)
        if para_num == 0 or para_num >= 14:
            item.priority = 1.0
        elif "learning" in item.content.lower():
            item.priority = 0.9
        else:
            item.priority = 0.7
    
    adaptive_win = create_adaptive_window(
        document,
        max_tokens=150,
        recency_weight=0.4,
        priority_weight=0.6
    )
    
    print(f"\nAdaptive window: {adaptive_win.current_tokens} tokens, "
          f"{len(adaptive_win.items)} items selected")
    print("\nSelected items (by combined score):")
    for item in adaptive_win.items:
        para = item.metadata.get("paragraph", "?")
        print(f"  Para {para} (priority {item.priority:.1f}): {item.content[:60]}...")
    
    # Example 6: Conversation history sliding window
    print("\n6. CONVERSATION HISTORY WINDOW")
    print("-"*80)
    print("Use case: Maintain recent conversation context")
    
    conversation = []
    for i in range(10):
        user_msg = ContextItem(
            content=f"User message {i+1}",
            token_count=5,
            metadata={"role": "user", "turn": i}
        )
        assistant_msg = ContextItem(
            content=f"Assistant response {i+1} with detailed explanation",
            token_count=10,
            metadata={"role": "assistant", "turn": i}
        )
        conversation.extend([user_msg, assistant_msg])
    
    print(f"Full conversation: {len(conversation)} messages, "
          f"{sum(m.token_count for m in conversation)} tokens")
    
    # Keep last 5 exchanges (10 messages)
    conv_windows = create_sliding_window(
        conversation,
        window_size=10,
        step_size=10
    )
    
    recent_context = conv_windows[-1] if conv_windows else ContextWindow()
    
    print(f"\nRecent context window: {len(recent_context.items)} messages")
    for item in recent_context.items[-4:]:  # Show last 2 exchanges
        role = item.metadata.get("role", "unknown")
        print(f"  [{role}] {item.content}")
    
    # Example 7: Code review sliding window
    print("\n7. CODE REVIEW SLIDING WINDOW")
    print("-"*80)
    print("Use case: Review large code files in manageable chunks")
    
    code_sections = [
        ContextItem(
            content="import statements and module docstring",
            token_count=count_tokens("import statements and module docstring"),
            metadata={"type": "imports", "line_range": "1-10"}
        ),
        ContextItem(
            content="Class definition with initialization method",
            token_count=count_tokens("Class definition with initialization method"),
            metadata={"type": "class", "line_range": "11-30"}
        ),
        ContextItem(
            content="Public method implementations for API",
            token_count=count_tokens("Public method implementations for API"),
            metadata={"type": "methods", "line_range": "31-60"}
        ),
        ContextItem(
            content="Private helper methods and utilities",
            token_count=count_tokens("Private helper methods and utilities"),
            metadata={"type": "helpers", "line_range": "61-80"}
        ),
        ContextItem(
            content="Error handling and exception classes",
            token_count=count_tokens("Error handling and exception classes"),
            metadata={"type": "errors", "line_range": "81-100"}
        ),
        ContextItem(
            content="Unit tests and test fixtures",
            token_count=count_tokens("Unit tests and test fixtures"),
            metadata={"type": "tests", "line_range": "101-150"}
        ),
    ]
    
    review_windows = create_sliding_window(code_sections, window_size=2, step_size=1)
    
    print(f"\nCreated {len(review_windows)} review windows")
    print("Each window shows context from adjacent sections:\n")
    
    for i, window in enumerate(review_windows[:3], 1):  # Show first 3
        types = [item.metadata.get("type", "?") for item in window.items]
        ranges = [item.metadata.get("line_range", "?") for item in window.items]
        print(f"Review Window {i}:")
        print(f"  Sections: {', '.join(types)}")
        print(f"  Lines: {', '.join(ranges)}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
