"""
Basic Context Window Example
============================

============================

This example demonstrates fundamental context window management for LLM applications.

Main concepts:
- Creating context windows to manage LLM input
- Adding context items with metadata
- Token counting and tracking
- Managing context window size
- Basic context retrieval
"""

from kerb.context import (
    ContextItem,
    ContextWindow,
    create_context_window,
)
from kerb.tokenizer import count_tokens


def main():
    """Run basic context window example."""
    
    print("="*80)
    print("BASIC CONTEXT WINDOW EXAMPLE")
    print("="*80)
    
    # Example 1: Manual context window creation
    print("\n1. MANUAL CONTEXT WINDOW CREATION")
    print("-"*80)
    
    window = ContextWindow(max_tokens=1000)
    print(f"Created empty context window (max_tokens={window.max_tokens})")
    
    # Add context items manually
    item1 = ContextItem(
        content="The user is building a Python web application using Flask.",
        priority=1.0,
        token_count=count_tokens("The user is building a Python web application using Flask."),
        metadata={"type": "system_info", "timestamp": "2024-01-01"}
    )
    
    item2 = ContextItem(
        content="User preferences: Dark mode enabled, syntax highlighting preferred.",
        priority=0.8,
        token_count=count_tokens("User preferences: Dark mode enabled, syntax highlighting preferred."),
        metadata={"type": "preferences"}
    )
    
    window.add_item(item1)
    window.add_item(item2)
    
    print(f"\nAdded {len(window.items)} items to context")
    print(f"Current tokens: {window.current_tokens}/{window.max_tokens}")
    
    # Example 2: Creating context window with helper function
    print("\n2. CREATING CONTEXT WINDOW WITH HELPER")
    print("-"*80)
    
    # Sample conversation history
    conversation = [
        "User: How do I implement authentication in Flask?",
        "Assistant: Flask authentication can be implemented using Flask-Login extension...",
        "User: Can you show me a code example?",
        "Assistant: Here's a basic example of Flask-Login setup...",
    ]
    
    # Create context items from conversation
    items = []
    for i, message in enumerate(conversation):
        item = ContextItem(
            content=message,
            priority=1.0 - (i * 0.1),  # More recent messages have higher priority
            token_count=count_tokens(message),
            metadata={"turn": i, "type": "conversation"}
        )
        items.append(item)
    
    # Create window with items
    conv_window = create_context_window(items, max_tokens=500)
    
    print(f"Created conversation context with {len(conv_window.items)} turns")
    print(f"Total tokens: {conv_window.current_tokens}")
    
    # Example 3: Managing different types of context
    print("\n3. MANAGING DIFFERENT CONTEXT TYPES")
    print("-"*80)
    
    # Context for code generation task
    code_context = [
        ContextItem(
            content="Task: Generate a REST API endpoint for user registration",
            priority=1.0,
            token_count=count_tokens("Task: Generate a REST API endpoint for user registration"),
            item_type="task",
            metadata={"source": "user_request"}
        ),
        ContextItem(
            content="Framework: Flask with SQLAlchemy ORM",
            priority=0.9,
            token_count=count_tokens("Framework: Flask with SQLAlchemy ORM"),
            item_type="system_info",
            metadata={"source": "project_config"}
        ),
        ContextItem(
            content="Database schema: User table with id, email, password_hash, created_at",
            priority=0.9,
            token_count=count_tokens("Database schema: User table with id, email, password_hash, created_at"),
            item_type="code",
            metadata={"source": "database_schema"}
        ),
        ContextItem(
            content="Security requirements: Use bcrypt for password hashing, validate email format",
            priority=0.8,
            token_count=count_tokens("Security requirements: Use bcrypt for password hashing, validate email format"),
            item_type="requirements",
            metadata={"source": "security_policy"}
        ),
    ]
    
    code_window = create_context_window(code_context, max_tokens=2000)
    
    print(f"Created code generation context:")
    for item in code_window.items:
        print(f"  - {item.item_type}: {item.content[:60]}... (priority={item.priority})")
    
    # Example 4: Retrieving and inspecting context
    print("\n4. RETRIEVING CONTEXT CONTENT")
    print("-"*80)
    
    # Get full context as string
    full_context = code_window.get_content()
    print(f"Full context length: {len(full_context)} characters")
    print(f"\nFormatted context:\n")
    print(full_context)
    
    # Example 5: Token budget tracking
    print("\n5. TOKEN BUDGET TRACKING")
    print("-"*80)
    
    max_budget = 1000
    tracking_window = ContextWindow(max_tokens=max_budget)
    
    print(f"Starting with budget: {max_budget} tokens")
    
    # Simulate adding context until near limit
    sample_contexts = [
        "Previous conversation about database design",
        "User mentioned preference for PostgreSQL over MySQL",
        "Discussed connection pooling strategies",
        "Reviewed indexing best practices",
        "Covered transaction management patterns",
    ]
    
    for i, ctx in enumerate(sample_contexts, 1):
        tokens = count_tokens(ctx)
        
        if tracking_window.current_tokens + tokens <= max_budget:
            item = ContextItem(
                content=ctx,
                priority=1.0,
                token_count=tokens
            )
            tracking_window.add_item(item)
            remaining = max_budget - tracking_window.current_tokens
            print(f"Added item {i}: {tokens} tokens (remaining: {remaining})")
        else:
            print(f"Item {i} would exceed budget ({tokens} tokens needed, "
                  f"{max_budget - tracking_window.current_tokens} remaining)")
            break
    
    # Example 6: Working with context metadata
    print("\n6. CONTEXT METADATA")
    print("-"*80)
    
    window_with_metadata = ContextWindow(
        max_tokens=2000,
        metadata={
            "session_id": "abc123",
            "user_id": "user456",
            "model": "gpt-4",
            "created_at": "2024-01-01T10:00:00Z"
        }
    )
    
    print("Window metadata:")
    for key, value in window_with_metadata.metadata.items():
        print(f"  {key}: {value}")
    
    # Example 7: Converting to dictionary for serialization
    print("\n7. SERIALIZATION")
    print("-"*80)
    
    simple_window = create_context_window([
        ContextItem(content="Example context 1", priority=1.0, token_count=5),
        ContextItem(content="Example context 2", priority=0.8, token_count=5),
    ])
    
    window_dict = simple_window.to_dict()
    print("Context window as dictionary:")
    print(f"  Items: {len(window_dict['items'])}")
    print(f"  Max tokens: {window_dict['max_tokens']}")
    print(f"  Current tokens: {window_dict['current_tokens']}")
    print(f"  Strategy: {window_dict['strategy']}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
