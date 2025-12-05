"""
Context Formatting Example
==========================

This example demonstrates formatting context for LLM consumption,
including conversion to chat messages and various output formats.

Main concepts:
- Formatting context windows for LLM input
- Converting context to chat message format
- Extracting context summaries
- Custom formatting templates
- Role-based message formatting
"""

from kerb.context import (
    ContextItem,
    ContextWindow,
    create_context_window,
    format_context_window,
    context_to_messages,
    extract_context_summary,
)
from kerb.tokenizer import count_tokens


def main():
    """Run context formatting example."""
    
    print("="*80)
    print("CONTEXT FORMATTING EXAMPLE")
    print("="*80)
    
    # Example 1: Basic context formatting
    print("\n1. BASIC CONTEXT FORMATTING")
    print("-"*80)
    print("Use case: Format context for single LLM prompt")
    
    items = [
        ContextItem(
            content="User is building a web application",
            token_count=count_tokens("User is building a web application"),
            metadata={"type": "context"}
        ),
        ContextItem(
            content="Tech stack: Python, Flask, PostgreSQL",
            token_count=count_tokens("Tech stack: Python, Flask, PostgreSQL"),
            metadata={"type": "info"}
        ),
        ContextItem(
            content="Current task: Implement user authentication",
            token_count=count_tokens("Current task: Implement user authentication"),
            metadata={"type": "task"}
        ),
    ]
    
    window = create_context_window(items)
    
    # Default formatting (simple concatenation)
    formatted = format_context_window(window)
    print("\nDefault format:")
    print(formatted)
    
    # Formatted with custom template
    template = "{content}"
    formatted_custom = format_context_window(window, format_template=template)
    print("\nWith custom template:")
    print(formatted_custom)
    
    # Example 2: Converting to chat messages
    print("\n2. CHAT MESSAGE FORMAT")
    print("-"*80)
    print("Use case: Format context for chat-based LLMs")
    
    conversation_items = [
        ContextItem(
            content="You are a helpful Python coding assistant.",
            token_count=count_tokens("You are a helpful Python coding assistant."),
            metadata={"role": "system"}
        ),
        ContextItem(
            content="How do I handle database connections in Flask?",
            token_count=count_tokens("How do I handle database connections in Flask?"),
            metadata={"role": "user"}
        ),
        ContextItem(
            content="You can use Flask-SQLAlchemy for database connections...",
            token_count=count_tokens("You can use Flask-SQLAlchemy for database connections..."),
            metadata={"role": "assistant"}
        ),
        ContextItem(
            content="Can you show me an example?",
            token_count=count_tokens("Can you show me an example?"),
            metadata={"role": "user"}
        ),
    ]
    
    conv_window = create_context_window(conversation_items)
    
    # Convert to message format
    messages = context_to_messages(conv_window)
    
    print("\nChat messages format:")
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"\n[{role.upper()}]")
        print(f"{content[:80]}..." if len(content) > 80 else content)
    
    # Example 3: Context summary extraction
    print("\n3. CONTEXT SUMMARY EXTRACTION")
    print("-"*80)
    print("Use case: Generate concise summary of context")
    
    detailed_context = [
        ContextItem(
            content="Project: E-commerce platform for selling handmade crafts",
            token_count=count_tokens("Project: E-commerce platform for selling handmade crafts")
        ),
        ContextItem(
            content="Features: Product catalog, shopping cart, payment processing, user reviews",
            token_count=count_tokens("Features: Product catalog, shopping cart, payment processing, user reviews")
        ),
        ContextItem(
            content="Tech stack: Django backend, React frontend, PostgreSQL database, Redis cache",
            token_count=count_tokens("Tech stack: Django backend, React frontend, PostgreSQL database, Redis cache")
        ),
        ContextItem(
            content="Current sprint: Implementing payment gateway integration with Stripe",
            token_count=count_tokens("Current sprint: Implementing payment gateway integration with Stripe")
        ),
        ContextItem(
            content="Challenges: Handling payment failures, webhook processing, transaction logging",
            token_count=count_tokens("Challenges: Handling payment failures, webhook processing, transaction logging")
        ),
    ]
    
    detail_window = create_context_window(detailed_context)
    
    # Extract summary
    summary = extract_context_summary(detail_window)
    
    print(f"\nOriginal context: {detail_window.current_tokens} tokens")
    print(f"\nSummary:\n{summary}")
    print(f"\nSummary tokens: {count_tokens(summary)}")
    
    # Example 4: Formatted context with metadata
    print("\n4. FORMATTED CONTEXT WITH METADATA")
    print("-"*80)
    print("Use case: Include metadata in formatted output")
    
    code_context = [
        ContextItem(
            content="def authenticate_user(username, password): ...",
            token_count=count_tokens("def authenticate_user(username, password): ..."),
            metadata={
                "type": "function",
                "file": "auth.py",
                "line": 45,
                "importance": "high"
            }
        ),
        ContextItem(
            content="class UserSession: ...",
            token_count=count_tokens("class UserSession: ..."),
            metadata={
                "type": "class",
                "file": "session.py",
                "line": 12,
                "importance": "medium"
            }
        ),
    ]
    
    code_window = create_context_window(code_context)
    
    print("\nCode context with metadata:")
    for item in code_window.items:
        meta = item.metadata
        print(f"\n[{meta.get('type', 'unknown').upper()}] "
              f"{meta.get('file', 'unknown')}:{meta.get('line', '?')}")
        print(f"Importance: {meta.get('importance', 'unknown')}")
        print(f"Content: {item.content}")
    
    # Example 5: Custom formatting templates
    print("\n5. CUSTOM FORMATTING TEMPLATES")
    print("-"*80)
    print("Use case: Apply specific format for different LLM providers")
    
    def format_for_instruction_model(window: ContextWindow) -> str:
        """Format context for instruction-following models."""

# %%
# Setup and Imports
# -----------------
        parts = []
        
        # Add instruction header
        parts.append("### Context Information\n")
        
        # Add each item with numbering
        for i, item in enumerate(window.items, 1):
            item_type = item.metadata.get("type", "info")
            parts.append(f"{i}. [{item_type.upper()}] {item.content}")
        
        # Add instruction footer
        parts.append("\n### Instructions")
        parts.append("Use the above context to answer the following question:")
        
        return "\n".join(parts)
    

# %%
# Format For Chat Model
# ---------------------

    def format_for_chat_model(window: ContextWindow) -> list:
        """Format context for chat-based models."""
        messages = []
        
        # System message with context
        context_parts = []
        for item in window.items:
            if item.metadata.get("role") == "system" or item.metadata.get("type") == "system":
                context_parts.append(item.content)
        
        if context_parts:
            messages.append({
                "role": "system",
                "content": "\n\n".join(context_parts)
            })
        
        # User messages
        for item in window.items:
            role = item.metadata.get("role")
            if role and role != "system":
                messages.append({
                    "role": role,
                    "content": item.content
                })
        
        return messages
    
    sample_items = [
        ContextItem(
            content="Database uses PostgreSQL 14",
            token_count=10,
            metadata={"type": "system"}
        ),
        ContextItem(
            content="Current schema includes users and posts tables",
            token_count=10,
            metadata={"type": "info"}
        ),
    ]
    
    sample_window = create_context_window(sample_items)
    
    print("\nInstruction model format:")
    print(format_for_instruction_model(sample_window))
    
    print("\n\nChat model format:")
    chat_msgs = format_for_chat_model(sample_window)
    for msg in chat_msgs:
        print(f"[{msg['role']}] {msg['content']}")
    
    # Example 6: Formatting for different use cases
    print("\n6. USE-CASE SPECIFIC FORMATTING")
    print("-"*80)
    
    # Code review format
    def format_for_code_review(window: ContextWindow) -> str:
        """Format context for code review."""
        sections = {
            "files": [],
            "changes": [],
            "concerns": []
        }
        
        for item in window.items:
            item_type = item.metadata.get("type", "other")
            if item_type in sections:
                sections[item_type].append(item.content)
        
        output = []
        output.append("CODE REVIEW CONTEXT")
        output.append("=" * 40)
        
        if sections["files"]:
            output.append("\nFiles Changed:")
            for f in sections["files"]:
                output.append(f"  - {f}")
        
        if sections["changes"]:
            output.append("\nChanges Summary:")
            for c in sections["changes"]:
                output.append(f"  - {c}")
        
        if sections["concerns"]:
            output.append("\nReview Concerns:")
            for concern in sections["concerns"]:
                output.append(f"  - {concern}")
        
        return "\n".join(output)
    
    review_items = [
        ContextItem(
            content="auth.py",
            token_count=5,
            metadata={"type": "files"}
        ),
        ContextItem(
            content="Added password strength validation",
            token_count=10,
            metadata={"type": "changes"}
        ),
        ContextItem(
            content="Check for SQL injection vulnerabilities",
            token_count=10,
            metadata={"type": "concerns"}
        ),
    ]
    
    review_window = create_context_window(review_items)
    
    print("\nCode review format:")
    print(format_for_code_review(review_window))
    
    # Documentation format

# %%
# Format For Documentation
# ------------------------

    def format_for_documentation(window: ContextWindow) -> str:
        """Format context for documentation generation."""
        output = []
        
        for item in window.items:
            doc_type = item.metadata.get("doc_type", "section")
            
            if doc_type == "title":
                output.append(f"# {item.content}\n")
            elif doc_type == "section":
                output.append(f"## {item.content}\n")
            elif doc_type == "code":
                output.append(f"```python\n{item.content}\n```\n")
            else:
                output.append(f"{item.content}\n")
        
        return "\n".join(output)
    
    doc_items = [
        ContextItem(
            content="Authentication Module",
            token_count=5,
            metadata={"doc_type": "title"}
        ),
        ContextItem(
            content="Overview",
            token_count=5,
            metadata={"doc_type": "section"}
        ),
        ContextItem(
            content="This module provides user authentication functionality.",
            token_count=10,
            metadata={"doc_type": "text"}
        ),
    ]
    
    doc_window = create_context_window(doc_items)
    
    print("\nDocumentation format:")
    print(format_for_documentation(doc_window))
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
