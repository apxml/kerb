"""Context Compression Example

This example demonstrates context compression techniques for fitting
large amounts of information into LLM token limits.

Main concepts:
- Different compression methods (summarize, extract, deduplicate, abbreviate)
- Automatic window compression
- Compression quality metrics
- Selective compression strategies
- Balancing compression with information retention
"""

from kerb.context import (
    ContextItem,
    ContextWindow,
    create_context_window,
    CompressionMethod,
)
from kerb.context.compression import compress_context, auto_compress_window
from kerb.tokenizer import count_tokens


def main():
    """Run context compression example."""
    
    print("="*80)
    print("CONTEXT COMPRESSION EXAMPLE")
    print("="*80)
    
    # Example 1: Basic compression with different methods
    print("\n1. BASIC COMPRESSION METHODS")
    print("-"*80)
    
    long_content = """
    Python is a high-level, interpreted programming language known for its 
    simplicity and readability. Python supports multiple programming paradigms 
    including procedural, object-oriented, and functional programming. Python 
    has a comprehensive standard library that includes modules for various tasks.
    Python is widely used in web development, data science, machine learning,
    automation, and scientific computing. The Python community is large and active,
    providing extensive resources and third-party packages through PyPI.
    Python's syntax emphasizes code readability with significant whitespace.
    """
    
    original_tokens = count_tokens(long_content)
    target = 30
    
    print(f"Original content: {original_tokens} tokens")
    print(f"Target: {target} tokens")
    
    # Try different compression methods
    methods = [
        CompressionMethod.SUMMARIZE,
        CompressionMethod.EXTRACT_KEY_INFO,
        CompressionMethod.REMOVE_REDUNDANCY,
        CompressionMethod.ABBREVIATE,
    ]
    
    for method in methods:
        result = compress_context(long_content, target, method=method)
        print(f"\n{method.value.upper()}:")
        print(f"  Compressed to {result.compressed_tokens} tokens "
              f"({result.compression_ratio:.1%} of original)")
        print(f"  Content: {result.compressed_content[:100]}...")
    
    # Example 2: Automatic window compression
    print("\n2. AUTOMATIC WINDOW COMPRESSION")
    print("-"*80)
    print("Use case: Compress entire context window to fit budget")
    
    # Create window with verbose documentation
    docs = [
        ContextItem(
            content="""The authentication module provides secure user login functionality.
            It includes password hashing with bcrypt, JWT token generation, session management,
            rate limiting to prevent brute force attacks, and email verification.""",
            priority=1.0,
            token_count=count_tokens("""The authentication module provides secure user login functionality.
            It includes password hashing with bcrypt, JWT token generation, session management,
            rate limiting to prevent brute force attacks, and email verification.""")
        ),
        ContextItem(
            content="""The database layer uses SQLAlchemy ORM for object-relational mapping.
            It supports connection pooling, automatic migrations with Alembic, query optimization,
            and provides models for User, Session, and Permission entities.""",
            priority=0.9,
            token_count=count_tokens("""The database layer uses SQLAlchemy ORM for object-relational mapping.
            It supports connection pooling, automatic migrations with Alembic, query optimization,
            and provides models for User, Session, and Permission entities.""")
        ),
        ContextItem(
            content="""API endpoints are implemented using Flask-RESTful framework.
            Each endpoint includes input validation, error handling, authentication checks,
            rate limiting, and comprehensive logging for debugging and monitoring.""",
            priority=0.8,
            token_count=count_tokens("""API endpoints are implemented using Flask-RESTful framework.
            Each endpoint includes input validation, error handling, authentication checks,
            rate limiting, and comprehensive logging for debugging and monitoring.""")
        ),
    ]
    
    doc_window = create_context_window(docs)
    print(f"Original window: {doc_window.current_tokens} tokens across {len(docs)} items")
    
    # Compress to 70% of original size
    compressed_window = auto_compress_window(doc_window, target_ratio=0.7)
    
    print(f"Compressed window: {compressed_window.current_tokens} tokens")
    print(f"Compression ratio: {compressed_window.current_tokens / doc_window.current_tokens:.1%}")
    print("\nCompressed items:")
    for i, item in enumerate(compressed_window.items, 1):
        print(f"\n  Item {i} ({item.token_count} tokens):")
        print(f"  {item.content[:80]}...")
    
    # Example 3: Selective compression
    print("\n3. SELECTIVE COMPRESSION")
    print("-"*80)
    print("Use case: Compress low-priority items, preserve high-priority")
    
    mixed_priority_items = [
        ContextItem(
            content="System prompt: You are an expert Python developer assistant.",
            priority=1.0,
            token_count=count_tokens("System prompt: You are an expert Python developer assistant.")
        ),
        ContextItem(
            content="""User's previous context includes discussion about Django REST framework,
            PostgreSQL database optimization, Redis caching strategies, Celery task queues,
            Docker containerization, and CI/CD pipeline configuration.""",
            priority=0.5,
            token_count=count_tokens("""User's previous context includes discussion about Django REST framework,
            PostgreSQL database optimization, Redis caching strategies, Celery task queues,
            Docker containerization, and CI/CD pipeline configuration.""")
        ),
        ContextItem(
            content="Current task: Debug authentication error in production API.",
            priority=1.0,
            token_count=count_tokens("Current task: Debug authentication error in production API.")
        ),
        ContextItem(
            content="""Background information: The application uses microservices architecture
            with service mesh, implements OAuth2 for authentication, uses GraphQL for some endpoints,
            has monitoring with Prometheus and Grafana, and deploys to Kubernetes clusters.""",
            priority=0.4,
            token_count=count_tokens("""Background information: The application uses microservices architecture
            with service mesh, implements OAuth2 for authentication, uses GraphQL for some endpoints,
            has monitoring with Prometheus and Grafana, and deploys to Kubernetes clusters.""")
        ),
    ]
    
    window = create_context_window(mixed_priority_items)
    print(f"Original: {window.current_tokens} tokens")
    
    # Compress only low-priority items
    for item in window.items:
        if item.priority < 0.8:
            target_tokens = int(item.token_count * 0.5)  # 50% compression
            result = compress_context(item.content, target_tokens, CompressionMethod.SUMMARIZE)
            item.content = result.compressed_content
            item.token_count = result.compressed_tokens
            print(f"\nCompressed item (priority {item.priority}):")
            print(f"  {result.original_tokens} -> {result.compressed_tokens} tokens")
        else:
            print(f"\nPreserved high-priority item (priority {item.priority})")
    
    # Recalculate total
    window.current_tokens = sum(item.token_count for item in window.items)
    print(f"\nFinal: {window.current_tokens} tokens")
    
    # Example 4: Compression for code context
    print("\n4. CODE CONTEXT COMPRESSION")
    print("-"*80)
    print("Use case: Compress code documentation while preserving structure")
    
    code_doc = """
    Function: authenticate_user(username: str, password: str) -> Optional[User]
    
    Description:
    Authenticates a user by verifying their username and password credentials.
    This function handles the complete authentication flow including password
    validation, account status checking, and session creation.
    
    Parameters:
    - username: The user's unique username (string, required)
    - password: The user's plaintext password (string, required)
    
    Returns:
    - User object if authentication succeeds
    - None if authentication fails
    
    Raises:
    - AuthenticationError: If credentials are invalid
    - AccountLockedException: If account is locked due to failed attempts
    - DatabaseError: If database connection fails
    
    Example:
    user = authenticate_user("john_doe", "secret_password")
    if user:
        create_session(user)
    """
    
    code_tokens = count_tokens(code_doc)
    print(f"Original code documentation: {code_tokens} tokens")
    
    # Compress while keeping key information
    compressed_code = compress_context(code_doc, target_tokens=50, method=CompressionMethod.EXTRACT_KEY_INFO)
    
    print(f"Compressed to: {compressed_code.compressed_tokens} tokens "
          f"({compressed_code.compression_ratio:.1%})")
    print(f"\nCompressed content:\n{compressed_code.compressed_content}")
    
    # Example 5: Compression metrics and quality
    print("\n5. COMPRESSION METRICS")
    print("-"*80)
    
    test_content = """
    Machine learning models require careful hyperparameter tuning for optimal performance.
    Common hyperparameters include learning rate, batch size, number of epochs, and regularization.
    Cross-validation helps prevent overfitting by testing on held-out data. Feature engineering
    often has more impact than model choice. Data quality and quantity matter significantly.
    """
    
    targets = [20, 30, 40]
    
    print("Compression at different target sizes:")
    print("\nTarget | Actual | Ratio | Method")
    print("-" * 60)
    
    for target in targets:
        result = compress_context(test_content, target, CompressionMethod.SUMMARIZE)
        print(f"{target:6} | {result.compressed_tokens:6} | {result.compression_ratio:5.1%} | "
              f"{result.method.value}")
    
    # Example 6: Compression strategies comparison
    print("\n6. STRATEGY COMPARISON")
    print("-"*80)
    
    strategies = {
        "AGGRESSIVE": (0.3, "Maximum compression, may lose details"),
        "MODERATE": (0.6, "Balanced compression and information retention"),
        "CONSERVATIVE": (0.8, "Minimal compression, preserve most information"),
    }
    
    sample = """
    The React component lifecycle includes mounting, updating, and unmounting phases.
    UseEffect hook manages side effects in functional components. State management
    can be handled with useState, useReducer, or external libraries like Redux.
    Component composition and props enable reusable UI elements.
    """
    
    sample_tokens = count_tokens(sample)
    
    print(f"Original: {sample_tokens} tokens\n")
    for strategy_name, (ratio, description) in strategies.items():
        target = int(sample_tokens * ratio)
        result = compress_context(sample, target, CompressionMethod.SUMMARIZE)
        print(f"{strategy_name} ({description}):")
        print(f"  Target ratio: {ratio:.0%}, Actual: {result.compression_ratio:.1%}")
        print(f"  Result: {result.compressed_content[:80]}...")
        print()
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
