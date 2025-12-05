"""
Basic Conversation Buffer Example
=================================

=================================

This example demonstrates the core functionality of ConversationBuffer for managing
conversation history in LLM applications.

Main concepts:
- Creating a ConversationBuffer instance
- Adding messages to the buffer
- Retrieving recent messages for context
- Managing buffer limits and automatic pruning
- Accessing conversation metadata
"""

from kerb.memory import ConversationBuffer
from kerb.core.types import Message


def main():
    """Run basic conversation buffer example."""
    
    print("="*80)
    print("BASIC CONVERSATION BUFFER EXAMPLE")
    print("="*80)
    
    # Create a conversation buffer with size limits
    buffer = ConversationBuffer(
        max_messages=100,      # Store up to 100 messages
        window_size=10,        # Default window size for recent context
        enable_summaries=True, # Create summaries when pruning
        enable_entity_tracking=True  # Track entities mentioned
    )
    
    print("\nCreated ConversationBuffer:")
    print(f"  Max messages: {buffer.max_messages}")
    print(f"  Window size: {buffer.window_size}")
    print(f"  Summaries enabled: {buffer.enable_summaries}")
    print(f"  Entity tracking enabled: {buffer.enable_entity_tracking}")
    
    # Simulate a multi-turn conversation
    print("\n" + "-"*80)
    print("SIMULATING CONVERSATION")
    print("-"*80)
    
    # System message
    buffer.add_message(
        "system",
        "You are a helpful AI assistant specialized in Python programming."
    )
    
    # User asks about async
    buffer.add_message(
        "user",
        "Can you explain async/await in Python? I'm working on a web scraper."
    )
    
    buffer.add_message(
        "assistant",
        "async/await in Python allows you to write asynchronous code. The 'async def' "
        "keyword defines a coroutine, and 'await' pauses execution until the awaited "
        "operation completes. For web scraping, this is great for making concurrent HTTP requests."
    )
    
    buffer.add_message(
        "user",
        "What libraries would you recommend for async web scraping?"
    )
    
    buffer.add_message(
        "assistant",
        "I recommend using 'aiohttp' for async HTTP requests and 'asyncio' for managing "
        "concurrent tasks. For parsing HTML, 'beautifulsoup4' works well with async code. "
        "You can also check out 'httpx' which supports both sync and async."
    )
    
    buffer.add_message(
        "user",
        "How do I handle rate limiting when scraping multiple sites?"
    )
    
    buffer.add_message(
        "assistant",
        "Use asyncio.Semaphore to limit concurrent requests. You can also implement "
        "exponential backoff with asyncio.sleep(). Consider using libraries like "
        "'aiolimiter' for more sophisticated rate limiting strategies."
    )
    
    # Display buffer state
    print(f"\nTotal messages in buffer: {len(buffer.messages)}")
    
    # Get recent context
    print("\n" + "-"*80)
    print("RETRIEVING RECENT CONTEXT")
    print("-"*80)
    
    # Get last 5 messages
    recent = buffer.get_recent_messages(count=5)
    print(f"\nLast 5 messages:")
    for i, msg in enumerate(recent, 1):
        print(f"\n  [{i}] {msg.role}:")
        print(f"      {msg.content[:80]}...")
    
    # Get formatted context for LLM
    print("\n" + "-"*80)
    print("FORMATTED CONTEXT FOR LLM")
    print("-"*80)
    
    context = buffer.get_context(include_summary=False)
    print(f"\n{context[:400]}...")
    
    # Search functionality
    print("\n" + "-"*80)
    print("SEARCHING MESSAGES")
    print("-"*80)
    
    search_results = buffer.search_messages("asyncio", max_results=3)
    print(f"\nFound {len(search_results)} messages containing 'asyncio':")
    for msg in search_results:
        print(f"  - {msg.role}: {msg.content[:60]}...")
    
    # Test automatic pruning by adding many messages
    print("\n" + "-"*80)
    print("TESTING AUTOMATIC PRUNING")
    print("-"*80)
    
    # Create a small buffer to demonstrate pruning
    small_buffer = ConversationBuffer(
        max_messages=5,
        enable_summaries=True
    )
    
    print("\nAdding 8 messages to a buffer with max_messages=5:")
    for i in range(8):
        small_buffer.add_message("user", f"Message {i+1}")
    
    print(f"Messages stored: {len(small_buffer.messages)} (pruned to {small_buffer.max_messages})")
    print(f"Summaries created: {len(small_buffer.summaries)}")
    
    if small_buffer.summaries:
        print(f"\nSummary of pruned messages:")
        print(f"  {small_buffer.summaries[0].summary}")
        print(f"  Message count: {small_buffer.summaries[0].message_count}")
    
    # Clear buffer
    print("\n" + "-"*80)
    print("CLEARING BUFFER")
    print("-"*80)
    
    messages_before = len(buffer.messages)
    buffer.clear(keep_summaries=True)
    
    print(f"\nMessages before clear: {messages_before}")
    print(f"Messages after clear: {len(buffer.messages)}")
    print(f"Summaries preserved: {len(buffer.summaries)}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
