"""
Conversation Summary Example
============================

This example demonstrates how to create and manage conversation summaries
for long-running conversations in LLM applications.

Main concepts:
- Progressive summaries that build on previous context
- Hierarchical summaries for different conversation segments
- Comprehensive conversation summaries with key points
- Using summaries to reduce token usage
"""

from kerb.memory import (
    ConversationBuffer,
    create_progressive_summary,
    summarize_conversation,
    create_hierarchical_summary
)
from kerb.core.types import Message


def main():
    """Run conversation summary example."""
    
    print("="*80)
    print("CONVERSATION SUMMARY EXAMPLE")
    print("="*80)
    
    # Simulate a long technical discussion
    messages = [
        Message("user", "I'm building a microservices architecture for an e-commerce platform. Where should I start?"),
        Message("assistant", "Start by identifying your core business domains: user management, product catalog, orders, payments, and inventory. Each should be its own service."),
        Message("user", "How do I handle communication between services?"),
        Message("assistant", "Use asynchronous messaging with a message broker like RabbitMQ or Kafka for event-driven communication. For synchronous calls, REST or gRPC work well."),
        Message("user", "What about data consistency across services?"),
        Message("assistant", "Implement the Saga pattern for distributed transactions. Each service maintains its own database, and you coordinate changes through events."),
        Message("user", "That sounds complex. Can you explain the Saga pattern more?"),
        Message("assistant", "The Saga pattern breaks a transaction into smaller steps. Each step has a compensating action. If a step fails, you execute compensating actions to rollback."),
        Message("user", "What technologies should I use for the API gateway?"),
        Message("assistant", "Popular choices include Kong, NGINX, or AWS API Gateway. They handle routing, authentication, rate limiting, and load balancing."),
        Message("user", "How do I monitor all these services?"),
        Message("assistant", "Use distributed tracing with Jaeger or Zipkin, centralized logging with ELK stack, and metrics with Prometheus and Grafana."),
        Message("user", "What about service discovery?"),
        Message("assistant", "Consul or Eureka for service discovery. Kubernetes has built-in service discovery if you're using it for orchestration."),
        Message("user", "Should I use Kubernetes from the start?"),
        Message("assistant", "Start simple with Docker Compose. Move to Kubernetes when you need auto-scaling, self-healing, and complex orchestration."),
        Message("user", "How do I handle authentication across services?"),
        Message("assistant", "Use JWT tokens with OAuth 2.0. The API gateway validates tokens, and services trust the gateway's authentication."),
        Message("user", "What's the best way to test microservices?"),
        Message("assistant", "Combine unit tests, integration tests with test containers, and contract testing with Pact. Also implement chaos engineering for resilience."),
    ]
    
    print(f"\nSimulated conversation with {len(messages)} messages")
    
    # Strategy 1: Progressive Summary
    print("\n" + "-"*80)
    print("STRATEGY 1: PROGRESSIVE SUMMARY")
    print("-"*80)
    
    # Build summary progressively as conversation grows
    summaries = []
    
    # Summarize first 4 messages
    chunk1 = messages[:4]
    summary1 = create_progressive_summary(chunk1, summary_length="short")
    summaries.append(summary1)
    print(f"\nSummary after 4 messages (short):")
    print(f"  {summary1}")
    
    # Add next 4 messages and update summary
    chunk2 = messages[4:8]
    summary2 = create_progressive_summary(chunk2, existing_summary=summary1, summary_length="medium")
    summaries.append(summary2)
    print(f"\nSummary after 8 messages (medium):")
    print(f"  {summary2}")
    
    # Add remaining messages
    chunk3 = messages[8:]
    summary3 = create_progressive_summary(chunk3, existing_summary=summary2, summary_length="long")
    summaries.append(summary3)
    print(f"\nSummary after all messages (long):")
    print(f"  {summary3[:300]}...")
    
    # Strategy 2: Comprehensive Summary
    print("\n" + "-"*80)
    print("STRATEGY 2: COMPREHENSIVE SUMMARY")
    print("-"*80)
    
    # Create detailed summary with metadata
    comprehensive = summarize_conversation(messages, summary_strategy="extractive", key_points=5)
    
    print(f"\nComprehensive Summary:")
    print(f"  Message count: {comprehensive.message_count}")
    print(f"  Time range: {comprehensive.start_time} to {comprehensive.end_time}")
    print(f"  Summary: {comprehensive.summary[:200]}...")
    
    if comprehensive.key_points:
        print(f"\n  Key Points ({len(comprehensive.key_points)}):")
        for i, point in enumerate(comprehensive.key_points, 1):
            print(f"    {i}. {point[:80]}...")
    
    if comprehensive.entities:
        print(f"\n  Entities Mentioned ({len(comprehensive.entities)}):")
        print(f"    {', '.join(comprehensive.entities[:10])}")
    
    # Strategy 3: Hierarchical Summary
    print("\n" + "-"*80)
    print("STRATEGY 3: HIERARCHICAL SUMMARY")
    print("-"*80)
    
    # Break conversation into chunks and summarize each
    hierarchical = create_hierarchical_summary(messages, chunk_size=5)
    
    print(f"\nCreated {len(hierarchical)} hierarchical summaries (5 messages each):")
    for i, summary in enumerate(hierarchical, 1):
        print(f"\n  Chunk {i} ({summary.message_count} messages):")
        print(f"    {summary.summary[:150]}...")
    
    # Real-world application: ConversationBuffer with automatic summarization
    print("\n" + "-"*80)
    print("REAL-WORLD: BUFFER WITH AUTO-SUMMARIZATION")
    print("-"*80)
    
    # Create buffer that automatically summarizes when pruning
    buffer = ConversationBuffer(
        max_messages=10,  # Small limit to trigger summarization
        enable_summaries=True
    )
    
    # Add all messages (will trigger summarization)
    print(f"\nAdding {len(messages)} messages to buffer with max_messages=10:")
    for msg in messages:
        buffer.add_message(msg.role, msg.content)
    
    print(f"  Messages in buffer: {len(buffer.messages)}")
    print(f"  Summaries created: {len(buffer.summaries)}")
    
    # Show summaries
    if buffer.summaries:
        print(f"\n  Summaries of pruned messages:")
        for i, summary in enumerate(buffer.summaries, 1):
            print(f"\n    Summary {i}:")
            print(f"      Messages: {summary.message_count}")
            print(f"      Content: {summary.summary[:120]}...")
    
    # Get context with summaries
    print("\n" + "-"*80)
    print("CONTEXT WITH SUMMARIES")
    print("-"*80)
    
    context = buffer.get_context(include_summary=True)
    print(f"\nContext for LLM (includes summaries + recent messages):")
    print(f"{context[:400]}...")
    
    # Use case: Reducing token usage with summaries
    print("\n" + "-"*80)
    print("USE CASE: TOKEN REDUCTION")
    print("-"*80)
    
    def estimate_tokens(text: str) -> int:
        """Estimate token count."""

# %%
# Setup and Imports
# -----------------
        return int(len(text.split()) / 0.75)
    
    # Full conversation tokens
    full_context = "\n".join(f"{m.role}: {m.content}" for m in messages)
    full_tokens = estimate_tokens(full_context)
    
    # Context with summaries
    summarized_context = buffer.get_context(include_summary=True)
    summarized_tokens = estimate_tokens(summarized_context)
    
    reduction = ((full_tokens - summarized_tokens) / full_tokens) * 100
    
    print(f"\nToken usage comparison:")
    print(f"  Full conversation: ~{full_tokens} tokens")
    print(f"  With summaries: ~{summarized_tokens} tokens")
    print(f"  Reduction: {reduction:.1f}%")
    
    # Multi-turn summary updates
    print("\n" + "-"*80)
    print("MULTI-TURN SUMMARY UPDATES")
    print("-"*80)
    
    # Simulate building summary across multiple conversation turns
    print("\nBuilding cumulative summary across conversation:")
    
    cumulative_summary = ""
    chunk_size = 6
    
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i:i + chunk_size]
        cumulative_summary = create_progressive_summary(
            chunk,
            existing_summary=cumulative_summary,
            summary_length="medium"
        )
        print(f"\n  After message {min(i + chunk_size, len(messages))}:")
        print(f"    {cumulative_summary[:150]}...")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
