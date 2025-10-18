"""Semantic Memory Patterns Example

This example demonstrates advanced memory patterns including semantic grouping
and episodic memory for better context organization in LLM applications.

Main concepts:
- Semantic memory: grouping similar conversations
- Episodic memory: time-based conversation segments
- Memory retrieval based on relevance
- Pattern-based memory organization
"""

from kerb.memory import ConversationBuffer
from kerb.memory.patterns import (
    create_semantic_memory,
    create_episodic_memory,
    get_relevant_memory
)
from kerb.core.types import Message


def main():
    """Run semantic memory patterns example."""
    
    print("="*80)
    print("SEMANTIC MEMORY PATTERNS EXAMPLE")
    print("="*80)
    
    # Create conversation buffer
    buffer = ConversationBuffer()
    
    # Simulate a diverse technical conversation
    conversation = [
        ("user", "How do I optimize database queries in PostgreSQL?"),
        ("assistant", "Use indexes, EXPLAIN ANALYZE, and optimize joins. Consider query planning and statistics."),
        ("user", "What's the best way to handle authentication in REST APIs?"),
        ("assistant", "Use JWT tokens with OAuth 2.0. Implement refresh tokens and secure token storage."),
        ("user", "Can you explain database connection pooling?"),
        ("assistant", "Connection pooling reuses database connections to reduce overhead. Use pgBouncer for PostgreSQL."),
        ("user", "How should I implement rate limiting for my API?"),
        ("assistant", "Use token bucket or sliding window algorithms. Implement at API gateway level with Redis for distributed systems."),
        ("user", "What are database indexes and when should I use them?"),
        ("assistant", "Indexes speed up queries but slow down writes. Use B-tree for equality, GiST for full-text, and covering indexes for specific queries."),
        ("user", "How do I secure API endpoints?"),
        ("assistant", "Implement authentication, authorization, input validation, rate limiting, and HTTPS. Use API keys or OAuth."),
        ("user", "What's the difference between INNER JOIN and LEFT JOIN?"),
        ("assistant", "INNER JOIN returns matching rows from both tables. LEFT JOIN returns all rows from left table plus matches from right."),
        ("user", "How do I implement OAuth 2.0 flows?"),
        ("assistant", "OAuth has authorization code, implicit, client credentials, and password flows. Use authorization code flow for web apps."),
    ]
    
    for role, content in conversation:
        buffer.add_message(role, content)
    
    print(f"Created conversation with {len(buffer.messages)} messages")
    
    # Pattern 1: Semantic Memory Grouping
    print("\n" + "-"*80)
    print("PATTERN 1: SEMANTIC MEMORY GROUPING")
    print("-"*80)
    
    # Group similar messages together
    semantic_groups = create_semantic_memory(buffer.messages)
    
    print(f"\nGrouped {len(buffer.messages)} messages into {len(semantic_groups)} semantic groups:")
    
    for i, (group_key, messages) in enumerate(semantic_groups.items(), 1):
        print(f"\n  Group {i}: {group_key[:50]}...")
        print(f"  Messages: {len(messages)}")
        for msg in messages[:2]:  # Show first 2
            print(f"    - {msg.role}: {msg.content[:60]}...")
    
    # Pattern 2: Episodic Memory
    print("\n" + "-"*80)
    print("PATTERN 2: EPISODIC MEMORY")
    print("-"*80)
    
    # Split conversation into episodes
    episodes = create_episodic_memory(buffer.messages, episode_duration=4)
    
    print(f"\nSplit conversation into {len(episodes)} episodes (4 messages each):")
    
    for i, episode in enumerate(episodes, 1):
        print(f"\n  Episode {i} ({len(episode)} messages):")
        # Show topic of episode
        user_msgs = [m for m in episode if m.role == "user"]
        if user_msgs:
            print(f"    Topic: {user_msgs[0].content[:70]}...")
        print(f"    Messages: {[m.role for m in episode]}")
    
    # Pattern 3: Relevant Memory Retrieval
    print("\n" + "-"*80)
    print("PATTERN 3: RELEVANT MEMORY RETRIEVAL")
    print("-"*80)
    
    # Retrieve memories relevant to specific queries
    queries = [
        "database performance",
        "API security",
        "OAuth authentication"
    ]
    
    for query in queries:
        relevant = get_relevant_memory(query, buffer, top_k=3)
        print(f"\nQuery: '{query}'")
        print(f"Found {len(relevant)} relevant memories:")
        for msg in relevant:
            print(f"  - {msg.role}: {msg.content[:70]}...")
    
    # Pattern 4: Hybrid Memory System
    print("\n" + "-"*80)
    print("PATTERN 4: HYBRID MEMORY SYSTEM")
    print("-"*80)
    
    # Combine episodic and semantic memories
    print("\nBuilding hybrid memory system:")
    
    # Create episodic structure
    episodic_structure = {}
    for i, episode in enumerate(episodes):
        episode_key = f"episode_{i+1}"
        # Group episode messages semantically
        episode_groups = create_semantic_memory(episode)
        episodic_structure[episode_key] = episode_groups
    
    print(f"\nHybrid structure:")
    for episode_key, groups in episodic_structure.items():
        print(f"  {episode_key}: {len(groups)} semantic group(s)")
        for group_key in list(groups.keys())[:2]:  # Show first 2 groups
            print(f"    - {group_key[:40]}...")
    
    # Pattern 5: Context-Aware Memory Retrieval
    print("\n" + "-"*80)
    print("PATTERN 5: CONTEXT-AWARE RETRIEVAL")
    print("-"*80)
    
    # Build context for a new question
    new_query = "How do I improve database performance?"
    
    print(f"\nNew query: '{new_query}'")
    print("Building context-aware memory...")
    
    # Step 1: Get semantically relevant messages
    relevant_messages = get_relevant_memory(new_query, buffer, top_k=4)
    
    # Step 2: Get recent context (last episode)
    recent_episode = episodes[-1] if episodes else []
    
    # Step 3: Combine for comprehensive context
    context_messages = []
    
    # Add relevant historical context
    for msg in relevant_messages:
        if msg not in context_messages:
            context_messages.append(msg)
    
    # Add recent context
    for msg in recent_episode:
        if msg not in context_messages:
            context_messages.append(msg)
    
    print(f"\nContext built with {len(context_messages)} messages:")
    print("  [Relevant historical + Recent episode]")
    for i, msg in enumerate(context_messages, 1):
        marker = "[H]" if msg in relevant_messages else "[R]"
        print(f"  [{i}] {marker} {msg.role}: {msg.content[:60]}...")
    
    # Pattern 6: Topic-Based Memory Organization
    print("\n" + "-"*80)
    print("PATTERN 6: TOPIC-BASED ORGANIZATION")
    print("-"*80)
    
    # Organize messages by detected topics
    topics = {
        "database": ["database", "postgresql", "query", "index", "join"],
        "api": ["api", "rest", "endpoint", "rate limiting"],
        "security": ["authentication", "oauth", "security", "token", "jwt"],
    }
    
    topic_memories = {topic: [] for topic in topics}
    
    for msg in buffer.messages:
        content_lower = msg.content.lower()
        for topic, keywords in topics.items():
            if any(keyword in content_lower for keyword in keywords):
                topic_memories[topic].append(msg)
    
    print("\nOrganized by topics:")
    for topic, messages in topic_memories.items():
        print(f"\n  {topic.upper()} ({len(messages)} messages):")
        for msg in messages[:2]:  # Show first 2
            print(f"    - {msg.content[:70]}...")
    
    # Real-world application: Smart Context Building
    print("\n" + "-"*80)
    print("APPLICATION: SMART CONTEXT BUILDING")
    print("-"*80)
    
    def build_smart_context(query: str, buffer: ConversationBuffer, max_messages: int = 5):
        """Build smart context using multiple memory patterns."""
        
        # 1. Get relevant memories
        relevant = get_relevant_memory(query, buffer, top_k=3)
        
        # 2. Get recent context
        recent = buffer.get_recent_messages(count=2)
        
        # 3. Combine without duplicates
        context = []
        for msg in relevant + recent:
            if msg not in context:
                context.append(msg)
        
        # 4. Limit to max_messages
        context = context[:max_messages]
        
        return context
    
    # Test smart context building
    test_query = "How to secure my database API?"
    smart_context = build_smart_context(test_query, buffer, max_messages=5)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Smart context ({len(smart_context)} messages):")
    
    for i, msg in enumerate(smart_context, 1):
        print(f"  [{i}] {msg.role}: {msg.content[:70]}...")
    
    # Pattern 7: Memory Importance Scoring
    print("\n" + "-"*80)
    print("PATTERN 7: MEMORY IMPORTANCE SCORING")
    print("-"*80)
    
    # Score messages by importance
    def score_message(msg: Message) -> int:
        """Simple importance scoring."""
        score = 0
        
        # Questions are important
        if msg.role == "user" and "?" in msg.content:
            score += 3
        
        # Longer messages might be more detailed
        if len(msg.content) > 100:
            score += 2
        
        # Messages with technical terms
        technical_terms = ["database", "api", "oauth", "security", "performance"]
        score += sum(1 for term in technical_terms if term in msg.content.lower())
        
        return score
    
    scored_messages = [(score_message(msg), msg) for msg in buffer.messages]
    scored_messages.sort(key=lambda x: x[0], reverse=True)
    
    print("\nTop 5 most important messages:")
    for i, (score, msg) in enumerate(scored_messages[:5], 1):
        print(f"  [{i}] Score: {score} - {msg.role}: {msg.content[:60]}...")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
