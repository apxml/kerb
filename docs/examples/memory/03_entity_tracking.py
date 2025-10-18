"""Entity Tracking Example

This example demonstrates entity extraction and tracking across conversations
for maintaining context awareness in LLM applications.

Main concepts:
- Extracting entities from conversation messages
- Tracking entity mentions over time
- Using entities for context enrichment
- Entity-based conversation analysis
"""

from kerb.memory import ConversationBuffer, Entity
from kerb.memory.entities import (
    extract_entities,
    track_entity_mentions,
    merge_entities
)
from kerb.core.types import Message


def main():
    """Run entity tracking example."""
    
    print("="*80)
    print("ENTITY TRACKING EXAMPLE")
    print("="*80)
    
    # Create buffer with entity tracking enabled
    buffer = ConversationBuffer(
        enable_entity_tracking=True,
        enable_summaries=False
    )
    
    # Simulate a conversation about a software project
    print("\n" + "-"*80)
    print("SIMULATING PROJECT DISCUSSION")
    print("-"*80)
    
    conversation = [
        ("user", "Hi, I'm working on a new project called DataPipeline. It's a Python library for ETL."),
        ("assistant", "That sounds interesting! Tell me more about DataPipeline. What problems does it solve?"),
        ("user", "DataPipeline helps extract data from various sources like PostgreSQL and MySQL databases."),
        ("assistant", "Working with databases is crucial. Are you using SQLAlchemy for database connections?"),
        ("user", "Yes! SQLAlchemy is our main ORM. We also integrate with Apache Kafka for streaming."),
        ("assistant", "Kafka is excellent for streaming. How are you handling data transformation?"),
        ("user", "We use pandas for data transformation. My team member Alice Johnson built that module."),
        ("assistant", "Alice's work on pandas integration sounds valuable. What about deployment?"),
        ("user", "We deploy to AWS using Docker containers. Contact us at support@datapipeline.io"),
        ("assistant", "AWS and Docker are solid choices. When is the release date?"),
        ("user", "We're targeting December 15, 2024 for the initial release."),
    ]
    
    for role, content in conversation:
        buffer.add_message(role, content)
        print(f"  {role}: {content[:80]}...")
    
    # Extract and analyze entities
    print("\n" + "-"*80)
    print("EXTRACTED ENTITIES")
    print("-"*80)
    
    entities = buffer.get_entities(min_mentions=1)
    print(f"\nFound {len(entities)} unique entities:")
    
    # Group entities by type
    by_type = {}
    for entity in entities:
        if entity.type not in by_type:
            by_type[entity.type] = []
        by_type[entity.type].append(entity)
    
    for entity_type, type_entities in sorted(by_type.items()):
        print(f"\n  {entity_type.upper()} ({len(type_entities)}):")
        for entity in sorted(type_entities, key=lambda e: e.mentions, reverse=True):
            print(f"    - {entity.name} (mentioned {entity.mentions} time(s))")
    
    # Show entity details
    print("\n" + "-"*80)
    print("ENTITY DETAILS")
    print("-"*80)
    
    # Focus on highly mentioned entities
    important_entities = [e for e in entities if e.mentions >= 2]
    print(f"\nEntities mentioned 2+ times ({len(important_entities)}):")
    
    for entity in important_entities:
        print(f"\n  Entity: {entity.name}")
        print(f"    Type: {entity.type}")
        print(f"    Mentions: {entity.mentions}")
        print(f"    First seen: {entity.first_seen[:19]}")
        print(f"    Last seen: {entity.last_seen[:19]}")
        if entity.context:
            print(f"    Context: {entity.context[0][:60]}...")
    
    # Track specific entity mentions over conversation
    print("\n" + "-"*80)
    print("ENTITY MENTION TRACKING")
    print("-"*80)
    
    # Track mentions of specific entities across messages
    messages = buffer.messages
    tracked_entity_names = ["DataPipeline", "AWS", "Alice"]
    
    print(f"\nTracking mentions of: {', '.join(tracked_entity_names)}")
    
    # Track each entity manually since function expects Entity objects
    mention_tracking = {}
    for entity_name in tracked_entity_names:
        mentions = []
        entity_name_lower = entity_name.lower()
        for idx, message in enumerate(messages):
            if entity_name_lower in message.content.lower():
                mentions.append((idx, message))
        mention_tracking[entity_name] = mentions
    
    for entity_name, mentions in mention_tracking.items():
        print(f"\n  '{entity_name}' mentioned in {len(mentions)} message(s):")
        for msg_idx, msg in mentions:
            print(f"    Message {msg_idx}: ...{msg.content[:60]}...")
    
    # Use entities to enrich context
    print("\n" + "-"*80)
    print("ENTITY-ENRICHED CONTEXT")
    print("-"*80)
    
    # Build context with entity information
    entity_context = []
    
    if entities:
        # Add entity summary
        key_entities = entities[:5]  # Top 5 entities
        entity_names = [f"{e.name} ({e.type})" for e in key_entities]
        entity_context.append(f"Key entities: {', '.join(entity_names)}")
    
    # Add recent conversation
    recent_messages = buffer.get_recent_messages(count=3)
    entity_context.append("\nRecent conversation:")
    for msg in recent_messages:
        entity_context.append(f"{msg.role}: {msg.content}")
    
    context = "\n".join(entity_context)
    print(f"\nEnriched context for LLM:\n{context}")
    
    # Demonstrate manual entity extraction
    print("\n" + "-"*80)
    print("MANUAL ENTITY EXTRACTION")
    print("-"*80)
    
    sample_texts = [
        "I work at Google in Mountain View. Contact me at john.doe@gmail.com",
        "The meeting is scheduled for March 15, 2024 at Microsoft Office",
        "Visit our website at https://example.com for more information"
    ]
    
    extracted = extract_entities(
        sample_texts,
        entity_types=["person", "organization", "location", "email", "url", "date"]
    )
    
    print(f"\nExtracted {len(extracted)} entities from sample texts:")
    for entity in extracted:
        print(f"  - {entity.name} ({entity.type})")
    
    # Merge similar entities
    print("\n" + "-"*80)
    print("ENTITY MERGING")
    print("-"*80)
    
    # Create entities with variations
    entity1 = Entity(name="DataPipeline", type="organization", mentions=3)
    entity2 = Entity(name="datapipeline", type="organization", mentions=2)
    entity3 = Entity(name="Data Pipeline", type="organization", mentions=1)
    
    print(f"\nBefore merging (3 entities):")
    print(f"  - {entity1.name} (mentions: {entity1.mentions})")
    print(f"  - {entity2.name} (mentions: {entity2.mentions})")
    print(f"  - {entity3.name} (mentions: {entity3.mentions})")
    
    # Merge pairwise
    merged_first = merge_entities(entity1, entity2, prefer="most_mentioned")
    merged_final = merge_entities(merged_first, entity3, prefer="most_mentioned")
    
    print(f"\nAfter merging (1 entity):")
    print(f"  - {merged_final.name} (mentions: {merged_final.mentions})")
    
    # Real-world application: Building a knowledge graph
    print("\n" + "-"*80)
    print("APPLICATION: KNOWLEDGE GRAPH")
    print("-"*80)
    
    # Build simple knowledge graph from entities
    knowledge_graph = {
        "project": None,
        "technologies": [],
        "people": [],
        "organizations": [],
        "dates": []
    }
    
    for entity in entities:
        if entity.type == "organization":
            if "pipeline" in entity.name.lower():
                knowledge_graph["project"] = entity.name
            else:
                knowledge_graph["organizations"].append(entity.name)
        elif entity.type == "person":
            knowledge_graph["people"].append(entity.name)
        elif entity.type == "date":
            knowledge_graph["dates"].append(entity.name)
    
    # Infer technologies from conversation
    tech_keywords = ["Python", "PostgreSQL", "MySQL", "SQLAlchemy", "Kafka", "pandas", "Docker", "AWS"]
    for msg in buffer.messages:
        for tech in tech_keywords:
            if tech in msg.content and tech not in knowledge_graph["technologies"]:
                knowledge_graph["technologies"].append(tech)
    
    print("\nKnowledge Graph:")
    print(f"  Project: {knowledge_graph['project']}")
    print(f"  Technologies: {', '.join(knowledge_graph['technologies'])}")
    print(f"  People: {', '.join(knowledge_graph['people'])}")
    print(f"  Organizations: {', '.join(knowledge_graph['organizations'])}")
    print(f"  Important Dates: {', '.join(knowledge_graph['dates'])}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
