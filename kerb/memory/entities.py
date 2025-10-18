"""Entity extraction and tracking functions.

This module provides functions for extracting and managing entities:
- extract_entities: Extract entities from text
- track_entity_mentions: Track entity mentions across messages
- extract_entity_relationships: Find relationships between entities
- merge_entities: Merge duplicate entities
"""

import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from kerb.core.types import Message

from .classes import Entity


def extract_entities(
    texts: List[str], entity_types: Optional[List[str]] = None
) -> List[Entity]:
    """Extract entities from text using pattern matching.

    Args:
        texts: List of text strings to extract from
        entity_types: Types to extract (defaults to common types)

    Returns:
        List[Entity]: Extracted entities with metadata

    Example:
        >>> entities = extract_entities(["My name is Alice"], entity_types=["person"])
    """
    if entity_types is None:
        entity_types = ["person", "location", "organization", "date", "email", "url"]

    entities_dict: Dict[str, Entity] = {}

    # Simple pattern-based extraction
    patterns = {
        "person": [
            r"\b(?:my name is|I am|I\'m|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            r"\b([A-Z][a-z]+)\s+(?:said|told|asked|replied)\b",
        ],
        "email": [
            r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",
        ],
        "url": [
            r"\b(https?://[^\s]+)\b",
        ],
        "date": [
            r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
            r"\b(\d{4}-\d{2}-\d{2})\b",
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        ],
        "location": [
            r"\b(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        ],
        "organization": [
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc|Corp|LLC|Ltd|Company|Organization)\b",
        ],
    }

    for text in texts:
        for entity_type in entity_types:
            if entity_type not in patterns:
                continue

            for pattern in patterns[entity_type]:
                matches = re.finditer(
                    pattern, text, re.IGNORECASE if entity_type != "person" else 0
                )

                for match in matches:
                    entity_name = match.group(1).strip()

                    # Skip very short or common words
                    if len(entity_name) < 2:
                        continue

                    entity_key = f"{entity_type}:{entity_name.lower()}"

                    if entity_key in entities_dict:
                        # Update existing entity
                        entity = entities_dict[entity_key]
                        entity.mentions += 1
                        entity.last_seen = datetime.now().isoformat()
                        if text[:100] not in entity.context:
                            entity.context.append(text[:100])
                    else:
                        # Create new entity
                        entities_dict[entity_key] = Entity(
                            name=entity_name, type=entity_type, context=[text[:100]]
                        )

    # Sort by mention count
    entities = sorted(entities_dict.values(), key=lambda e: e.mentions, reverse=True)

    return entities


def track_entity_mentions(
    messages: List[Message], entity: Entity
) -> List[Tuple[int, Message]]:
    """Track mentions of a specific entity across messages.

    Args:
        messages: Messages to search
        entity: Entity to track

    Returns:
        List[Tuple[int, Message]]: List of (index, message) tuples where entity appears

    Example:
        >>> mentions = track_entity_mentions(messages, entity)
    """
    mentions = []
    entity_name_lower = entity.name.lower()

    for idx, message in enumerate(messages):
        if entity_name_lower in message.content.lower():
            mentions.append((idx, message))

    return mentions


def extract_entity_relationships(
    messages: List[Message], entities: List[Entity]
) -> Dict[str, List[str]]:
    """Extract relationships between entities.

    Args:
        messages: Messages to analyze
        entities: List of entities to find relationships for

    Returns:
        Dict[str, List[str]]: Mapping of entity names to related entities

    Example:
        >>> relationships = extract_entity_relationships(messages, entities)
    """
    relationships: Dict[str, List[str]] = defaultdict(list)

    # Simple co-occurrence based relationship extraction
    entity_names = {e.name.lower(): e.name for e in entities}

    for message in messages:
        content_lower = message.content.lower()

        # Find entities mentioned in this message
        mentioned = []
        for name_lower, name_original in entity_names.items():
            if name_lower in content_lower:
                mentioned.append(name_original)

        # Create relationships between co-occurring entities
        if len(mentioned) >= 2:
            for i, entity1 in enumerate(mentioned):
                for entity2 in mentioned[i + 1 :]:
                    if entity2 not in relationships[entity1]:
                        relationships[entity1].append(entity2)
                    if entity1 not in relationships[entity2]:
                        relationships[entity2].append(entity1)

    return dict(relationships)


def merge_entities(
    entity1: Entity, entity2: Entity, prefer: str = "most_mentioned"
) -> Entity:
    """Merge two entities (useful for deduplication).

    Args:
        entity1: First entity
        entity2: Second entity
        prefer: "most_mentioned", "first", or "second"

    Returns:
        Entity: Merged entity

    Example:
        >>> merged = merge_entities(entity1, entity2, prefer="most_mentioned")
    """
    if prefer == "most_mentioned":
        primary = entity1 if entity1.mentions >= entity2.mentions else entity2
        secondary = entity2 if primary == entity1 else entity1
    elif prefer == "first":
        primary, secondary = entity1, entity2
    else:  # "second"
        primary, secondary = entity2, entity1

    # Merge data
    merged = Entity(
        name=primary.name,
        type=primary.type,
        mentions=primary.mentions + secondary.mentions,
        first_seen=min(primary.first_seen, secondary.first_seen),
        last_seen=max(primary.last_seen, secondary.last_seen),
        context=primary.context + secondary.context,
        metadata={**primary.metadata, **secondary.metadata},
    )

    return merged
