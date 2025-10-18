"""Priority management for context items.

This module provides functions for assigning and managing priorities
for context items based on various strategies.
"""

from typing import List, Callable, Optional

from .types import ContextItem


def assign_priorities(
    items: List[ContextItem],
    priority_fn: Callable[[ContextItem], float]
) -> List[ContextItem]:
    """Assign priorities to context items using custom function.
    
    Args:
        items: List of context items
        priority_fn: Function that takes ContextItem and returns priority score
        
    Returns:
        List[ContextItem]: Items with updated priorities
        
    Example:
        >>> items = assign_priorities(items, lambda x: len(x.content) / 100)
    """
    for item in items:
        item.priority = priority_fn(item)
    
    return items


def priority_by_recency(items: List[ContextItem]) -> List[ContextItem]:
    """Assign priorities based on recency (newer = higher priority).
    
    Args:
        items: List of context items
        
    Returns:
        List[ContextItem]: Items with recency-based priorities
    """
    for i, item in enumerate(items):
        item.priority = (i + 1) / len(items)
    
    return items


def priority_by_relevance(
    items: List[ContextItem],
    query: str,
    relevance_fn: Optional[Callable[[str, str], float]] = None
) -> List[ContextItem]:
    """Assign priorities based on relevance to query.
    
    Args:
        items: List of context items
        query: Query string for relevance calculation
        relevance_fn: Custom relevance function
        
    Returns:
        List[ContextItem]: Items with relevance-based priorities
        
    Example:
        >>> items = priority_by_relevance(items, "machine learning")
    """
    if relevance_fn is None:
        # Simple keyword-based relevance
        query_words = set(query.lower().split())
        
        def default_relevance(content: str, q: str) -> float:
            content_words = set(content.lower().split())
            if not query_words:
                return 0.0
            overlap = len(query_words & content_words)
            return overlap / len(query_words)
        
        relevance_fn = lambda content: default_relevance(content, query)
    
    for item in items:
        item.priority = relevance_fn(item.content)
    
    return items


def priority_by_diversity(
    items: List[ContextItem],
    similarity_fn: Optional[Callable[[str, str], float]] = None
) -> List[ContextItem]:
    """Assign priorities to maximize diversity (MMR-style).
    
    Args:
        items: List of context items
        similarity_fn: Function to compute similarity between items
        
    Returns:
        List[ContextItem]: Items with diversity-based priorities
    """
    if not items:
        return items
    
    if similarity_fn is None:
        # Simple word overlap similarity
        def word_overlap(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            overlap = len(words1 & words2)
            union = len(words1 | words2)
            return overlap / union if union > 0 else 0.0
        
        similarity_fn = word_overlap
    
    # Start with first item at high priority
    items[0].priority = 1.0
    selected = [items[0]]
    
    # For each remaining item, compute diversity score
    for item in items[1:]:
        # Find maximum similarity to already selected items
        max_similarity = max(
            similarity_fn(item.content, s.content) 
            for s in selected
        )
        # Priority is inverse of similarity (diverse = high priority)
        item.priority = 1.0 - max_similarity
        selected.append(item)
    
    return items
