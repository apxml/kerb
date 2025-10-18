"""Content classification and pattern matching functions.

This module provides functions for pattern matching, content classification,
risk scoring, and entity extraction.
"""

import re
from typing import List, Tuple, Dict, Optional

from .enums import ContentCategory
from .moderation import (
    check_toxicity,
    check_sexual_content,
    check_violence,
    check_hate_speech,
    check_profanity
)
from .pii import (
    detect_email,
    detect_phone,
    detect_url,
    detect_ip_address,
    detect_ssn,
    detect_credit_card
)


def match_patterns(
    text: str,
    patterns: List[str],
    case_sensitive: bool = False
) -> List[Tuple[str, List[str]]]:
    """Match text against safety patterns.
    
    Args:
        text: Text to match
        patterns: List of regex patterns
        case_sensitive: Whether matching is case sensitive
        
    Returns:
        List of tuples (pattern, list of matches)
        
    Examples:
        >>> patterns = [r'\b\d{3}-\d{2}-\d{4}\b', r'\b\w+@\w+\.\w+\b']
        >>> matches = match_patterns(text, patterns)
    """
    results = []
    flags = 0 if case_sensitive else re.IGNORECASE
    
    for pattern in patterns:
        matches = re.findall(pattern, text, flags)
        if matches:
            results.append((pattern, matches))
    
    return results


def classify_content(
    text: str,
    categories: Optional[List[ContentCategory]] = None
) -> Dict[ContentCategory, float]:
    """Classify content into safety categories.
    
    Args:
        text: Text to classify
        categories: Specific categories to check (None = all)
        
    Returns:
        Dictionary mapping categories to confidence scores
        
    Examples:
        >>> scores = classify_content("I hate this stupid thing")
        >>> print(scores)
        {ContentCategory.TOXICITY: 0.7, ContentCategory.HATE_SPEECH: 0.6, ...}
    """
    if categories is None:
        categories = [
            ContentCategory.TOXICITY,
            ContentCategory.SEXUAL,
            ContentCategory.VIOLENCE,
            ContentCategory.HATE_SPEECH,
            ContentCategory.PROFANITY,
        ]
    
    scores = {}
    
    for category in categories:
        if category == ContentCategory.TOXICITY:
            result = check_toxicity(text)
            scores[category] = 1.0 - result.score  # Invert: high score = more toxic
        elif category == ContentCategory.SEXUAL:
            result = check_sexual_content(text)
            scores[category] = 1.0 - result.score
        elif category == ContentCategory.VIOLENCE:
            result = check_violence(text)
            scores[category] = 1.0 - result.score
        elif category == ContentCategory.HATE_SPEECH:
            result = check_hate_speech(text)
            scores[category] = 1.0 - result.score
        elif category == ContentCategory.PROFANITY:
            result = check_profanity(text)
            scores[category] = 1.0 - result.score
        else:
            scores[category] = 0.0
    
    return scores


def score_content(
    text: str,
    weights: Optional[Dict[ContentCategory, float]] = None
) -> float:
    """Score content for safety risk.
    
    Args:
        text: Text to score
        weights: Category weights (defaults to equal weight)
        
    Returns:
        Overall safety risk score (0.0 = safe, 1.0 = very unsafe)
        
    Examples:
        >>> score = score_content("This is a normal message")
        >>> print(score)  # Close to 0.0 (safe)
        
        >>> score = score_content("I hate you stupid idiot")
        >>> print(score)  # Higher value (unsafe)
    """
    category_scores = classify_content(text)
    
    if weights is None:
        # Equal weights
        weights = {cat: 1.0 for cat in category_scores.keys()}
    
    # Weighted average
    total_weight = sum(weights.get(cat, 1.0) for cat in category_scores.keys())
    weighted_sum = sum(
        score * weights.get(cat, 1.0)
        for cat, score in category_scores.items()
    )
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def extract_entities(
    text: str,
    entity_types: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """Extract sensitive entities from text.
    
    Args:
        text: Text to extract from
        entity_types: Types of entities to extract (None = common types)
        
    Returns:
        Dictionary mapping entity types to lists of extracted entities
        
    Examples:
        >>> entities = extract_entities("Email john@example.com at 555-1234")
        >>> print(entities)
        {'email': ['john@example.com'], 'phone': ['555-1234']}
    """
    if entity_types is None:
        entity_types = ['email', 'phone', 'url', 'ip_address']
    
    entities = {}
    
    if 'email' in entity_types:
        matches = [m.text for m in detect_email(text)]
        if matches:
            entities['email'] = matches
    
    if 'phone' in entity_types:
        matches = [m.text for m in detect_phone(text)]
        if matches:
            entities['phone'] = matches
    
    if 'url' in entity_types:
        matches = [m.text for m in detect_url(text)]
        if matches:
            entities['url'] = matches
    
    if 'ip_address' in entity_types:
        matches = [m.text for m in detect_ip_address(text)]
        if matches:
            entities['ip_address'] = matches
    
    if 'ssn' in entity_types:
        matches = [m.text for m in detect_ssn(text)]
        if matches:
            entities['ssn'] = matches
    
    if 'credit_card' in entity_types:
        matches = [m.text for m in detect_credit_card(text)]
        if matches:
            entities['credit_card'] = matches
    
    return entities
