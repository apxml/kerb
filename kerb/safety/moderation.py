"""Content moderation functions.

This module provides functions for detecting and moderating various types
of harmful or inappropriate content including toxicity, sexual content,
violence, hate speech, self-harm, and profanity.
"""

import re
from typing import List, Optional

from .enums import SafetyLevel, ContentCategory, ToxicityLevel
from .types import SafetyResult, ModerationResult
from .patterns import (
    TOXICITY_PATTERNS,
    SEXUAL_PATTERNS,
    VIOLENCE_PATTERNS,
    HATE_SPEECH_PATTERNS,
    SELF_HARM_PATTERNS,
    PROFANITY_PATTERNS,
)


def moderate_content(
    text: str,
    categories: Optional[List[ContentCategory]] = None,
    threshold: float = 0.5,
    level: SafetyLevel = SafetyLevel.MODERATE
) -> ModerationResult:
    """Check content against multiple safety categories.
    
    Args:
        text: Text to moderate
        categories: Specific categories to check (None = all)
        threshold: Score threshold for flagging (0.0-1.0)
        level: Safety strictness level
        
    Returns:
        ModerationResult with overall assessment
        
    Examples:
        >>> result = moderate_content("This is a normal message")
        >>> print(result.safe)  # True
        
        >>> result = moderate_content("I hate you stupid idiot")
        >>> print(result.safe)  # False
        >>> print(result.flagged_categories)  # [ContentCategory.TOXICITY]
    """
    if categories is None:
        categories = [
            ContentCategory.TOXICITY,
            ContentCategory.SEXUAL,
            ContentCategory.VIOLENCE,
            ContentCategory.HATE_SPEECH,
            ContentCategory.PROFANITY,
        ]
    
    category_scores = {}
    flagged = []
    
    # Check each category
    if ContentCategory.TOXICITY in categories:
        result = check_toxicity(text, level)
        category_scores[ContentCategory.TOXICITY] = result.score
        if not result.safe:
            flagged.append(ContentCategory.TOXICITY)
    
    if ContentCategory.SEXUAL in categories:
        result = check_sexual_content(text, level)
        category_scores[ContentCategory.SEXUAL] = result.score
        if not result.safe:
            flagged.append(ContentCategory.SEXUAL)
    
    if ContentCategory.VIOLENCE in categories:
        result = check_violence(text, level)
        category_scores[ContentCategory.VIOLENCE] = result.score
        if not result.safe:
            flagged.append(ContentCategory.VIOLENCE)
    
    if ContentCategory.HATE_SPEECH in categories:
        result = check_hate_speech(text, level)
        category_scores[ContentCategory.HATE_SPEECH] = result.score
        if not result.safe:
            flagged.append(ContentCategory.HATE_SPEECH)
    
    if ContentCategory.PROFANITY in categories:
        result = check_profanity(text, level)
        category_scores[ContentCategory.PROFANITY] = result.score
        if not result.safe:
            flagged.append(ContentCategory.PROFANITY)
    
    # Calculate overall score (average of category scores)
    overall_score = sum(category_scores.values()) / len(category_scores) if category_scores else 1.0
    safe = len(flagged) == 0 and overall_score >= threshold
    
    # Determine toxicity level
    if overall_score >= 0.9:
        toxicity_level = ToxicityLevel.NONE
    elif overall_score >= 0.7:
        toxicity_level = ToxicityLevel.LOW
    elif overall_score >= 0.5:
        toxicity_level = ToxicityLevel.MEDIUM
    elif overall_score >= 0.3:
        toxicity_level = ToxicityLevel.HIGH
    else:
        toxicity_level = ToxicityLevel.SEVERE
    
    return ModerationResult(
        safe=safe,
        categories=category_scores,
        flagged_categories=flagged,
        overall_score=overall_score,
        toxicity_level=toxicity_level,
        details={'threshold': threshold, 'level': level.value}
    )


def check_toxicity(text: str, level: SafetyLevel = SafetyLevel.MODERATE) -> SafetyResult:
    """Detect toxic, hateful, or harmful content.
    
    Args:
        text: Text to check
        level: Safety strictness level
        
    Returns:
        SafetyResult with toxicity assessment
        
    Examples:
        >>> result = check_toxicity("You're an idiot and I hate you")
        >>> print(result.safe)  # False
        >>> print(result.score)  # Low score indicates high toxicity
    """
    text_lower = text.lower()
    
    # Calculate weighted toxicity score
    total_toxicity = 0.0
    matched_patterns = []
    
    # Check patterns by severity
    for severity, patterns in TOXICITY_PATTERNS.items():
        for pattern, weight in patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            match_list = list(matches)
            if match_list:
                total_toxicity += weight * len(match_list)
                matched_patterns.append({
                    'severity': severity,
                    'pattern': pattern[:30],
                    'count': len(match_list),
                    'weight': weight
                })
    
    # Normalize score based on text length and number of patterns
    text_words = max(len(text.split()), 1)
    # Score decreases as toxicity increases
    normalized_toxicity = min(total_toxicity / text_words, 1.0)
    score = max(0.0, 1.0 - normalized_toxicity)
    
    # Apply strictness threshold
    thresholds = {
        SafetyLevel.PERMISSIVE: 0.3,
        SafetyLevel.MODERATE: 0.6,
        SafetyLevel.STRICT: 0.8,
    }
    
    threshold = thresholds[level]
    safe = score >= threshold
    
    # Determine reason based on severity
    reason = None
    if not safe:
        if any(p['severity'] == 'severe' for p in matched_patterns):
            reason = "Contains severely toxic language"
        elif any(p['severity'] == 'high' for p in matched_patterns):
            reason = "Contains highly toxic language"
        elif any(p['severity'] == 'medium' for p in matched_patterns):
            reason = "Contains moderately toxic language"
        else:
            reason = "Contains mildly toxic language"
    
    return SafetyResult(
        safe=safe,
        score=score,
        category=ContentCategory.TOXICITY,
        confidence=0.85,
        reason=reason,
        details={
            'total_toxicity': total_toxicity,
            'matched_patterns': len(matched_patterns),
            'patterns': matched_patterns
        }
    )


def check_sexual_content(text: str, level: SafetyLevel = SafetyLevel.MODERATE) -> SafetyResult:
    """Detect sexual or adult content.
    
    Args:
        text: Text to check
        level: Safety strictness level
        
    Returns:
        SafetyResult with sexual content assessment
    """
    text_lower = text.lower()
    
    total_score = 0.0
    matched_patterns = []
    
    for pattern, weight in SEXUAL_PATTERNS:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            total_score += weight * len(matches)
            matched_patterns.append({
                'pattern': pattern[:30],
                'count': len(matches),
                'weight': weight
            })
    
    # Normalize score
    text_words = max(len(text.split()), 1)
    normalized_score = min(total_score / text_words, 1.0)
    score = max(0.0, 1.0 - normalized_score)
    
    thresholds = {
        SafetyLevel.PERMISSIVE: 0.3,
        SafetyLevel.MODERATE: 0.6,
        SafetyLevel.STRICT: 0.8,
    }
    
    safe = score >= thresholds[level]
    reason = "Contains sexual or adult content" if not safe else None
    
    return SafetyResult(
        safe=safe,
        score=score,
        category=ContentCategory.SEXUAL,
        confidence=0.8,
        reason=reason,
        details={'total_score': total_score, 'matched_patterns': len(matched_patterns)}
    )


def check_violence(text: str, level: SafetyLevel = SafetyLevel.MODERATE) -> SafetyResult:
    """Detect violent content.
    
    Args:
        text: Text to check
        level: Safety strictness level
        
    Returns:
        SafetyResult with violence assessment
    """
    text_lower = text.lower()
    
    total_score = 0.0
    matched_patterns = []
    
    for pattern, weight in VIOLENCE_PATTERNS:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            total_score += weight * len(matches)
            matched_patterns.append({
                'pattern': pattern[:30],
                'count': len(matches),
                'weight': weight
            })
    
    # Normalize score
    text_words = max(len(text.split()), 1)
    normalized_score = min(total_score / text_words, 1.0)
    score = max(0.0, 1.0 - normalized_score)
    
    thresholds = {
        SafetyLevel.PERMISSIVE: 0.3,
        SafetyLevel.MODERATE: 0.6,
        SafetyLevel.STRICT: 0.8,
    }
    
    safe = score >= thresholds[level]
    reason = "Contains violent content" if not safe else None
    
    return SafetyResult(
        safe=safe,
        score=score,
        category=ContentCategory.VIOLENCE,
        confidence=0.8,
        reason=reason,
        details={'total_score': total_score, 'matched_patterns': len(matched_patterns)}
    )


def check_hate_speech(text: str, level: SafetyLevel = SafetyLevel.MODERATE) -> SafetyResult:
    """Detect hate speech or discrimination.
    
    Args:
        text: Text to check
        level: Safety strictness level
        
    Returns:
        SafetyResult with hate speech assessment
    """
    text_lower = text.lower()
    
    total_score = 0.0
    matched_patterns = []
    
    for pattern, weight in HATE_SPEECH_PATTERNS:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            total_score += weight * len(matches)
            matched_patterns.append({
                'pattern': pattern[:30],
                'count': len(matches),
                'weight': weight
            })
    
    # Normalize score
    text_words = max(len(text.split()), 1)
    normalized_score = min(total_score / text_words, 1.0)
    score = max(0.0, 1.0 - normalized_score)
    
    thresholds = {
        SafetyLevel.PERMISSIVE: 0.3,
        SafetyLevel.MODERATE: 0.6,
        SafetyLevel.STRICT: 0.8,
    }
    
    safe = score >= thresholds[level]
    reason = "Contains hate speech or discriminatory content" if not safe else None
    
    return SafetyResult(
        safe=safe,
        score=score,
        category=ContentCategory.HATE_SPEECH,
        confidence=0.8,
        reason=reason,
        details={'total_score': total_score, 'matched_patterns': len(matched_patterns)}
    )


def check_self_harm(text: str, level: SafetyLevel = SafetyLevel.MODERATE) -> SafetyResult:
    """Detect self-harm related content.
    
    Args:
        text: Text to check
        level: Safety strictness level
        
    Returns:
        SafetyResult with self-harm assessment
    """
    text_lower = text.lower()
    
    total_score = 0.0
    matched_patterns = []
    
    for pattern, weight in SELF_HARM_PATTERNS:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        if matches:
            total_score += weight * len(matches)
            matched_patterns.append({
                'pattern': pattern[:30],
                'count': len(matches),
                'weight': weight
            })
    
    # Normalize score - self-harm is very serious
    text_words = max(len(text.split()), 1)
    normalized_score = min(total_score / text_words, 1.0)
    score = max(0.0, 1.0 - normalized_score)
    
    thresholds = {
        SafetyLevel.PERMISSIVE: 0.3,
        SafetyLevel.MODERATE: 0.7,
        SafetyLevel.STRICT: 0.9,
    }
    
    safe = score >= thresholds[level]
    reason = "Contains self-harm related content" if not safe else None
    
    return SafetyResult(
        safe=safe,
        score=score,
        category=ContentCategory.SELF_HARM,
        confidence=0.85,
        reason=reason,
        details={'total_score': total_score, 'matched_patterns': len(matched_patterns)}
    )


def check_profanity(text: str, level: SafetyLevel = SafetyLevel.MODERATE) -> SafetyResult:
    """Detect profane or offensive language.
    
    Args:
        text: Text to check
        level: Safety strictness level
        
    Returns:
        SafetyResult with profanity assessment
    """
    text_lower = text.lower()
    
    total_score = 0.0
    matched_patterns = []
    
    for severity, patterns in PROFANITY_PATTERNS.items():
        for pattern, weight in patterns:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if matches:
                total_score += weight * len(matches)
                matched_patterns.append({
                    'severity': severity,
                    'pattern': pattern[:30],
                    'count': len(matches),
                    'weight': weight
                })
    
    # Normalize score
    text_words = max(len(text.split()), 1)
    normalized_score = min(total_score / text_words, 1.0)
    score = max(0.0, 1.0 - normalized_score)
    
    thresholds = {
        SafetyLevel.PERMISSIVE: 0.2,
        SafetyLevel.MODERATE: 0.5,
        SafetyLevel.STRICT: 0.8,
    }
    
    safe = score >= thresholds[level]
    reason = "Contains profanity" if not safe else None
    
    return SafetyResult(
        safe=safe,
        score=score,
        category=ContentCategory.PROFANITY,
        confidence=0.9,
        reason=reason,
        details={'total_score': total_score, 'matched_patterns': len(matched_patterns)}
    )
