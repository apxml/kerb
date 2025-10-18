"""Utility functions for safety operations.

This module contains helper functions used internally by safety checks.
"""

from typing import List


def _calculate_context_weight(text: str, match_pos: int, context_window: int = 10) -> float:
    """Calculate weight modifier based on surrounding context.
    
    Args:
        text: Full text
        match_pos: Position of the match
        context_window: Characters to check before/after
        
    Returns:
        Weight modifier (0.5 to 1.5)
    """
    # Get context around the match
    start = max(0, match_pos - context_window)
    end = min(len(text), match_pos + context_window)
    context = text[start:end].lower()
    
    # Intensifiers increase severity
    intensifiers = ['very', 'extremely', 'really', 'so', 'totally', 'absolutely']
    if any(word in context for word in intensifiers):
        return 1.3
    
    # Negations might reduce severity
    negations = ["not", "don't", "never", "no", "isn't", "aren't", "wasn't", "weren't"]
    if any(word in context for word in negations):
        return 0.7
    
    # Questions might reduce severity
    if '?' in context:
        return 0.8
    
    return 1.0


def _extract_ngrams(text: str, n: int = 2) -> List[str]:
    """Extract n-grams from text for pattern matching.
    
    Args:
        text: Text to process
        n: N-gram size (2=bigrams, 3=trigrams)
        
    Returns:
        List of n-grams
    """
    words = text.lower().split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams


def _check_repeated_patterns(text: str) -> float:
    """Check for repeated toxic patterns (spam/trolling indicator).
    
    Args:
        text: Text to check
        
    Returns:
        Repetition penalty (0.0 to 1.0)
    """
    words = text.lower().split()
    if len(words) < 3:
        return 0.0
    
    # Check for repeated words
    word_counts = {}
    for word in words:
        if len(word) > 3:  # Skip short words
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Calculate repetition ratio
    max_count = max(word_counts.values()) if word_counts else 1
    if max_count > 3:
        return min((max_count - 2) * 0.2, 1.0)
    
    return 0.0
