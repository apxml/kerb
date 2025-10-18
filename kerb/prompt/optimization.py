"""Prompt compression and optimization utilities.

This module provides tools for reducing prompt size and optimizing
prompt structure while maintaining clarity and effectiveness.
"""

import re
from typing import Optional, List, Dict, Any

from kerb.preprocessing import truncate_text
from .template import extract_template_variables


def compress_prompt(
    prompt: str,
    max_length: Optional[int] = None,
    strategies: Optional[List[str]] = None
) -> str:
    """Compress a prompt using multiple optimization strategies.
    
    Args:
        prompt (str): Prompt to compress
        max_length (Optional[int]): Target maximum length. If None, applies all strategies
            without length constraint. Defaults to None.
        strategies (Optional[List[str]]): List of strategies to apply.
            Options: ["whitespace", "abbreviations"].
            If None, applies all strategies. Defaults to None.
            
    Returns:
        str: Compressed prompt
        
    Examples:
        >>> compress_prompt("Hello    world!  This   is   a    test.")
        'Hello world! This is a test.'
    """
    if not prompt:
        return prompt
    
    if strategies is None:
        strategies = ["whitespace", "abbreviations"]
    
    result = prompt
    
    # Apply optimization strategies
    if "whitespace" in strategies:
        result = optimize_whitespace(result)
    
    if "abbreviations" in strategies:
        result = _apply_abbreviations(result)
    
    # Truncate if max_length is specified
    if max_length and len(result) > max_length:
        result = truncate_text(result, max_length, strategy="smart")
    
    return result


def optimize_whitespace(prompt: str) -> str:
    """Optimize whitespace in a prompt.
    
    Removes excessive spaces, trailing whitespace, and normalizes line breaks.
    
    Args:
        prompt (str): Prompt to optimize
        
    Returns:
        str: Prompt with optimized whitespace
        
    Examples:
        >>> optimize_whitespace("Hello    world!  \\n\\n\\n  Test")
        'Hello world!\\n\\nTest'
    """
    if not prompt:
        return prompt
    
    # Replace multiple spaces with single space
    result = re.sub(r' +', ' ', prompt)
    
    # Replace multiple newlines with double newline (paragraph break)
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    
    # Remove trailing whitespace from each line
    lines = result.split('\n')
    lines = [line.rstrip() for line in lines]
    result = '\n'.join(lines)
    
    # Remove leading/trailing whitespace from entire prompt
    result = result.strip()
    
    return result


def _apply_abbreviations(prompt: str) -> str:
    """Apply common abbreviations to reduce prompt length.
    
    Args:
        prompt (str): Prompt to abbreviate
        
    Returns:
        str: Prompt with abbreviations applied
    """
    # Common abbreviations that maintain clarity
    abbreviations = {
        r'\bfor example\b': 'e.g.',
        r'\bthat is\b': 'i.e.',
        r'\band so on\b': 'etc.',
        r'\band others\b': 'et al.',
    }
    
    result = prompt
    for pattern, replacement in abbreviations.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def analyze_prompt(prompt: str, tokenizer: Optional[Any] = None) -> Dict[str, Any]:
    """Analyze a prompt and return statistics.
    
    Args:
        prompt (str): Prompt to analyze
        tokenizer (Optional[Any]): Tokenizer to use for token counting.
            If None, uses character approximation. Defaults to None.
            
    Returns:
        Dict[str, Any]: Analysis results including length, word count, line count, etc.
        
    Examples:
        >>> analyze_prompt("Hello world! This is a test.")
        {
            'length': 28,
            'words': 6,
            'lines': 1,
            'sentences': 2,
            'tokens_approx': 7,
            'variables': []
        }
    """
    if not prompt:
        return {
            'length': 0,
            'words': 0,
            'lines': 0,
            'sentences': 0,
            'tokens_approx': 0,
            'variables': []
        }
    
    # Basic statistics
    length = len(prompt)
    words = len(prompt.split())
    lines = len(prompt.split('\n'))
    
    # Count sentences (approximate)
    sentences = len(re.findall(r'[.!?]+', prompt))
    
    # Token approximation (4 chars per token is typical for English)
    tokens_approx = length // 4
    
    # Try to use actual tokenizer if provided
    if tokenizer is not None:
        try:
            from ..tokenizer import count_tokens
            tokens_approx = count_tokens(prompt, tokenizer)
        except Exception:
            pass
    
    # Extract template variables
    variables = extract_template_variables(prompt)
    
    return {
        'length': length,
        'words': words,
        'lines': lines,
        'sentences': sentences,
        'tokens_approx': tokens_approx,
        'variables': variables,
        'avg_word_length': length / words if words > 0 else 0,
        'avg_sentence_length': words / sentences if sentences > 0 else 0
    }
