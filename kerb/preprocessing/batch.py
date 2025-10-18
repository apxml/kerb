"""Batch processing utilities."""

from typing import List, Optional, Callable

from .text import normalize_text


def preprocess_batch(
    texts: List[str],
    operations: Optional[List[Callable]] = None,
    **kwargs
) -> List[str]:
    """Apply preprocessing pipeline to batch.
    
    Args:
        texts: List of texts to preprocess
        operations: List of preprocessing functions
        **kwargs: Arguments to pass to operations
        
    Returns:
        List of preprocessed texts
        
    Examples:
        >>> preprocess_batch(["  HELLO  ", "  WORLD  "], [str.lower, str.strip])
        ['hello', 'world']
    """
    if not texts:
        return []
    
    if operations is None:
        operations = [normalize_text]
    
    result = texts
    for operation in operations:
        result = [operation(text, **kwargs) if kwargs else operation(text) for text in result]
    
    return result


def preprocess_pipeline(*operations: Callable) -> Callable:
    """Create custom preprocessing pipeline.
    
    Args:
        *operations: Preprocessing functions to chain
        
    Returns:
        Pipeline function
        
    Examples:
        >>> pipeline = preprocess_pipeline(str.lower, str.strip)
        >>> pipeline("  HELLO  ")
        'hello'
    """
    def pipeline(text: str, **kwargs) -> str:
        result = text
        for operation in operations:
            if kwargs:
                result = operation(result, **kwargs)
            else:
                result = operation(result)
        return result
    
    return pipeline
