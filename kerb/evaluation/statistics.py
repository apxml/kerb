"""Statistical analysis utilities for evaluation.

This module provides statistical functions for analyzing evaluation scores:
- Calculate descriptive statistics
- Compute confidence intervals
"""

import math
import statistics
from typing import List, Dict, Tuple


# ============================================================================
# Statistical Analysis
# ============================================================================

def calculate_statistics(scores: List[float]) -> Dict[str, float]:
    """Calculate statistical measures for a list of scores.
    
    Args:
        scores: List of numeric scores
        
    Returns:
        dict: Statistical measures (mean, median, stdev, min, max, percentiles)
        
    Example:
        >>> stats = calculate_statistics([0.5, 0.7, 0.8, 0.9, 1.0])
        >>> stats['mean']
        0.78
    """
    if not scores:
        return {
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "variance": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0
        }
    
    sorted_scores = sorted(scores)
    
    return {
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "variance": statistics.variance(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "count": len(scores),
        "p25": sorted_scores[len(sorted_scores) // 4],
        "p75": sorted_scores[3 * len(sorted_scores) // 4],
        "p90": sorted_scores[9 * len(sorted_scores) // 10],
        "p95": sorted_scores[19 * len(sorted_scores) // 20] if len(sorted_scores) >= 20 else sorted_scores[-1]
    }


def confidence_interval(
    scores: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for scores.
    
    Args:
        scores: List of scores
        confidence: Confidence level (default: 0.95)
        
    Returns:
        tuple: (lower_bound, upper_bound)
        
    Example:
        >>> lower, upper = confidence_interval([0.7, 0.8, 0.9])
        >>> lower < 0.8 < upper
        True
    """
    if not scores or len(scores) < 2:
        return (0.0, 0.0)
    
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)
    n = len(scores)
    
    # Use t-distribution approximation (simplified)
    # For 95% confidence and reasonable n, t ≈ 2
    if confidence == 0.95:
        t_value = 2.0
    elif confidence == 0.99:
        t_value = 2.6
    else:
        t_value = 1.96  # z-value for 95%
    
    margin = t_value * stdev / math.sqrt(n)
    
    return (mean - margin, mean + margin)
