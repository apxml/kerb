"""A/B testing and output comparison utilities.

This module provides functions for comparing multiple outputs and performing
A/B testing on different variants.
"""

import math
import statistics
from typing import List, Dict, Any, Tuple, Callable, Optional

from .quality import assess_coherence, assess_fluency


# ============================================================================
# A/B Testing and Comparison
# ============================================================================

def ab_test(
    outputs_a: List[str],
    outputs_b: List[str],
    evaluation_fn: Callable[[str], float],
    labels: Tuple[str, str] = ("A", "B")
) -> Dict[str, Any]:
    """Perform A/B testing on two sets of outputs.
    
    Args:
        outputs_a: Outputs from variant A
        outputs_b: Outputs from variant B
        evaluation_fn: Function to score each output (returns float)
        labels: Labels for variants (default: ("A", "B"))
        
    Returns:
        dict: A/B test results with statistics
        
    Example:
        >>> results = ab_test(
        ...     ["Good answer", "Great answer"],
        ...     ["OK answer", "Bad answer"],
        ...     lambda x: len(x.split())
        ... )
        >>> results['winner']
        'A'
    """
    if len(outputs_a) != len(outputs_b):
        raise ValueError("Both output lists must have the same length")
    
    scores_a = [evaluation_fn(output) for output in outputs_a]
    scores_b = [evaluation_fn(output) for output in outputs_b]
    
    avg_a = statistics.mean(scores_a) if scores_a else 0.0
    avg_b = statistics.mean(scores_b) if scores_b else 0.0
    
    stdev_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0.0
    stdev_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0.0
    
    # Determine winner
    if avg_a > avg_b * 1.05:  # 5% threshold
        winner = labels[0]
    elif avg_b > avg_a * 1.05:
        winner = labels[1]
    else:
        winner = "tie"
    
    # Simple statistical significance (paired t-test approximation)
    differences = [a - b for a, b in zip(scores_a, scores_b)]
    avg_diff = statistics.mean(differences) if differences else 0.0
    stdev_diff = statistics.stdev(differences) if len(differences) > 1 else 0.0
    
    # T-statistic approximation
    n = len(differences)
    if stdev_diff > 0 and n > 1:
        t_stat = avg_diff / (stdev_diff / math.sqrt(n))
        # Very rough p-value approximation (would need scipy for exact)
        p_value = 0.05 if abs(t_stat) > 2.0 else 0.5
    else:
        p_value = 1.0
    
    return {
        "winner": winner,
        "variant_a": {
            "label": labels[0],
            "mean": avg_a,
            "stdev": stdev_a,
            "scores": scores_a
        },
        "variant_b": {
            "label": labels[1],
            "mean": avg_b,
            "stdev": stdev_b,
            "scores": scores_b
        },
        "difference": avg_a - avg_b,
        "improvement_pct": ((avg_a - avg_b) / avg_b * 100) if avg_b != 0 else 0.0,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "sample_size": n
    }


def compare_outputs(
    outputs: List[Tuple[str, str]],
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compare multiple outputs using various metrics.
    
    Args:
        outputs: List of (id, output) tuples
        metrics: List of metrics to compute (default: all)
        
    Returns:
        dict: Comparison results with rankings
        
    Example:
        >>> results = compare_outputs([
        ...     ("v1", "Short answer"),
        ...     ("v2", "This is a longer and more detailed answer")
        ... ])
        >>> results['rankings']['length'][0]
        'v2'
    """
    if metrics is None:
        metrics = ["length", "coherence", "fluency"]
    
    results: Dict[str, Any] = {
        "outputs": {},
        "rankings": {},
        "scores": {}
    }
    
    for output_id, output in outputs:
        results["outputs"][output_id] = output
        results["scores"][output_id] = {}
    
    # Calculate metrics
    for metric in metrics:
        scores = {}
        
        for output_id, output in outputs:
            if metric == "length":
                score = len(output.split())
            elif metric == "coherence":
                score = assess_coherence(output).score
            elif metric == "fluency":
                score = assess_fluency(output).score
            else:
                score = 0.0
            
            scores[output_id] = score
            results["scores"][output_id][metric] = score
        
        # Rank by this metric
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results["rankings"][metric] = [output_id for output_id, _ in ranked]
    
    # Calculate overall ranking (average rank across metrics)
    avg_ranks = {}
    for output_id, _ in outputs:
        ranks = [results["rankings"][m].index(output_id) for m in metrics]
        avg_ranks[output_id] = statistics.mean(ranks)
    
    overall_ranked = sorted(avg_ranks.items(), key=lambda x: x[1])
    results["rankings"]["overall"] = [output_id for output_id, _ in overall_ranked]
    
    return results
