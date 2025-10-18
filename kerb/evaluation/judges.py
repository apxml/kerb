"""LLM-as-judge evaluation functions.

This module provides functions for using LLMs to evaluate output quality:
- LLM as Judge: Use LLM to rate outputs on specific criteria
- Pairwise Comparison: Compare two outputs to determine the better one
"""

import re
from typing import Optional, Callable, Union, Tuple

from .types import EvaluationResult, ComparisonResult, JudgmentCriterion
from .metrics import calculate_semantic_similarity, calculate_f1_score
from .quality import assess_coherence


# ============================================================================
# LLM-as-Judge Functions
# ============================================================================

def llm_as_judge(
    output: str,
    criterion: Union[str, JudgmentCriterion],
    context: Optional[str] = None,
    reference: Optional[str] = None,
    scale: int = 5,
    llm_function: Optional[Callable] = None
) -> EvaluationResult:
    """Use an LLM to judge the quality of an output.
    
    Args:
        output: The text to evaluate
        criterion: Judgment criterion (relevance, accuracy, coherence, etc.)
        context: Optional context (e.g., the prompt or question)
        reference: Optional reference answer
        scale: Rating scale (default: 1-5)
        llm_function: Function to call LLM (should accept prompt and return string)
        
    Returns:
        EvaluationResult: Result with score and reasoning
        
    Example:
        >>> result = llm_as_judge(
        ...     "Python is a programming language",
        ...     JudgmentCriterion.RELEVANCE,
        ...     context="What is Python?"
        ... )
        >>> result.score
        4.5
    """
    if isinstance(criterion, JudgmentCriterion):
        criterion_name = criterion.value
    else:
        criterion_name = criterion
    
    # Build evaluation prompt
    prompt_parts = [
        f"Evaluate the following output on a scale of 1 to {scale} for {criterion_name}."
    ]
    
    if context:
        prompt_parts.append(f"\nContext/Question: {context}")
    
    if reference:
        prompt_parts.append(f"\nReference Answer: {reference}")
    
    prompt_parts.append(f"\nOutput to Evaluate: {output}")
    prompt_parts.append(f"\nProvide a rating from 1 to {scale} and explain your reasoning.")
    prompt_parts.append(f"Format: Rating: X\nReasoning: <explanation>")
    
    prompt = "\n".join(prompt_parts)
    
    # Call LLM if function provided
    if llm_function is not None:
        try:
            response = llm_function(prompt)
            score, reasoning = _parse_judge_response(response, scale)
        except Exception as e:
            # If LLM call fails, return a default result
            score = scale / 2
            reasoning = f"LLM evaluation failed: {str(e)}"
    else:
        # Without LLM, use heuristic scoring
        score, reasoning = _heuristic_judge(output, criterion_name, context, reference, scale)
    
    return EvaluationResult(
        metric=f"llm_judge_{criterion_name}",
        score=score / scale,  # Normalize to 0-1
        details={"raw_score": score, "scale": scale, "reasoning": reasoning}
    )


def pairwise_comparison(
    output_a: str,
    output_b: str,
    criterion: str,
    context: Optional[str] = None,
    llm_function: Optional[Callable] = None
) -> ComparisonResult:
    """Compare two outputs using LLM-as-judge.
    
    Args:
        output_a: First output to compare
        output_b: Second output to compare
        criterion: Comparison criterion
        context: Optional context (e.g., the prompt)
        llm_function: Function to call LLM
        
    Returns:
        ComparisonResult: Winner and reasoning
        
    Example:
        >>> result = pairwise_comparison(
        ...     "Python is great",
        ...     "Python is a high-level programming language",
        ...     "completeness"
        ... )
        >>> result.winner
        'b'
    """
    prompt = f"""Compare the following two outputs based on {criterion}.

Output A: {output_a}

Output B: {output_b}
"""
    
    if context:
        prompt = f"Context: {context}\n\n" + prompt
    
    prompt += "\nWhich output is better? Respond with 'A', 'B', or 'TIE' and explain why."
    
    if llm_function is not None:
        try:
            response = llm_function(prompt)
            winner, reasoning, confidence = _parse_comparison_response(response)
        except Exception as e:
            winner = None
            reasoning = f"Comparison failed: {str(e)}"
            confidence = 0.0
    else:
        # Heuristic comparison without LLM
        winner, reasoning, confidence = _heuristic_comparison(output_a, output_b, criterion)
    
    return ComparisonResult(
        output_a_id="a",
        output_b_id="b",
        winner=winner,
        scores={"a": 0.5 + (0.5 if winner == "a" else -0.25 if winner == "b" else 0.0),
                "b": 0.5 + (0.5 if winner == "b" else -0.25 if winner == "a" else 0.0)},
        confidence=confidence,
        reasoning=reasoning
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _parse_judge_response(response: str, scale: int) -> Tuple[float, str]:
    """Parse LLM judge response to extract rating and reasoning."""
    # Look for rating pattern
    rating_match = re.search(r'[Rr]ating:\s*(\d+(?:\.\d+)?)', response)
    reasoning_match = re.search(r'[Rr]easoning:\s*(.+)', response, re.DOTALL)
    
    if rating_match:
        rating = float(rating_match.group(1))
        rating = min(scale, max(1, rating))  # Clamp to scale
    else:
        # Try to find any number
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        if numbers:
            rating = float(numbers[0])
            rating = min(scale, max(1, rating))
        else:
            rating = scale / 2  # Default to middle
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response[:200]
    
    return rating, reasoning


def _parse_comparison_response(response: str) -> Tuple[Optional[str], str, float]:
    """Parse comparison response to extract winner, reasoning, and confidence."""
    response_upper = response.upper()
    
    # Determine winner
    if 'OUTPUT A' in response_upper or response_upper.startswith('A'):
        winner = 'a'
    elif 'OUTPUT B' in response_upper or response_upper.startswith('B'):
        winner = 'b'
    elif 'TIE' in response_upper:
        winner = None
    else:
        # Look for first A or B
        if 'A' in response_upper[:20]:
            winner = 'a'
        elif 'B' in response_upper[:20]:
            winner = 'b'
        else:
            winner = None
    
    # Extract reasoning
    reasoning = response
    
    # Estimate confidence based on language strength
    confidence_words = ['clearly', 'definitely', 'obviously', 'significantly', 'much better']
    confidence = 0.5
    
    for word in confidence_words:
        if word in response.lower():
            confidence += 0.1
    
    confidence = min(1.0, confidence)
    
    return winner, reasoning, confidence


def _heuristic_judge(
    output: str,
    criterion: str,
    context: Optional[str],
    reference: Optional[str],
    scale: int
) -> Tuple[float, str]:
    """Heuristic-based judgment when LLM is not available."""
    score = scale / 2  # Default to middle
    reasoning = f"Heuristic evaluation for {criterion}"
    
    if criterion == "relevance":
        if context:
            similarity = calculate_semantic_similarity(output, context, method="jaccard")
            score = 1 + (scale - 1) * similarity
            reasoning = f"Relevance based on overlap: {similarity:.2f}"
    
    elif criterion == "completeness":
        # Longer outputs score higher
        word_count = len(output.split())
        if word_count > 50:
            score = scale * 0.9
        elif word_count > 20:
            score = scale * 0.7
        else:
            score = scale * 0.5
        reasoning = f"Completeness based on length: {word_count} words"
    
    elif criterion == "coherence":
        result = assess_coherence(output)
        score = 1 + (scale - 1) * result.score
        reasoning = f"Coherence score: {result.score:.2f}"
    
    elif criterion == "accuracy":
        if reference:
            f1 = calculate_f1_score(output, reference)
            score = 1 + (scale - 1) * f1
            reasoning = f"Accuracy (F1) vs reference: {f1:.2f}"
    
    return score, reasoning


def _heuristic_comparison(
    output_a: str,
    output_b: str,
    criterion: str
) -> Tuple[Optional[str], str, float]:
    """Heuristic-based comparison when LLM is not available."""
    len_a = len(output_a.split())
    len_b = len(output_b.split())
    
    if criterion == "completeness":
        if len_a > len_b * 1.2:
            return 'a', f"Output A is more complete ({len_a} vs {len_b} words)", 0.7
        elif len_b > len_a * 1.2:
            return 'b', f"Output B is more complete ({len_b} vs {len_a} words)", 0.7
        else:
            return None, f"Both outputs similar in length ({len_a} vs {len_b} words)", 0.5
    
    elif criterion == "coherence":
        coherence_a = assess_coherence(output_a).score
        coherence_b = assess_coherence(output_b).score
        
        if coherence_a > coherence_b * 1.1:
            return 'a', f"Output A is more coherent ({coherence_a:.2f} vs {coherence_b:.2f})", 0.6
        elif coherence_b > coherence_a * 1.1:
            return 'b', f"Output B is more coherent ({coherence_b:.2f} vs {coherence_a:.2f})", 0.6
        else:
            return None, f"Both outputs similarly coherent ({coherence_a:.2f} vs {coherence_b:.2f})", 0.5
    
    else:
        # Default: compare by length
        if abs(len_a - len_b) < 5:
            return None, "Outputs are similar in length", 0.5
        elif len_a > len_b:
            return 'a', f"Output A is longer ({len_a} vs {len_b} words)", 0.5
        else:
            return 'b', f"Output B is longer ({len_b} vs {len_a} words)", 0.5
