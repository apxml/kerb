"""Response comparison and diff utilities."""

import difflib
from typing import Dict, Any, List


def diff_responses(
    response1: str,
    response2: str,
    context_lines: int = 3
) -> str:
    """Generate diff between two responses.
    
    Args:
        response1: First response
        response2: Second response
        context_lines: Number of context lines
        
    Returns:
        Diff string
    """
    diff = difflib.unified_diff(
        response1.splitlines(keepends=True),
        response2.splitlines(keepends=True),
        fromfile="response1",
        tofile="response2",
        n=context_lines
    )
    return "".join(diff)


def compare_responses(
    responses: Dict[str, str]
) -> Dict[str, Any]:
    """Compare multiple responses.
    
    Args:
        responses: Dict mapping labels to responses
        
    Returns:
        Comparison results
    """
    from difflib import SequenceMatcher
    
    labels = list(responses.keys())
    similarities = {}
    
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            similarity = SequenceMatcher(
                None,
                responses[label1],
                responses[label2]
            ).ratio()
            similarities[f"{label1}_vs_{label2}"] = similarity
    
    return {
        "similarities": similarities,
        "most_similar": max(similarities.items(), key=lambda x: x[1]) if similarities else None,
        "least_similar": min(similarities.items(), key=lambda x: x[1]) if similarities else None
    }


def highlight_differences(
    response1: str,
    response2: str
) -> List[Dict[str, str]]:
    """Highlight key differences between responses.
    
    Args:
        response1: First response
        response2: Second response
        
    Returns:
        List of differences with context
    """
    matcher = difflib.SequenceMatcher(None, response1, response2)
    differences = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            differences.append({
                "type": tag,
                "response1": response1[i1:i2],
                "response2": response2[j1:j2],
                "position": (i1, i2, j1, j2)
            })
    
    return differences
