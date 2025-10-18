"""Assertion helpers for testing responses."""

import re
import json
from typing import Union, List, Optional, Dict, Any


def assert_response_contains(
    response: str,
    expected: Union[str, List[str]],
    case_sensitive: bool = False
) -> None:
    """Assert that response contains expected text.
    
    Args:
        response: Response to check
        expected: Expected text or list of expected texts
        case_sensitive: Whether to do case-sensitive matching
    """
    if isinstance(expected, str):
        expected = [expected]
    
    check_response = response if case_sensitive else response.lower()
    
    for exp in expected:
        check_exp = exp if case_sensitive else exp.lower()
        assert check_exp in check_response, f"Response does not contain: {exp}"


def assert_response_matches(
    response: str,
    pattern: str,
    flags: int = 0
) -> None:
    """Assert that response matches regex pattern.
    
    Args:
        response: Response to check
        pattern: Regex pattern
        flags: Regex flags
    """
    assert re.search(pattern, response, flags), f"Response does not match pattern: {pattern}"


def assert_response_json(
    response: str,
    expected_schema: Optional[Dict] = None
) -> Dict[str, Any]:
    """Assert that response is valid JSON.
    
    Args:
        response: Response to check
        expected_schema: Optional JSON schema to validate against
        
    Returns:
        Parsed JSON data
    """
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Response is not valid JSON: {e}")
    
    if expected_schema:
        # Basic schema validation
        for key, expected_type in expected_schema.items():
            assert key in data, f"Missing required key: {key}"
            assert isinstance(data[key], expected_type), f"Invalid type for {key}"
    
    return data


def assert_response_length(
    response: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> None:
    """Assert response length constraints.
    
    Args:
        response: Response to check
        min_length: Minimum length (characters)
        max_length: Maximum length (characters)
    """
    length = len(response)
    
    if min_length is not None:
        assert length >= min_length, f"Response too short: {length} < {min_length}"
    
    if max_length is not None:
        assert length <= max_length, f"Response too long: {length} > {max_length}"


def assert_response_quality(
    response: str,
    min_words: Optional[int] = None,
    no_repetition: bool = False,
    no_empty_lines: bool = False
) -> None:
    """Assert response quality metrics.
    
    Args:
        response: Response to check
        min_words: Minimum word count
        no_repetition: Check for excessive repetition
        no_empty_lines: Check for empty lines
    """
    if min_words is not None:
        word_count = len(response.split())
        assert word_count >= min_words, f"Too few words: {word_count} < {min_words}"
    
    if no_repetition:
        # Check for repeated sequences
        words = response.split()
        for i in range(len(words) - 4):
            sequence = " ".join(words[i:i+3])
            rest = " ".join(words[i+3:])
            count = rest.count(sequence)
            assert count < 3, f"Excessive repetition detected: '{sequence}'"
    
    if no_empty_lines:
        lines = response.split("\n")
        assert all(line.strip() for line in lines), "Empty lines detected"


def assert_no_hallucination(
    response: str,
    source_texts: List[str],
    threshold: float = 0.8
) -> None:
    """Check for potential hallucinations.
    
    Args:
        response: Response to check
        source_texts: Source texts that response should be based on
        threshold: Similarity threshold for considering text grounded
    """
    from difflib import SequenceMatcher
    
    sentences = response.split(".")
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
        
        max_similarity = 0.0
        for source in source_texts:
            similarity = SequenceMatcher(None, sentence, source).ratio()
            max_similarity = max(max_similarity, similarity)
        
        assert max_similarity >= threshold, f"Potential hallucination: '{sentence}'"


def assert_safety_compliance(
    response: str,
    forbidden_terms: Optional[List[str]] = None,
    require_disclaimer: bool = False
) -> None:
    """Assert safety compliance.
    
    Args:
        response: Response to check
        forbidden_terms: Terms that should not appear
        require_disclaimer: Whether a disclaimer is required
    """
    if forbidden_terms:
        response_lower = response.lower()
        for term in forbidden_terms:
            assert term.lower() not in response_lower, f"Forbidden term detected: {term}"
    
    if require_disclaimer:
        disclaimer_patterns = [
            r"i('m| am) (an ai|a language model|not able to)",
            r"i cannot",
            r"i don't have",
            r"as an ai"
        ]
        found = any(re.search(p, response.lower()) for p in disclaimer_patterns)
        assert found, "Required disclaimer not found"
