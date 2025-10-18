"""Output validation and filtering functions.

This module provides functions for validating and filtering LLM outputs
to ensure they meet safety requirements.
"""

import json
import re
from typing import List, Optional

from .enums import ContentCategory, SafetyLevel
from .moderation import check_toxicity, moderate_content
from .patterns import PROFANITY_PATTERNS
from .pii import detect_pii, detect_url, redact_pii
from .security import validate_url_safety
from .types import ModerationResult, SafetyResult


def validate_output(
    text: str,
    max_length: Optional[int] = None,
    allowed_patterns: Optional[List[str]] = None,
    blocked_patterns: Optional[List[str]] = None,
    check_pii: bool = False,
    check_toxicity: bool = True,
) -> SafetyResult:
    """Validate LLM output against safety rules.

    Args:
        text: LLM output to validate
        max_length: Maximum allowed length
        allowed_patterns: Patterns that must be present
        blocked_patterns: Patterns that must not be present
        check_pii: Whether to check for PII
        check_toxicity: Whether to check for toxic content

    Returns:
        SafetyResult with validation assessment
    """
    issues = []
    score = 1.0

    # Check length
    if max_length and len(text) > max_length:
        issues.append(f"Output exceeds maximum length ({len(text)} > {max_length})")
        score -= 0.2

    # Check allowed patterns
    if allowed_patterns:
        for pattern in allowed_patterns:
            if not re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Required pattern not found: {pattern}")
                score -= 0.3

    # Check blocked patterns
    if blocked_patterns:
        for pattern in blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Blocked pattern found: {pattern}")
                score -= 0.4

    # Check PII
    if check_pii:
        pii_matches = detect_pii(text)
        if pii_matches:
            issues.append(f"Contains PII ({len(pii_matches)} matches)")
            score -= 0.3

    # Check toxicity
    if check_toxicity:
        toxicity_result = check_toxicity(text)
        if not toxicity_result.safe:
            issues.append("Contains toxic content")
            score -= 0.4

    score = max(0.0, score)
    safe = len(issues) == 0 and score >= 0.5

    return SafetyResult(
        safe=safe,
        score=score,
        category=ContentCategory.SAFE if safe else ContentCategory.MALICIOUS,
        confidence=0.8,
        reason="; ".join(issues) if issues else None,
        details={"issues": issues},
    )


def filter_output(
    text: str,
    remove_pii: bool = True,
    remove_profanity: bool = True,
    replacement: str = "[FILTERED]",
) -> str:
    """Filter or sanitize LLM output.

    Args:
        text: LLM output to filter
        remove_pii: Whether to remove PII
        remove_profanity: Whether to remove profanity
        replacement: Replacement text for filtered content

    Returns:
        Filtered text

    Examples:
        >>> output = "Email me at john@example.com, you damn fool!"
        >>> filtered = filter_output(output)
        >>> print(filtered)
        "Email me at [FILTERED], you [FILTERED] fool!"
    """
    filtered = text

    # Remove PII
    if remove_pii:
        filtered, _ = redact_pii(filtered, replacement=replacement)

    # Remove profanity - using pattern-based approach
    if remove_profanity:
        for severity, patterns in PROFANITY_PATTERNS.items():
            for pattern, weight in patterns:
                filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

    return filtered


def check_output_safety(
    text: str, level: SafetyLevel = SafetyLevel.MODERATE
) -> ModerationResult:
    """Comprehensive output safety check.

    Args:
        text: LLM output to check
        level: Safety strictness level

    Returns:
        ModerationResult with comprehensive assessment
    """
    return moderate_content(text, level=level)


def ensure_safe_json(
    json_str: str, check_code: bool = True, check_urls: bool = True
) -> SafetyResult:
    """Validate JSON output for safety.

    Args:
        json_str: JSON string to validate
        check_code: Whether to check for code injection
        check_urls: Whether to check for unsafe URLs

    Returns:
        SafetyResult with JSON safety assessment
    """
    issues = []
    score = 1.0

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return SafetyResult(
            safe=False,
            score=0.0,
            reason=f"Invalid JSON: {str(e)}",
        )

    # Check for code injection in values
    if check_code:

        def check_value(val):
            if isinstance(val, str):
                if detect_code_injection(val).safe is False:
                    issues.append("Potential code injection detected in JSON value")
                    return False
            elif isinstance(val, dict):
                return all(check_value(v) for v in val.values())
            elif isinstance(val, list):
                return all(check_value(v) for v in val)
            return True

        if not check_value(data):
            score -= 0.5

    # Check for unsafe URLs
    if check_urls:

        def check_urls_in_value(val):
            if isinstance(val, str):
                urls = detect_url(val)
                if urls:
                    for url_match in urls:
                        if not validate_url_safety(url_match.text).safe:
                            issues.append(f"Unsafe URL detected: {url_match.text}")
                            return False
            elif isinstance(val, dict):
                return all(check_urls_in_value(v) for v in val.values())
            elif isinstance(val, list):
                return all(check_urls_in_value(v) for v in val)
            return True

        if not check_urls_in_value(data):
            score -= 0.5

    score = max(0.0, score)
    safe = len(issues) == 0 and score >= 0.5

    return SafetyResult(
        safe=safe,
        score=score,
        reason="; ".join(issues) if issues else None,
        details={"issues": issues},
    )


def detect_code_injection(text: str) -> SafetyResult:
    """Detect code injection in outputs.

    Args:
        text: Text to check for code injection

    Returns:
        SafetyResult with code injection detection
    """
    dangerous_patterns = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__",
        r"subprocess",
        r"os\.system",
    ]

    matches = sum(
        1 for pattern in dangerous_patterns if re.search(pattern, text, re.IGNORECASE)
    )
    score = max(0.0, 1.0 - (matches / len(dangerous_patterns)))
    safe = matches == 0

    reason = (
        f"Detected potential code injection ({matches} patterns)" if not safe else None
    )

    return SafetyResult(
        safe=safe,
        score=score,
        category=ContentCategory.MALICIOUS,
        confidence=0.8,
        reason=reason,
        details={"matched_patterns": matches},
    )
