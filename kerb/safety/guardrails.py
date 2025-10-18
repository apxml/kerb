"""Guardrails and policy enforcement functions.

This module provides functions for creating and applying custom safety guardrails
and content policies.
"""

from typing import List, Dict, Any, Callable

from .types import Guardrail, SafetyResult
from .pii import detect_pii


def create_guardrail(
    name: str,
    check_function: Callable[[str], SafetyResult],
    description: str = None
) -> Guardrail:
    """Create a custom safety guardrail.
    
    Args:
        name: Guardrail name
        check_function: Function that takes text and returns SafetyResult
        description: Optional description
        
    Returns:
        Guardrail object
        
    Examples:
        >>> def no_caps(text):
        ...     has_caps = any(c.isupper() for c in text)
        ...     return SafetyResult(safe=not has_caps, score=0.0 if has_caps else 1.0)
        >>> guardrail = create_guardrail("no_caps", no_caps, "Reject all caps")
    """
    return Guardrail(
        name=name,
        check_function=check_function,
        description=description
    )


def apply_guardrails(
    text: str,
    guardrails: List[Guardrail]
) -> Dict[str, SafetyResult]:
    """Apply multiple guardrails to content.
    
    Args:
        text: Text to check
        guardrails: List of Guardrail objects
        
    Returns:
        Dictionary mapping guardrail names to results
        
    Examples:
        >>> guardrails = [guardrail1, guardrail2]
        >>> results = apply_guardrails(text, guardrails)
        >>> all_safe = all(r.safe for r in results.values())
    """
    results = {}
    for guardrail in guardrails:
        if guardrail.enabled:
            results[guardrail.name] = guardrail.check_function(text)
    return results


def check_content_policy(
    text: str,
    policy: Dict[str, Any]
) -> SafetyResult:
    """Check against custom content policy.
    
    Args:
        text: Text to check
        policy: Policy dictionary with rules
        
    Returns:
        SafetyResult with policy check assessment
        
    Example policy:
        {
            'max_length': 1000,
            'blocked_words': ['spam', 'scam'],
            'required_phrases': ['terms of service'],
            'allow_pii': False
        }
    """
    issues = []
    score = 1.0
    
    # Check max length
    if 'max_length' in policy:
        if len(text) > policy['max_length']:
            issues.append(f"Exceeds max length ({len(text)} > {policy['max_length']})")
            score -= 0.3
    
    # Check blocked words
    if 'blocked_words' in policy:
        text_lower = text.lower()
        for word in policy['blocked_words']:
            if word.lower() in text_lower:
                issues.append(f"Contains blocked word: {word}")
                score -= 0.4
    
    # Check required phrases
    if 'required_phrases' in policy:
        text_lower = text.lower()
        for phrase in policy['required_phrases']:
            if phrase.lower() not in text_lower:
                issues.append(f"Missing required phrase: {phrase}")
                score -= 0.3
    
    # Check PII
    if 'allow_pii' in policy and not policy['allow_pii']:
        pii_matches = detect_pii(text)
        if pii_matches:
            issues.append(f"Contains PII (policy disallows)")
            score -= 0.5
    
    score = max(0.0, score)
    safe = len(issues) == 0 and score >= 0.5
    
    return SafetyResult(
        safe=safe,
        score=score,
        reason="; ".join(issues) if issues else None,
        details={'issues': issues, 'policy': policy}
    )


def validate_against_rules(
    text: str,
    rules: List[Callable[[str], bool]],
    rule_names: List[str] = None
) -> SafetyResult:
    """Validate content against rule set.
    
    Args:
        text: Text to validate
        rules: List of rule functions (return True if valid)
        rule_names: Optional names for rules
        
    Returns:
        SafetyResult with validation assessment
        
    Examples:
        >>> rules = [
        ...     lambda t: len(t) < 1000,
        ...     lambda t: '@' not in t,
        ...     lambda t: t.strip() == t
        ... ]
        >>> result = validate_against_rules(text, rules)
    """
    if rule_names is None:
        rule_names = [f"rule_{i}" for i in range(len(rules))]
    
    failed_rules = []
    for i, rule in enumerate(rules):
        try:
            if not rule(text):
                failed_rules.append(rule_names[i])
        except Exception as e:
            failed_rules.append(f"{rule_names[i]} (error: {str(e)})")
    
    score = 1.0 - (len(failed_rules) / len(rules)) if rules else 1.0
    safe = len(failed_rules) == 0
    
    return SafetyResult(
        safe=safe,
        score=score,
        reason=f"Failed rules: {', '.join(failed_rules)}" if failed_rules else None,
        details={'failed_rules': failed_rules, 'total_rules': len(rules)}
    )
