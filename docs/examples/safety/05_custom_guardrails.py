"""Custom Guardrails Example

This example demonstrates how to create and apply custom safety guardrails
and content policies for LLM applications with specific requirements.

Main concepts:
- Creating custom guardrails with specific rules
- Applying multiple guardrails to content
- Defining and enforcing content policies
- Validating against custom rule sets
- Building domain-specific safety checks
"""

from kerb.safety import (
    create_guardrail,
    apply_guardrails,
    check_content_policy,
    validate_against_rules,
    SafetyResult,
    ContentCategory,
    detect_pii
)


def create_length_guardrail(min_length: int, max_length: int):
    """Create a guardrail for content length."""
    def check_length(text: str) -> SafetyResult:
        length = len(text)
        safe = min_length <= length <= max_length
        score = 1.0 if safe else 0.0
        reason = None if safe else f"Length {length} not in range [{min_length}, {max_length}]"
        return SafetyResult(safe=safe, score=score, reason=reason)
    
    return create_guardrail(
        name="length_check",
        check_function=check_length,
        description=f"Ensures text length is between {min_length} and {max_length}"
    )


def create_topic_guardrail(allowed_topics: list, blocked_topics: list):
    """Create a guardrail for topic restrictions."""
    def check_topics(text: str) -> SafetyResult:
        text_lower = text.lower()
        
        # Check for blocked topics
        for topic in blocked_topics:
            if topic.lower() in text_lower:
                return SafetyResult(
                    safe=False,
                    score=0.0,
                    reason=f"Contains blocked topic: {topic}"
                )
        
        # Check for allowed topics (if specified)
        if allowed_topics:
            has_allowed = any(topic.lower() in text_lower for topic in allowed_topics)
            if not has_allowed:
                return SafetyResult(
                    safe=False,
                    score=0.3,
                    reason="Does not contain any allowed topics"
                )
        
        return SafetyResult(safe=True, score=1.0)
    
    return create_guardrail(
        name="topic_check",
        check_function=check_topics,
        description="Ensures content matches topic restrictions"
    )


def create_format_guardrail(required_format: str):
    """Create a guardrail for output format."""
    def check_format(text: str) -> SafetyResult:
        import re
        matches = re.match(required_format, text)
        safe = matches is not None
        score = 1.0 if safe else 0.0
        reason = None if safe else f"Does not match required format: {required_format}"
        return SafetyResult(safe=safe, score=score, reason=reason)
    
    return create_guardrail(
        name="format_check",
        check_function=check_format,
        description=f"Ensures text matches format: {required_format}"
    )


def create_no_pii_guardrail():
    """Create a guardrail that blocks PII."""
    def check_no_pii(text: str) -> SafetyResult:
        pii_matches = detect_pii(text)
        safe = len(pii_matches) == 0
        score = 1.0 if safe else 0.0
        reason = None if safe else f"Contains {len(pii_matches)} PII instance(s)"
        return SafetyResult(safe=safe, score=score, reason=reason)
    
    return create_guardrail(
        name="no_pii",
        check_function=check_no_pii,
        description="Ensures no PII is present in content"
    )


def main():
    """Run custom guardrails example."""
    
    print("="*80)
    print("CUSTOM GUARDRAILS EXAMPLE")
    print("="*80)
    
    # Example 1: Creating custom guardrails
    print("\nExample 1: Creating Custom Guardrails")
    print("-"*80)
    
    # Create various guardrails
    length_guard = create_length_guardrail(min_length=10, max_length=200)
    topic_guard = create_topic_guardrail(
        allowed_topics=["technology", "science", "education"],
        blocked_topics=["politics", "religion"]
    )
    format_guard = create_format_guardrail(r"^Answer: .+")
    pii_guard = create_no_pii_guardrail()
    
    print("Created guardrails:")
    for guard in [length_guard, topic_guard, format_guard, pii_guard]:
        print(f"  - {guard.name}: {guard.description}")
    
    # Example 2: Applying single guardrail
    print("\n\nExample 2: Applying Single Guardrail")
    print("-"*80)
    
    test_texts = [
        "Short",  # Too short
        "This is a good length response about technology and science.",  # Good
        "x" * 300  # Too long
    ]
    
    for text in test_texts:
        result = length_guard.check_function(text)
        display_text = text if len(text) < 50 else f"{text[:50]}... (length: {len(text)})"
        print(f"\nText: {display_text}")
        print(f"Safe: {result.safe}")
        if not result.safe:
            print(f"Reason: {result.reason}")
    
    # Example 3: Applying multiple guardrails
    print("\n\nExample 3: Applying Multiple Guardrails")
    print("-"*80)
    
    guardrails = [length_guard, topic_guard, pii_guard]
    
    test_cases = [
        "Let's discuss technology innovations in education.",
        "Politics and religion are controversial topics.",
        "Contact me at john@example.com for tech advice.",
        "Too short"
    ]
    
    for text in test_cases:
        results = apply_guardrails(text, guardrails)
        
        print(f"\nText: {text}")
        all_safe = all(r.safe for r in results.values())
        print(f"All Safe: {all_safe}")
        
        for name, result in results.items():
            status = "PASS" if result.safe else "FAIL"
            print(f"  {name}: {status}", end="")
            if not result.safe:
                print(f" - {result.reason}")
            else:
                print()
    
    # Example 4: Content policy enforcement
    print("\n\nExample 4: Content Policy Enforcement")
    print("-"*80)
    
    # Define a content policy for a customer service chatbot
    customer_service_policy = {
        'max_length': 500,
        'blocked_words': ['competitor', 'lawsuit', 'sue'],
        'required_phrases': ['thank you', 'help'],
        'allow_pii': False
    }
    
    test_responses = [
        "Thank you for contacting us! I'm here to help with your question.",
        "Our competitor offers similar features.",
        "I'll help you with that issue right away.",
        "Contact me at support@company.com and I'll help you."
    ]
    
    for response in test_responses:
        result = check_content_policy(response, customer_service_policy)
        print(f"\nResponse: {response}")
        print(f"Compliant: {result.safe}")
        print(f"Score: {result.score:.3f}")
        if not result.safe:
            print(f"Issues: {result.reason}")
    
    # Example 5: Custom rule set validation
    print("\n\nExample 5: Custom Rule Set Validation")
    print("-"*80)
    
    # Define rules for financial advice responses
    financial_rules = [
        lambda t: len(t.split()) >= 10,  # Minimum 10 words
        lambda t: "disclaimer" in t.lower(),  # Must include disclaimer
        lambda t: not any(word in t.lower() for word in ["guaranteed", "promise", "certain"]),  # No guarantees
        lambda t: "investment" in t.lower() or "advice" in t.lower()  # Must be about finance
    ]
    
    rule_names = [
        "minimum_length",
        "has_disclaimer",
        "no_guarantees",
        "financial_context"
    ]
    
    test_advice = [
        "Invest now! Guaranteed returns!",
        "Consider diversifying your portfolio. Disclaimer: This is not professional financial advice.",
        "Stocks are good. Buy them.",
        "Based on market analysis, investment in diverse assets may be beneficial. Disclaimer: consult a professional."
    ]
    
    for advice in test_advice:
        result = validate_against_rules(advice, financial_rules, rule_names)
        print(f"\nAdvice: {advice}")
        print(f"Valid: {result.safe}")
        if not result.safe:
            print(f"Failed rules: {result.reason}")
    
    # Example 6: Domain-specific guardrail system
    print("\n\nExample 6: Domain-Specific Guardrail System")
    print("-"*80)
    
    # Medical chatbot guardrails
    def create_medical_guardrails():
        """Create guardrails for medical chatbot."""
        
        def no_diagnosis(text: str) -> SafetyResult:
            diagnosis_terms = ["you have", "you are diagnosed", "you suffer from"]
            text_lower = text.lower()
            has_diagnosis = any(term in text_lower for term in diagnosis_terms)
            safe = not has_diagnosis
            reason = "Contains medical diagnosis language" if has_diagnosis else None
            return SafetyResult(safe=safe, score=0.0 if has_diagnosis else 1.0, reason=reason)
        
        def requires_disclaimer(text: str) -> SafetyResult:
            has_disclaimer = "consult" in text.lower() and "doctor" in text.lower()
            safe = has_disclaimer
            reason = "Missing 'consult a doctor' disclaimer" if not has_disclaimer else None
            return SafetyResult(safe=safe, score=1.0 if has_disclaimer else 0.0, reason=reason)
        
        return [
            create_guardrail("no_diagnosis", no_diagnosis, "Prevents direct diagnosis"),
            create_guardrail("requires_disclaimer", requires_disclaimer, "Requires medical disclaimer")
        ]
    
    medical_guards = create_medical_guardrails()
    
    medical_responses = [
        "You have diabetes based on your symptoms.",
        "Your symptoms may indicate various conditions. Please consult a doctor for proper diagnosis.",
        "These symptoms are common with the flu.",
        "It could be several things. Consult your doctor to get proper medical advice."
    ]
    
    print("Testing medical chatbot responses:\n")
    
    for response in medical_responses:
        results = apply_guardrails(response, medical_guards)
        all_safe = all(r.safe for r in results.values())
        
        print(f"Response: {response}")
        print(f"Safe: {all_safe}")
        
        for name, result in results.items():
            if not result.safe:
                print(f"  Issue ({name}): {result.reason}")
        print()
    
    # Example 7: Combining guardrails with policies
    print("\nExample 7: Combining Guardrails with Policies")
    print("-"*80)
    
    # Educational content policy
    education_policy = {
        'max_length': 300,
        'blocked_words': ['buy', 'purchase', 'subscribe'],
        'allow_pii': False
    }
    
    # Educational content guardrails
    education_guards = [
        create_length_guardrail(20, 300),
        create_topic_guardrail(
            allowed_topics=["learn", "study", "education", "knowledge"],
            blocked_topics=["advertisement", "promotion"]
        )
    ]
    
    content = "Learn Python programming with our comprehensive course. Study at your own pace."
    
    print(f"Content: {content}\n")
    
    # Check policy
    policy_result = check_content_policy(content, education_policy)
    print(f"Policy Check: {policy_result.safe} (score: {policy_result.score:.3f})")
    
    # Check guardrails
    guard_results = apply_guardrails(content, education_guards)
    all_guards_pass = all(r.safe for r in guard_results.values())
    print(f"Guardrails Check: {all_guards_pass}")
    
    overall_safe = policy_result.safe and all_guards_pass
    print(f"\nOverall Result: {'APPROVED' if overall_safe else 'REJECTED'}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
