"""
Output Validation Example
=========================

This example demonstrates how to validate and filter LLM outputs to ensure
they are safe, appropriate, and meet quality standards.

Main concepts:
- Validating LLM outputs against safety rules
- Filtering harmful content from outputs
- Ensuring safe JSON generation
- Detecting code injection in outputs
- Comprehensive output safety checks
"""

from kerb.safety import (
    validate_output,
    filter_output,
    check_output_safety,
    ensure_safe_json,
    detect_code_injection,
    SafetyLevel
)
from kerb.safety.moderation import check_toxicity as check_text_toxicity


def simulate_llm_output(output_type: str) -> str:
    """Simulate various LLM outputs for demonstration."""
    outputs = {
        "clean": "The capital of France is Paris. It has a population of over 2 million people.",
        "too_long": "x" * 600,  # Very long output
        "with_pii": "You can contact our support team at support@company.com or call 555-0123.",
        "with_profanity": "This damn feature is not working properly!",
        "toxic": "You're an idiot if you don't understand this simple concept.",
        "mixed_issues": "Email me at admin@site.com, this is stupid and won't work!",
        "code_injection": "Here's the solution: <script>alert('xss')</script>",
        "safe_json": '{"name": "John", "age": 30, "city": "Paris"}',
        "unsafe_json": '{"email": "user@example.com", "script": "<script>alert(1)</script>"}'
    }
    return outputs.get(output_type, "This is a safe output.")


def main():
    """Run output validation example."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("OUTPUT VALIDATION EXAMPLE")
    print("="*80)
    
    # Example 1: Basic output validation
    print("\nExample 1: Basic Output Validation")
    print("-"*80)
    
    clean_output = simulate_llm_output("clean")
    result = validate_output(clean_output, max_length=500, check_toxicity=False)
    
    print(f"Output: {clean_output}")
    print(f"Valid: {result.safe}")
    print(f"Score: {result.score:.3f}")
    
    # Example 2: Length validation
    print("\n\nExample 2: Length Validation")
    print("-"*80)
    
    long_output = simulate_llm_output("too_long")
    result = validate_output(long_output, max_length=500, check_toxicity=False)
    
    print(f"Output length: {len(long_output)} characters")
    print(f"Max allowed: 500 characters")
    print(f"Valid: {result.safe}")
    print(f"Reason: {result.reason}")
    
    # Example 3: PII detection in outputs
    print("\n\nExample 3: PII Detection in Outputs")
    print("-"*80)
    
    pii_output = simulate_llm_output("with_pii")
    result = validate_output(pii_output, check_pii=True, check_toxicity=False)
    
    print(f"Output: {pii_output}")
    print(f"Valid: {result.safe}")
    if not result.safe:
        print(f"Issues: {result.reason}")
    
    # Example 4: Filtering output content
    print("\n\nExample 4: Filtering Output Content")
    print("-"*80)
    
    test_outputs = [
        ("with_pii", "Output with PII"),
        ("with_profanity", "Output with profanity"),
        ("mixed_issues", "Output with mixed issues")
    ]
    
    for output_type, description in test_outputs:
        original = simulate_llm_output(output_type)
        filtered = filter_output(original, remove_pii=True, remove_profanity=True)
        
        print(f"\n{description}:")
        print(f"  Original: {original}")
        print(f"  Filtered: {filtered}")
    
    # Example 5: Comprehensive output safety check
    print("\n\nExample 5: Comprehensive Output Safety Check")
    print("-"*80)
    
    toxic_output = simulate_llm_output("toxic")
    result = check_output_safety(toxic_output, level=SafetyLevel.MODERATE)
    
    print(f"Output: {toxic_output}")
    print(f"Safe: {result.safe}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Toxicity Level: {result.toxicity_level.value}")
    if result.flagged_categories:
        print(f"Flagged: {[cat.value for cat in result.flagged_categories]}")
    
    # Example 6: Pattern-based validation
    print("\n\nExample 6: Pattern-Based Validation")
    print("-"*80)
    
    # Require specific format
    output1 = "Answer: The result is 42"
    output2 = "I think the result is 42"
    
    # Must start with "Answer:"
    required_pattern = r"^Answer:"
    
    for output in [output1, output2]:
        result = validate_output(output, allowed_patterns=[required_pattern], check_toxicity=False)
        print(f"\nOutput: {output}")
        print(f"Valid: {result.safe}")
        if not result.safe:
            print(f"Reason: {result.reason}")
    
    # Example 7: Blocked pattern validation
    print("\n\nExample 7: Blocked Pattern Validation")
    print("-"*80)
    
    # Block outputs mentioning competitors
    blocked_patterns = [r"\b(competitor|rival)\b"]
    
    output1 = "Our product is the best solution for you."
    output2 = "Unlike our competitor, we offer better features."
    
    for output in [output1, output2]:
        result = validate_output(output, blocked_patterns=blocked_patterns, check_toxicity=False)
        print(f"\nOutput: {output}")
        print(f"Valid: {result.safe}")
        if not result.safe:
            print(f"Reason: {result.reason}")
    
    # Example 8: Safe JSON validation
    print("\n\nExample 8: Safe JSON Validation")
    print("-"*80)
    
    json_outputs = [
        ("safe_json", "Clean JSON"),
        ("unsafe_json", "JSON with issues")
    ]
    
    for output_type, description in json_outputs:
        json_output = simulate_llm_output(output_type)
        result = ensure_safe_json(json_output, check_code=True, check_urls=True)
        
        print(f"\n{description}:")
        print(f"  JSON: {json_output}")
        print(f"  Safe: {result.safe}")
        if not result.safe:
            print(f"  Issues: {result.reason}")
    
    # Example 9: Code injection detection
    print("\n\nExample 9: Code Injection Detection")
    print("-"*80)
    
    code_output = simulate_llm_output("code_injection")
    result = detect_code_injection(code_output)
    
    print(f"Output: {code_output}")
    print(f"Safe: {result.safe}")
    print(f"Score: {result.score:.3f}")
    if not result.safe:
        print(f"Reason: {result.reason}")
    
    # Example 10: LLM output pipeline
    print("\n\nExample 10: LLM Output Pipeline")
    print("-"*80)
    

# %%
# Safe Llm Output Pipeline
# ------------------------

    def safe_llm_output_pipeline(raw_output: str) -> tuple:
        """Process LLM output through safety pipeline."""
        # Step 1: Validate (skip toxicity check to avoid bug)
        validation = validate_output(
            raw_output,
            max_length=1000,
            check_pii=True,
            check_toxicity=False
        )
        
        if not validation.safe:
            return None, f"Validation failed: {validation.reason}"
        
        # Step 2: Check comprehensive safety
        safety = check_output_safety(raw_output, level=SafetyLevel.MODERATE)
        
        if not safety.safe:
            return None, f"Safety check failed: {[cat.value for cat in safety.flagged_categories]}"
        
        # Step 3: Filter any remaining issues
        filtered = filter_output(raw_output, remove_pii=True, remove_profanity=True)
        
        return filtered, "OK"
    
    test_cases = ["clean", "with_pii", "toxic", "mixed_issues"]
    
    print("Processing outputs through safety pipeline:\n")
    
    for test_case in test_cases:
        raw_output = simulate_llm_output(test_case)
        filtered_output, status = safe_llm_output_pipeline(raw_output)
        
        print(f"Input: {raw_output[:60]}...")
        if filtered_output:
            print(f"Output: {filtered_output[:60]}...")
            print(f"Status: {status}\n")
        else:
            print(f"Output: BLOCKED")
            print(f"Status: {status}\n")
    
    print("="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
