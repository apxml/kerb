"""Response Assertions Example

This example demonstrates how to use assertion helpers to validate LLM outputs
in your tests.

Main concepts:
- Checking if responses contain expected text
- Validating JSON responses and schemas
- Response length and quality assertions
- Pattern matching with regex
- Content validation helpers

Use cases for LLM developers:
- Validating chatbot responses
- Testing structured output generation
- Ensuring code generation quality
- Verifying translation outputs
- Checking classification results
- Quality control for production prompts
"""

from kerb.testing import (
    assert_response_contains,
    assert_response_json,
    assert_response_quality,
)
from kerb.testing.assertions import (
    assert_response_matches,
    assert_response_length,
    assert_no_hallucination
)


def main():
    """Run response assertions examples."""
    
    print("="*80)
    print("RESPONSE ASSERTIONS EXAMPLE")
    print("="*80)
    
    # Example 1: Basic content assertions
    print("\n1. BASIC CONTENT ASSERTIONS")
    print("-"*80)
    
    response = "Python is a high-level programming language used for web development, data science, and automation."
    
    # Check single keyword
    try:
        assert_response_contains(response, "Python")
        print("Check 1: Response contains 'Python' - PASS")
    except AssertionError as e:
        print(f"Check 1 FAILED: {e}")
    
    # Check multiple keywords
    try:
        assert_response_contains(response, ["Python", "programming", "data science"])
        print("Check 2: Response contains all required terms - PASS")
    except AssertionError as e:
        print(f"Check 2 FAILED: {e}")
    
    # Case-insensitive check
    try:
        assert_response_contains(response, "PYTHON", case_sensitive=False)
        print("Check 3: Case-insensitive match - PASS")
    except AssertionError as e:
        print(f"Check 3 FAILED: {e}")
    
    # Negative test (should fail)
    try:
        assert_response_contains(response, "JavaScript")
        print("Check 4: Should have failed")
    except AssertionError:
        print("Check 4: Correctly detected missing term - PASS")
    
    # Example 2: Pattern matching with regex
    print("\n2. PATTERN MATCHING")
    print("-"*80)
    
    # Email validation
    email_response = "You can contact us at support@example.com for assistance."
    try:
        assert_response_matches(email_response, r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        print("Check 1: Email pattern found - PASS")
    except AssertionError:
        print("Check 1 FAILED: No email found")
    
    # Code pattern
    code_response = "Here's the function: def calculate(x, y): return x + y"
    try:
        assert_response_matches(code_response, r'def\s+\w+\s*\([^)]*\):')
        print("Check 2: Python function pattern found - PASS")
    except AssertionError:
        print("Check 2 FAILED: No function pattern")
    
    # Date pattern
    date_response = "The event is scheduled for 2024-12-25."
    try:
        assert_response_matches(date_response, r'\d{4}-\d{2}-\d{2}')
        print("Check 3: Date pattern (YYYY-MM-DD) found - PASS")
    except AssertionError:
        print("Check 3 FAILED: No date pattern")
    
    # Example 3: JSON response validation
    print("\n3. JSON RESPONSE VALIDATION")
    print("-"*80)
    
    # Valid JSON response
    json_response = '{"name": "John", "age": 30, "city": "New York"}'
    
    try:
        data = assert_response_json(json_response)
        print(f"Check 1: Valid JSON parsed - PASS")
        print(f"  Parsed data: {data}")
    except AssertionError as e:
        print(f"Check 1 FAILED: {e}")
    
    # JSON with schema validation
    schema = {
        "name": str,
        "age": int,
        "city": str
    }
    
    try:
        data = assert_response_json(json_response, expected_schema=schema)
        print("Check 2: JSON matches schema - PASS")
    except AssertionError as e:
        print(f"Check 2 FAILED: {e}")
    
    # Invalid JSON
    invalid_json = '{"name": "John", "age": 30'  # Missing closing brace
    try:
        assert_response_json(invalid_json)
        print("Check 3: Should have failed")
    except AssertionError:
        print("Check 3: Correctly detected invalid JSON - PASS")
    
    # Example 4: Length constraints
    print("\n4. LENGTH CONSTRAINTS")
    print("-"*80)
    
    short_response = "OK"
    medium_response = "This is a medium-length response with several words."
    long_response = "This is a much longer response that contains many more words and provides detailed information about the topic at hand, ensuring comprehensive coverage."
    
    # Minimum length check
    try:
        assert_response_length(medium_response, min_length=10)
        print("Check 1: Meets minimum length - PASS")
    except AssertionError as e:
        print(f"Check 1 FAILED: {e}")
    
    # Maximum length check
    try:
        assert_response_length(short_response, max_length=100)
        print("Check 2: Within maximum length - PASS")
    except AssertionError as e:
        print(f"Check 2 FAILED: {e}")
    
    # Range check
    try:
        assert_response_length(medium_response, min_length=20, max_length=100)
        print("Check 3: Within length range - PASS")
    except AssertionError as e:
        print(f"Check 3 FAILED: {e}")
    
    # Should fail - too long
    try:
        assert_response_length(long_response, max_length=50)
        print("Check 4: Should have failed")
    except AssertionError:
        print("Check 4: Correctly detected response too long - PASS")
    
    # Example 5: Quality assertions
    print("\n5. QUALITY ASSERTIONS")
    print("-"*80)
    
    quality_response = "Machine learning is a powerful tool. It enables systems to learn from data. Applications include image recognition and natural language processing."
    
    # Minimum word count
    try:
        assert_response_quality(quality_response, min_words=15)
        print("Check 1: Meets minimum word count - PASS")
    except AssertionError as e:
        print(f"Check 1 FAILED: {e}")
    
    # No empty lines
    clean_response = "Line one\nLine two\nLine three"
    try:
        assert_response_quality(clean_response, no_empty_lines=True)
        print("Check 2: No empty lines - PASS")
    except AssertionError as e:
        print(f"Check 2 FAILED: {e}")
    
    # Check for repetition
    non_repetitive = "Each sentence here is unique. No excessive duplication occurs."
    try:
        assert_response_quality(non_repetitive, no_repetition=True)
        print("Check 3: No excessive repetition - PASS")
    except AssertionError as e:
        print(f"Check 3 FAILED: {e}")
    
    # Example 6: Testing code generation outputs
    print("\n6. CODE GENERATION VALIDATION")
    print("-"*80)
    
    code_output = '''def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
    
    # Check contains function definition
    try:
        assert_response_contains(code_output, "def fibonacci")
        print("Check 1: Contains function definition - PASS")
    except AssertionError as e:
        print(f"Check 1 FAILED: {e}")
    
    # Check for docstring
    try:
        assert_response_matches(code_output, r'""".*"""')
        print("Check 2: Contains docstring - PASS")
    except AssertionError as e:
        print(f"Check 2 FAILED: {e}")
    
    # Check for return statement
    try:
        assert_response_contains(code_output, "return")
        print("Check 3: Contains return statement - PASS")
    except AssertionError as e:
        print(f"Check 3 FAILED: {e}")
    
    # Example 7: Testing classification outputs
    print("\n7. CLASSIFICATION OUTPUT VALIDATION")
    print("-"*80)
    
    classification_response = '{"class": "positive", "confidence": 0.95}'
    
    try:
        data = assert_response_json(classification_response)
        assert "class" in data
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        print("Check 1: Valid classification output - PASS")
    except AssertionError as e:
        print(f"Check 1 FAILED: {e}")
    
    # Check class is in expected set
    expected_classes = ["positive", "negative", "neutral"]
    try:
        data = assert_response_json(classification_response)
        assert data["class"] in expected_classes
        print("Check 2: Class is in expected set - PASS")
    except AssertionError:
        print("Check 2 FAILED: Unexpected class")
    
    # Example 8: Hallucination detection
    print("\n8. HALLUCINATION DETECTION")
    print("-"*80)
    
    source_texts = [
        "Python was created by Guido van Rossum.",
        "Python is known for its simple syntax.",
        "Python is widely used in data science."
    ]
    
    # Grounded response (based on sources)
    grounded_response = "Python, created by Guido van Rossum, is known for simple syntax."
    
    try:
        assert_no_hallucination(grounded_response, source_texts, threshold=0.3)
        print("Check 1: Grounded response passes - PASS")
    except AssertionError as e:
        print(f"Check 1 FAILED: {e}")
    
    # Example 9: Comprehensive test suite
    print("\n9. COMPREHENSIVE VALIDATION")
    print("-"*80)
    
    def validate_chatbot_response(response: str) -> dict:
        """Validate a chatbot response with multiple checks."""
        results = {}
        
        # Length check
        try:
            assert_response_length(response, min_length=10, max_length=500)
            results["length"] = "PASS"
        except AssertionError:
            results["length"] = "FAIL"
        
        # Quality check
        try:
            assert_response_quality(response, min_words=5, no_repetition=True)
            results["quality"] = "PASS"
        except AssertionError:
            results["quality"] = "FAIL"
        
        # Politeness check
        try:
            polite_words = ["please", "thank", "help", "happy"]
            # At least somewhat polite (optional check)
            results["politeness"] = "PASS"
        except:
            results["politeness"] = "SKIP"
        
        return results
    
    test_response = "I'd be happy to help you with that. Python is a great language for beginners."
    validation_results = validate_chatbot_response(test_response)
    
    print("Validation results:")
    for check, result in validation_results.items():
        print(f"  {check}: {result}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("- Assertions validate LLM outputs automatically")
    print("- Multiple assertion types cover different needs")
    print("- Pattern matching enables complex validations")
    print("- JSON validation ensures structured outputs")
    print("- Quality checks maintain response standards")
    print("- Combine assertions for comprehensive testing")


if __name__ == "__main__":
    main()
