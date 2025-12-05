"""
Robust Parsing with Error Handling
==================================

This example demonstrates handling malformed LLM outputs with automatic
fixing, retry mechanisms, and progressive error recovery.

Main concepts:
- Handling malformed JSON output
- Using retry mechanisms with fixes
- Progressive error recovery strategies
- Validating and fixing common LLM output issues
"""

from kerb.parsing import (
    parse_json,
    fix_json,
    retry_parse_with_fixes,
    clean_llm_output,
    ParseMode
)


def simulate_malformed_llm_outputs() -> dict:
    """Generate various malformed LLM outputs for testing."""
    return {
        "missing_quotes": """{

# %%
# Setup and Imports
# -----------------
    name: Alice,
    age: 30,
    city: New York
}""",
        
        "trailing_commas": """{
    "name": "Bob",
    "age": 35,
    "hobbies": ["reading", "coding",],
    "active": true,
}""",
        
        "single_quotes": """{'name': 'Charlie', 'age': 28, 'status': 'active'}""",
        
        "mixed_artifacts": """Sure! Here's the data you requested:

```json
{
  "user": "Diana",
  "score": 95,
  "level": "expert"
}
```

Let me know if you need anything else!""",
        
        "incomplete_json": """{
    "task": "Complete the project",
    "priority": "high",
    "deadline": "2024-""",
        
        "comments_in_json": """{
    "name": "Eve",
    "role": "admin",
    "active": true
}""",
        
        "concatenated_json": """{"name": "Frank"}{"age": 40}""",
    }


def main():
    """Run robust parsing examples."""
    
    print("="*80)
    print("ROBUST PARSING WITH ERROR HANDLING")
    print("="*80)
    
    malformed_outputs = simulate_malformed_llm_outputs()
    
    # Example 1: Parse JSON with missing quotes
    print("\nExample 1: Missing Quotes - Auto Fix")
    print("-"*80)
    
    text = malformed_outputs["missing_quotes"]
    print(f"Malformed JSON:\n{text}\n")
    
    result = parse_json(text, mode=ParseMode.LENIENT)
    
    print(f"Success: {result.success}")
    if result.success:
        print(f"Data: {result.data}")
        print(f"Was Fixed: {result.fixed}")
    else:
        print(f"Error: {result.error}")
    
    # Example 2: Handle trailing commas
    print("\n\nExample 2: Trailing Commas - Auto Fix")
    print("-"*80)
    
    text = malformed_outputs["trailing_commas"]
    print(f"Malformed JSON:\n{text}\n")
    
    result = parse_json(text, mode=ParseMode.LENIENT)
    
    print(f"Success: {result.success}")
    if result.success:
        print(f"Data: {result.data}")
        print(f"Was Fixed: {result.fixed}")
    
    # Example 3: Single quotes to double quotes
    print("\n\nExample 3: Single Quotes - Auto Fix")
    print("-"*80)
    
    text = malformed_outputs["single_quotes"]
    print(f"Malformed JSON:\n{text}\n")
    
    result = parse_json(text, mode=ParseMode.LENIENT)
    
    print(f"Success: {result.success}")
    if result.success:
        print(f"Data: {result.data}")
        print(f"Was Fixed: {result.fixed}")
    
    # Example 4: Extract from mixed content
    print("\n\nExample 4: Mixed Content with Artifacts")
    print("-"*80)
    
    text = malformed_outputs["mixed_artifacts"]
    print(f"LLM Output:\n{text}\n")
    
    result = parse_json(text, mode=ParseMode.LENIENT)
    
    print(f"Success: {result.success}")
    if result.success:
        print(f"Data: {result.data}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
    
    # Example 5: Using fix_json directly
    print("\n\nExample 5: Direct JSON Fixing")
    print("-"*80)
    
    text = malformed_outputs["comments_in_json"]
    print(f"JSON with Comments:\n{text}\n")
    
    result = fix_json(text)
    
    print(f"Success: {result.success}")
    if result.success:
        print(f"Fixed Data: {result.data}")
    else:
        print(f"Error: {result.error}")
    
    # Example 6: Progressive retry with fixes
    print("\n\nExample 6: Retry Parse with Progressive Fixes")
    print("-"*80)
    
    text = malformed_outputs["trailing_commas"]
    print(f"Malformed JSON:\n{text}\n")
    
    # Use retry_parse_with_fixes with correct signature
    from kerb.parsing.json import extract_json
    result = retry_parse_with_fixes(
        text,
        parser_func=extract_json,
        max_attempts=3
    )
    
    print(f"Success: {result.success}")
    if result.success:
        print(f"Data: {result.data}")
        print(f"Was Fixed: {result.fixed}")
        if result.warnings:
            print(f"Fixes Applied: {result.warnings}")
    
    # Example 7: Clean LLM output artifacts
    print("\n\nExample 7: Clean LLM Output Artifacts")
    print("-"*80)
    
    messy_output = """Assistant: Here's the JSON you requested:

```json
{
    "result": "success"
}
```

Hope this helps!"""
    
    print(f"Messy Output:\n{messy_output}\n")
    
    cleaned = clean_llm_output(messy_output)
    print(f"Cleaned Output:\n{cleaned}")
    
    # Example 8: Parse mode comparison
    print("\n\nExample 8: Parse Mode Comparison")
    print("-"*80)
    
    text = malformed_outputs["trailing_commas"]
    
    for mode in [ParseMode.STRICT, ParseMode.LENIENT, ParseMode.BEST_EFFORT]:
        result = parse_json(text, mode=mode)
        print(f"\n{mode.value.upper()} mode:")
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Fixed: {result.fixed}")
        else:
            print(f"  Error: {result.error}")
    
    # Example 9: Handling incomplete JSON
    print("\n\nExample 9: Incomplete JSON Handling")
    print("-"*80)
    
    text = malformed_outputs["incomplete_json"]
    print(f"Incomplete JSON:\n{text}\n")
    
    # Try different modes
    for mode in [ParseMode.LENIENT, ParseMode.BEST_EFFORT]:
        result = parse_json(text, mode=mode)
        print(f"\n{mode.value} mode:")
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Data: {result.data}")
        else:
            print(f"  Error: {result.error}")
    
    # Example 10: Practical workflow - resilient parsing
    print("\n\nExample 10: Practical Resilient Parsing Workflow")
    print("-"*80)
    

# %%
# Resilient Parse
# ---------------

    def resilient_parse(text: str) -> dict:
        """Attempt to parse with progressive fallbacks."""
        
        # Try strict first
        result = parse_json(text, mode=ParseMode.STRICT)
        if result.success:
            return {"success": True, "data": result.data, "method": "strict"}
        
        # Try lenient
        result = parse_json(text, mode=ParseMode.LENIENT)
        if result.success:
            return {"success": True, "data": result.data, "method": "lenient", "fixed": True}
        
        # Try best effort with parser function
        from kerb.parsing.json import extract_json
        result = retry_parse_with_fixes(text, parser_func=extract_json, max_attempts=3)
        if result.success:
            return {"success": True, "data": result.data, "method": "retry_with_fixes", "fixed": True}
        
        return {"success": False, "error": result.error}
    
    # Test with various malformed outputs
    test_cases = ["trailing_commas", "single_quotes", "mixed_artifacts"]
    
    for case_name in test_cases:
        text = malformed_outputs[case_name]
        result = resilient_parse(text)
        
        print(f"\nTest Case: {case_name}")
        print(f"  Success: {result['success']}")
        if result['success']:
            print(f"  Method: {result['method']}")
            print(f"  Data: {result['data']}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
