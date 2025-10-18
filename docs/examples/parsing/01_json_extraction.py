"""JSON Extraction from LLM Outputs

This example demonstrates how to extract and parse JSON from LLM responses
that may contain markdown formatting, explanatory text, or other artifacts.

Main concepts:
- Extracting JSON from mixed text content
- Handling markdown code blocks
- Using different parse modes (strict, lenient, best_effort)
- Working with parse results and warnings
"""

from kerb.parsing import extract_json, parse_json, ParseMode


def simulate_llm_response(prompt: str) -> str:
    """Simulate LLM responses with various JSON formatting styles."""
    if "user_data" in prompt:
        # Response with JSON in markdown
        return """Here's the user data you requested:

```json
{
  "name": "Alice Johnson",
  "email": "alice@example.com",
  "age": 28,
  "roles": ["developer", "team_lead"]
}
```

This data was extracted from the user database."""
    
    elif "settings" in prompt:
        # Response with plain JSON
        return '{"theme": "dark", "notifications": true, "language": "en"}'
    
    elif "config" in prompt:
        # Response with JSON embedded in text
        return 'The configuration is: {"api_key": "sk-xxx", "timeout": 30, "retries": 3} which should work.'
    
    else:
        return '{"status": "ok"}'


def main():
    """Run JSON extraction examples."""
    
    print("="*80)
    print("JSON EXTRACTION FROM LLM OUTPUTS")
    print("="*80)
    
    # Example 1: Extract JSON from markdown code block
    print("\nExample 1: JSON in Markdown Code Block")
    print("-"*80)
    
    llm_output = simulate_llm_response("get user_data")
    print(f"LLM Output:\n{llm_output}\n")
    
    result = extract_json(llm_output)
    print(f"Success: {result.success}")
    print(f"Data: {result.data}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    
    # Example 2: Extract plain JSON
    print("\n\nExample 2: Plain JSON Response")
    print("-"*80)
    
    llm_output = simulate_llm_response("get settings")
    print(f"LLM Output: {llm_output}\n")
    
    result = extract_json(llm_output)
    print(f"Success: {result.success}")
    print(f"Data: {result.data}")
    
    # Example 3: Extract JSON embedded in text
    print("\n\nExample 3: JSON Embedded in Text")
    print("-"*80)
    
    llm_output = simulate_llm_response("get config")
    print(f"LLM Output: {llm_output}\n")
    
    result = extract_json(llm_output, mode=ParseMode.LENIENT)
    print(f"Success: {result.success}")
    print(f"Data: {result.data}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    
    # Example 4: Using parse_json for automatic fixing
    print("\n\nExample 4: Parse JSON with Auto-Fixing")
    print("-"*80)
    
    # Malformed JSON (missing quotes, trailing comma)
    malformed = """{
        name: "Bob",
        age: 35,
        active: true,
    }"""
    
    print(f"Malformed JSON:\n{malformed}\n")
    
    result = parse_json(malformed, mode=ParseMode.LENIENT)
    print(f"Success: {result.success}")
    print(f"Data: {result.data}")
    print(f"Was Fixed: {result.fixed}")
    
    # Example 5: Parse modes comparison
    print("\n\nExample 5: Parse Mode Comparison")
    print("-"*80)
    
    text_with_json = "The result is: [1, 2, 3] and that's final."
    
    for mode in [ParseMode.STRICT, ParseMode.LENIENT, ParseMode.BEST_EFFORT]:
        result = extract_json(text_with_json, mode=mode)
        print(f"\n{mode.value.upper()} mode:")
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Data: {result.data}")
        else:
            print(f"  Error: {result.error}")
    
    # Example 6: Practical use case - parsing LLM tool responses
    print("\n\nExample 6: Parsing Tool Responses")
    print("-"*80)
    
    tool_response = """Based on the search, here are the results:

```json
{
  "query": "Python tutorials",
  "results": [
    {"title": "Python Basics", "url": "https://example.com/1", "score": 0.95},
    {"title": "Advanced Python", "url": "https://example.com/2", "score": 0.87}
  ],
  "total_count": 2
}
```

These are the most relevant results."""
    
    print(f"Tool Response:\n{tool_response}\n")
    
    result = extract_json(tool_response)
    if result.success:
        data = result.data
        print(f"Query: {data['query']}")
        print(f"Total Results: {data['total_count']}")
        print("\nResults:")
        for i, item in enumerate(data['results'], 1):
            print(f"  {i}. {item['title']} (score: {item['score']})")
            print(f"     URL: {item['url']}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
