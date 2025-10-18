"""Tests for the parsing module."""

import pytest

from kerb.parsing import (ParseMode, ValidationLevel, clean_llm_output,
                          ensure_dict_output, ensure_json_output,
                          ensure_list_output, extract_code_blocks,
                          extract_json, extract_json_array,
                          extract_json_object, extract_list_items,
                          extract_markdown_sections, extract_xml_tag, fix_json,
                          format_function_call, format_function_result,
                          format_tool_call, parse_function_call, parse_json,
                          parse_markdown_table, parse_to_pydantic,
                          pydantic_to_function, pydantic_to_schema,
                          retry_parse_with_fixes, validate_json_schema,
                          validate_output, validate_pydantic)

# ============================================================================
# JSON Extraction Tests
# ============================================================================


def test_extract_json_simple():
    """Test simple JSON extraction."""
    text = '{"name": "Alice", "age": 30}'
    result = extract_json(text)

    assert result.success
    assert result.data == {"name": "Alice", "age": 30}
    assert not result.fixed


def test_extract_json_from_markdown():
    """Test JSON extraction from markdown code blocks."""
    text = """
    Here's the data:
    ```json
    {"status": "success"}
    ```
    """
    result = extract_json(text)

    assert result.success
    assert result.data == {"status": "success"}


def test_extract_json_from_text():
    """Test JSON extraction from surrounding text."""
    text = 'The result is {"count": 42, "valid": true} and that is all.'
    result = extract_json(text)

    assert result.success
    assert result.data == {"count": 42, "valid": True}


def test_extract_json_array():
    """Test JSON array extraction."""
    text = "[1, 2, 3, 4, 5]"
    result = extract_json_array(text)

    assert result.success
    assert result.data == [1, 2, 3, 4, 5]


def test_extract_json_object():
    """Test JSON object extraction."""
    text = '{"key": "value"}'
    result = extract_json_object(text)

    assert result.success
    assert result.data == {"key": "value"}


def test_extract_json_array_wrong_type():
    """Test that extracting array fails for objects."""
    text = '{"key": "value"}'
    result = extract_json_array(text)

    assert not result.success
    assert "not an array" in result.error


def test_extract_json_object_wrong_type():
    """Test that extracting object fails for arrays."""
    text = "[1, 2, 3]"
    result = extract_json_object(text)

    assert not result.success
    assert "not an object" in result.error


def test_extract_json_fails():
    """Test extraction failure on invalid input."""
    text = "This is not JSON at all!"
    result = extract_json(text, mode=ParseMode.STRICT)

    assert not result.success
    assert result.error is not None


# ============================================================================
# JSON Fixing Tests
# ============================================================================


def test_fix_json_trailing_comma():
    """Test fixing trailing commas."""
    text = '{"name": "Bob", "age": 25,}'
    result = fix_json(text)

    assert result.success
    assert result.data == {"name": "Bob", "age": 25}
    assert result.fixed


def test_fix_json_single_quotes():
    """Test fixing single quotes (basic cases)."""
    text = "{'name': 'Charlie'}"
    result = fix_json(text)

    # Note: This is a simplified fix; complex cases may not work
    assert result.success
    assert result.data.get("name") == "Charlie"


def test_fix_json_comments():
    """Test removing comments."""
    text = """
    {
        "name": "Dave",  // This is a comment
        "age": 30
    }
    """
    result = fix_json(text)

    assert result.success
    assert result.data == {"name": "Dave", "age": 30}


def test_fix_json_truncated():
    """Test fixing truncated JSON (missing closing braces)."""
    text = '{"name": "Eve", "age": 28'
    result = fix_json(text)

    assert result.success
    assert result.data == {"name": "Eve", "age": 28}
    assert result.fixed


# ============================================================================
# Safe Extraction Tests
# ============================================================================


def test_ensure_json_output_success():
    """Test safe JSON extraction with valid input."""
    text = '{"key": "value"}'
    result = ensure_json_output(text)

    assert result == {"key": "value"}


def test_ensure_json_output_failure():
    """Test safe JSON extraction with invalid input returns default."""
    text = "Not JSON"
    result = ensure_json_output(text, default={"error": True})

    assert result == {"error": True}


def test_ensure_list_output():
    """Test safe list extraction."""
    valid = "[1, 2, 3]"
    assert ensure_list_output(valid) == [1, 2, 3]

    invalid = "Not a list"
    assert ensure_list_output(invalid, default=[]) == []


def test_ensure_dict_output():
    """Test safe dict extraction."""
    valid = '{"key": "value"}'
    assert ensure_dict_output(valid) == {"key": "value"}

    invalid = "Not a dict"
    assert ensure_dict_output(invalid, default={}) == {}


# ============================================================================
# Schema Validation Tests
# ============================================================================


def test_validate_json_schema_valid():
    """Test valid schema validation."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }

    data = {"name": "Alice", "age": 30}
    result = validate_json_schema(data, schema)

    # May fail if jsonschema not installed
    if "not installed" not in str(result.errors):
        assert result.valid


def test_validate_json_schema_invalid():
    """Test invalid schema validation."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }

    data = {"name": "Bob"}  # Missing age
    result = validate_json_schema(data, schema)

    # May fail if jsonschema not installed
    if "not installed" not in str(result.errors):
        assert not result.valid


# ============================================================================
# Pydantic Integration Tests
# ============================================================================


def test_parse_to_pydantic():
    """Test parsing to Pydantic model."""
    try:
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        text = '{"name": "Alice", "age": 30}'
        result = parse_to_pydantic(text, User)

        assert result.success
        assert result.data.name == "Alice"
        assert result.data.age == 30

    except ImportError:
        pytest.skip("Pydantic not installed")


def test_parse_to_pydantic_from_markdown():
    """Test parsing Pydantic from markdown."""
    try:
        from pydantic import BaseModel

        class Product(BaseModel):
            id: int
            name: str

        text = """
        ```json
        {"id": 1, "name": "Widget"}
        ```
        """
        result = parse_to_pydantic(text, Product)

        assert result.success
        assert result.data.id == 1
        assert result.data.name == "Widget"

    except ImportError:
        pytest.skip("Pydantic not installed")


def test_pydantic_to_schema():
    """Test converting Pydantic to JSON Schema."""
    try:
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            price: float

        schema = pydantic_to_schema(Item)

        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "price" in schema["properties"]

    except ImportError:
        pytest.skip("Pydantic not installed")


def test_validate_pydantic():
    """Test Pydantic validation."""
    try:
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        valid_data = {"name": "Bob", "age": 25}
        result = validate_pydantic(valid_data, User)
        assert result.valid

        invalid_data = {"name": "Charlie"}  # Missing age
        result = validate_pydantic(invalid_data, User)
        assert not result.valid

    except ImportError:
        pytest.skip("Pydantic not installed")


def test_pydantic_to_function():
    """Test converting Pydantic to function definition."""
    try:
        from pydantic import BaseModel

        class SearchParams(BaseModel):
            query: str
            limit: int

        func_def = pydantic_to_function(
            SearchParams, name="search", description="Search function"
        )

        assert func_def["name"] == "search"
        assert func_def["description"] == "Search function"
        assert "query" in func_def["parameters"]["properties"]
        assert "limit" in func_def["parameters"]["properties"]

    except ImportError:
        pytest.skip("Pydantic not installed")


# ============================================================================
# Function Calling Tests
# ============================================================================


def test_format_function_call():
    """Test formatting function call."""
    func_def = format_function_call(
        name="test_func",
        description="Test function",
        parameters={"arg1": {"type": "string"}},
        required=["arg1"],
    )

    assert func_def["name"] == "test_func"
    assert func_def["description"] == "Test function"
    assert "arg1" in func_def["parameters"]["properties"]
    assert func_def["parameters"]["required"] == ["arg1"]


def test_format_tool_call():
    """Test formatting tool call."""
    tool_def = format_tool_call(
        name="test_tool",
        description="Test tool",
        parameters={"arg1": {"type": "string"}},
        required=["arg1"],
    )

    assert tool_def["type"] == "function"
    assert tool_def["function"]["name"] == "test_tool"


def test_parse_function_call_json():
    """Test parsing function call from JSON."""
    text = '{"name": "get_weather", "arguments": {"location": "NYC"}}'
    result = parse_function_call(text)

    assert result.success
    assert result.data["name"] == "get_weather"
    assert result.data["arguments"]["location"] == "NYC"


def test_parse_function_call_plain():
    """Test parsing function call from plain format."""
    text = "get_weather(location=NYC, units=celsius)"
    result = parse_function_call(text)

    assert result.success
    assert result.data["name"] == "get_weather"
    assert "location" in result.data["arguments"]


def test_format_function_result():
    """Test formatting function result."""
    result = format_function_result({"temp": 72}, name="get_weather")

    assert result["role"] == "function"
    assert result["name"] == "get_weather"
    assert "temp" in result["content"]


# ============================================================================
# Code & Text Extraction Tests
# ============================================================================


def test_extract_code_blocks():
    """Test extracting code blocks from markdown."""
    text = """
    ```python
    print("hello")
    ```
    
    ```javascript
    console.log("hi");
    ```
    """

    blocks = extract_code_blocks(text)

    assert len(blocks) == 2
    assert blocks[0]["language"] == "python"
    assert blocks[1]["language"] == "javascript"


def test_extract_code_blocks_by_language():
    """Test filtering code blocks by language."""
    text = """
    ```python
    print("hello")
    ```
    
    ```javascript
    console.log("hi");
    ```
    """

    python_blocks = extract_code_blocks(text, language="python")

    assert len(python_blocks) == 1
    assert python_blocks[0]["language"] == "python"


def test_extract_xml_tag():
    """Test extracting XML tags."""
    text = "<answer>42</answer><thinking>Let me think</thinking>"

    answers = extract_xml_tag(text, "answer")
    thinking = extract_xml_tag(text, "thinking")

    assert len(answers) == 1
    assert answers[0] == "42"
    assert thinking[0] == "Let me think"


def test_extract_markdown_sections():
    """Test extracting markdown sections."""
    text = """
## Section 1
Content 1

## Section 2
Content 2
    """

    sections = extract_markdown_sections(text, heading_level=2)

    assert "Section 1" in sections
    assert "Section 2" in sections
    assert "Content 1" in sections["Section 1"]


def test_extract_list_items():
    """Test extracting list items."""
    text = """
- Item 1
- Item 2
- Item 3
    """

    items = extract_list_items(text, ordered=False)

    assert len(items) == 3
    assert "Item 1" in items


def test_parse_markdown_table():
    """Test parsing markdown table."""
    text = """
| Name  | Age |
|-------|-----|
| Alice | 30  |
| Bob   | 25  |
    """

    rows = parse_markdown_table(text)

    assert len(rows) == 2
    assert rows[0]["Name"] == "Alice"
    assert rows[0]["Age"] == "30"
    assert rows[1]["Name"] == "Bob"


# ============================================================================
# Output Validation Tests
# ============================================================================


def test_validate_output_json():
    """Test validating JSON output."""
    text = '{"key": "value"}'
    result = validate_output(text, output_type="json")

    assert result.valid
    assert result.data == {"key": "value"}


def test_validate_output_json_array():
    """Test validating JSON array output."""
    text = "[1, 2, 3]"
    result = validate_output(text, output_type="json_array")

    assert result.valid
    assert result.data == [1, 2, 3]


def test_validate_output_json_object():
    """Test validating JSON object output."""
    text = '{"key": "value"}'
    result = validate_output(text, output_type="json_object")

    assert result.valid
    assert isinstance(result.data, dict)


def test_validate_output_code():
    """Test validating code output."""
    text = """
    ```python
    print("hello")
    ```
    """
    result = validate_output(text, output_type="code")

    assert result.valid
    assert len(result.data) > 0


def test_validate_output_with_schema():
    """Test validating with schema."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    text = '{"name": "Alice"}'
    result = validate_output(text, output_type="json", schema=schema)

    # May pass or fail depending on jsonschema installation
    assert result.valid or "not installed" in str(result.errors)


# ============================================================================
# Utility Tests
# ============================================================================


def test_clean_llm_output():
    """Test cleaning LLM output."""
    text = """
    Sure, here's the JSON:
    ```json
    {"key": "value"}
    ```
    """

    cleaned = clean_llm_output(text)

    assert "Sure" not in cleaned
    assert "```" not in cleaned


def test_truncate_for_context():
    """Test truncating text - now uses preprocessing module."""
    from kerb.preprocessing import truncate_text

    text = "a" * 2000
    truncated = truncate_text(text, max_length=103, suffix="...")

    assert len(truncated) == 103
    assert truncated.endswith("...")


def test_split_by_delimiter_removed():
    """Test that split_by_delimiter was removed - use str.split() instead."""
    text = "Part 1---Part 2---Part 3"
    # Use standard library instead
    parts = [part.strip() for part in text.split("---") if part.strip()]

    assert len(parts) == 3
    assert "Part 1" in parts
    assert "Part 2" in parts


def test_retry_parse_with_fixes():
    """Test retry parsing with progressive fixes."""
    # Valid JSON should work on first try
    valid_json = '{"key": "value"}'
    result = retry_parse_with_fixes(valid_json, extract_json)

    assert result.success
    assert result.data == {"key": "value"}

    # Malformed JSON should be fixed
    malformed = '{"key": "value",}'
    result = retry_parse_with_fixes(malformed, extract_json)

    assert result.success


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_extract_json_empty_string():
    """Test extracting from empty string."""
    result = extract_json("")
    assert not result.success


def test_extract_json_nested():
    """Test extracting nested JSON."""
    text = '{"outer": {"inner": {"deep": "value"}}}'
    result = extract_json(text)

    assert result.success
    assert result.data["outer"]["inner"]["deep"] == "value"


def test_parse_function_call_invalid():
    """Test parsing invalid function call."""
    text = "This is not a function call"
    result = parse_function_call(text)

    assert not result.success


def test_extract_code_blocks_none():
    """Test extracting when no code blocks exist."""
    text = "Just plain text"
    blocks = extract_code_blocks(text)

    assert len(blocks) == 0


def test_parse_markdown_table_empty():
    """Test parsing empty table."""
    text = ""
    rows = parse_markdown_table(text)

    assert len(rows) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
