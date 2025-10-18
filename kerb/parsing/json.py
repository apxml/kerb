"""JSON parsing and extraction utilities.

This module provides functions for extracting and parsing JSON from LLM outputs,
including automatic fixing of common formatting issues.
"""

import json
import re
from typing import Any, Dict, List, Optional

from .types import ParseMode, ParseResult


def extract_json(text: str, mode: ParseMode = ParseMode.LENIENT) -> ParseResult:
    """Extract JSON from text that may contain additional content.

    This function intelligently extracts JSON objects or arrays from LLM outputs
    that may include markdown formatting, explanatory text, or other artifacts.

    Args:
        text (str): Text containing JSON (may have markdown, explanations, etc.)
        mode (ParseMode): Parsing mode - strict, lenient, or best_effort

    Returns:
        ParseResult: Parsed JSON data and metadata

    Examples:
        >>> extract_json('Here is the data: {"name": "John", "age": 30}')
        ParseResult(success=True, data={'name': 'John', 'age': 30}, ...)

        >>> extract_json('```json\\n{"key": "value"}\\n```')
        ParseResult(success=True, data={'key': 'value'}, ...)
    """
    original = text
    warnings = []

    # Try direct parsing first
    try:
        data = json.loads(text)
        return ParseResult(success=True, data=data, original=original)
    except json.JSONDecodeError:
        pass

    # Extract from markdown code blocks
    json_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        for match in matches:
            try:
                data = json.loads(match)
                warnings.append("Extracted JSON from markdown code block")
                return ParseResult(
                    success=True, data=data, original=original, warnings=warnings
                )
            except json.JSONDecodeError:
                continue

    # Try to find JSON object or array in text
    # Look for outermost { } or [ ]
    json_patterns = [
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Nested objects
        r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]",  # Nested arrays
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                warnings.append("Extracted JSON from surrounding text")
                return ParseResult(
                    success=True, data=data, original=original, warnings=warnings
                )
            except json.JSONDecodeError:
                continue

    if mode == ParseMode.STRICT:
        return ParseResult(
            success=False, error="No valid JSON found in text", original=original
        )

    # Try fixing common issues
    if mode in [ParseMode.LENIENT, ParseMode.BEST_EFFORT]:
        fixed_result = fix_json(text)
        if fixed_result.success:
            return fixed_result

    return ParseResult(
        success=False,
        error="Could not extract or parse JSON from text",
        original=original,
        warnings=warnings,
    )


def parse_json(text: str, mode: ParseMode = ParseMode.LENIENT) -> ParseResult:
    """Parse JSON with automatic fixing for common LLM output issues.

    Args:
        text (str): JSON text to parse
        mode (ParseMode): Parsing mode - strict, lenient, or best_effort

    Returns:
        ParseResult: Parsed JSON data and metadata
    """
    return extract_json(text, mode)


def fix_json(text: str) -> ParseResult:
    """Attempt to fix common JSON formatting issues in LLM outputs.

    Common fixes:
    - Remove trailing commas
    - Fix single quotes to double quotes
    - Remove comments
    - Fix missing/extra brackets
    - Handle truncated JSON

    Args:
        text (str): Potentially malformed JSON text

    Returns:
        ParseResult: Fixed and parsed JSON if successful
    """
    original = text
    fixed = text
    fixes_applied = []

    # Remove markdown formatting
    fixed = re.sub(r"```(?:json)?\s*\n?", "", fixed)

    # Remove comments (// style and /* */ style)
    fixed = re.sub(r"//.*?$", "", fixed, flags=re.MULTILINE)
    fixed = re.sub(r"/\*.*?\*/", "", fixed, flags=re.DOTALL)

    # Fix single quotes to double quotes (be careful with apostrophes)
    # Only replace quotes that are likely JSON delimiters
    fixed = re.sub(r"(?<=[{\[,:])\s*'([^']*)'(?=\s*[,:\]}])", r'"\1"', fixed)

    # Remove trailing commas before closing brackets
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)

    # Try parsing
    try:
        data = json.loads(fixed)
        if fixed != original:
            fixes_applied.append("Applied automatic JSON fixes")
        return ParseResult(
            success=True,
            data=data,
            fixed=fixed != original,
            original=original,
            warnings=fixes_applied,
        )
    except json.JSONDecodeError as e:
        pass

    # Try to complete truncated JSON
    if text.count("{") > text.count("}"):
        fixed = fixed + "}" * (text.count("{") - text.count("}"))
        fixes_applied.append("Added missing closing braces")

    if text.count("[") > text.count("]"):
        fixed = fixed + "]" * (text.count("[") - text.count("]"))
        fixes_applied.append("Added missing closing brackets")

    try:
        data = json.loads(fixed)
        return ParseResult(
            success=True,
            data=data,
            fixed=True,
            original=original,
            warnings=fixes_applied,
        )
    except json.JSONDecodeError as e:
        return ParseResult(
            success=False,
            error=f"Could not fix JSON: {str(e)}",
            original=original,
            warnings=fixes_applied,
        )


def extract_json_array(text: str, mode: ParseMode = ParseMode.LENIENT) -> ParseResult:
    """Extract a JSON array from text.

    Args:
        text (str): Text containing JSON array
        mode (ParseMode): Parsing mode

    Returns:
        ParseResult: Parsed JSON array
    """
    result = extract_json(text, mode)

    if result.success and not isinstance(result.data, list):
        return ParseResult(
            success=False, error="Extracted JSON is not an array", original=text
        )

    return result


def extract_json_object(text: str, mode: ParseMode = ParseMode.LENIENT) -> ParseResult:
    """Extract a JSON object from text.

    Args:
        text (str): Text containing JSON object
        mode (ParseMode): Parsing mode

    Returns:
        ParseResult: Parsed JSON object
    """
    result = extract_json(text, mode)

    if result.success and not isinstance(result.data, dict):
        return ParseResult(
            success=False, error="Extracted JSON is not an object", original=text
        )

    return result


def ensure_json_output(text: str, default: Any = None) -> Any:
    """Extract JSON from text, returning default if parsing fails.

    Args:
        text (str): Text containing JSON
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    result = extract_json(text, mode=ParseMode.BEST_EFFORT)
    return result.data if result.success else default


def ensure_list_output(text: str, default: Optional[List] = None) -> List:
    """Extract JSON array from text, returning default if parsing fails.

    Args:
        text (str): Text containing JSON array
        default (List, optional): Default value if parsing fails

    Returns:
        Parsed list or default value
    """
    result = extract_json_array(text, mode=ParseMode.BEST_EFFORT)
    return result.data if result.success else (default or [])


def ensure_dict_output(text: str, default: Optional[Dict] = None) -> Dict:
    """Extract JSON object from text, returning default if parsing fails.

    Args:
        text (str): Text containing JSON object
        default (Dict, optional): Default value if parsing fails

    Returns:
        Parsed dict or default value
    """
    result = extract_json_object(text, mode=ParseMode.BEST_EFFORT)
    return result.data if result.success else (default or {})
