"""Output validators for testing."""

import re
import json
import ast
from typing import Tuple, Optional, Dict, Any, List


def validate_json_schema(
    response: str,
    schema: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """Validate JSON response against schema.
    
    Args:
        response: JSON response string
        schema: JSON schema dict
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    
    # Basic schema validation
    for key, expected_type in schema.items():
        if key not in data:
            return False, f"Missing required key: {key}"
        
        if not isinstance(data[key], expected_type):
            return False, f"Invalid type for {key}: expected {expected_type.__name__}"
    
    return True, None


def validate_code_syntax(
    response: str,
    language: str = "python"
) -> Tuple[bool, Optional[str]]:
    """Validate code syntax.
    
    Args:
        response: Code string
        language: Programming language
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if language == "python":
        try:
            ast.parse(response)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    else:
        # For other languages, just check basic structure
        if response.strip():
            return True, None
        return False, "Empty code"


def validate_format(
    response: str,
    format_type: str
) -> Tuple[bool, Optional[str]]:
    """Validate response format.
    
    Args:
        response: Response to validate
        format_type: Format type (email, url, phone, etc.)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    patterns = {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "url": r"^https?://[^\s]+$",
        "phone": r"^\+?1?\d{9,15}$",
        "date": r"^\d{4}-\d{2}-\d{2}$",
    }
    
    pattern = patterns.get(format_type)
    if not pattern:
        return False, f"Unknown format type: {format_type}"
    
    if re.match(pattern, response.strip()):
        return True, None
    return False, f"Invalid {format_type} format"


def validate_consistency(
    responses: List[str],
    similarity_threshold: float = 0.8
) -> Tuple[bool, Optional[str]]:
    """Check consistency across multiple generations.
    
    Args:
        responses: List of responses to compare
        similarity_threshold: Minimum similarity required
        
    Returns:
        Tuple of (is_consistent, error_message)
    """
    if len(responses) < 2:
        return True, None
    
    from difflib import SequenceMatcher
    
    # Compare all pairs
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            similarity = SequenceMatcher(None, responses[i], responses[j]).ratio()
            if similarity < similarity_threshold:
                return False, f"Low consistency: {similarity:.2f} < {similarity_threshold}"
    
    return True, None
