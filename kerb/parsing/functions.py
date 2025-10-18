"""Function calling and tool use formatting.

This module provides utilities for formatting function definitions for LLM function
calling, parsing function calls from LLM outputs, and formatting results.
"""

import json
import re
from typing import Any, Dict, List, Optional

from .types import ParseMode, ParseResult
from .json import extract_json


def format_function_call(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Format a function definition for LLM function calling.
    
    Args:
        name (str): Function name
        description (str): Function description
        parameters (Dict): Parameter schema (JSON Schema format)
        required (List[str], optional): List of required parameter names
        
    Returns:
        Dict: Formatted function definition
        
    Examples:
        >>> format_function_call(
        ...     name="get_weather",
        ...     description="Get weather for a location",
        ...     parameters={"location": {"type": "string"}},
        ...     required=["location"]
        ... )
    """
    function_def = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": parameters,
        }
    }
    
    if required:
        function_def["parameters"]["required"] = required
    
    return function_def


def format_tool_call(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Format a tool definition for LLM tool use (OpenAI format).
    
    Args:
        name (str): Tool name
        description (str): Tool description
        parameters (Dict): Parameter schema (JSON Schema format)
        required (List[str], optional): List of required parameter names
        
    Returns:
        Dict: Formatted tool definition
    """
    return {
        "type": "function",
        "function": format_function_call(name, description, parameters, required)
    }


def parse_function_call(text: str, mode: ParseMode = ParseMode.LENIENT) -> ParseResult:
    """Parse a function call from LLM output.
    
    Extracts function name and arguments from various formats:
    - JSON format: {"name": "func", "arguments": {...}}
    - Plain format: func(arg1=val1, arg2=val2)
    - Markdown format with code blocks
    
    Args:
        text (str): Text containing function call
        mode (ParseMode): Parsing mode
        
    Returns:
        ParseResult: Parsed function call with name and arguments
    """
    original = text
    
    # Try JSON format first
    json_result = extract_json(text, mode)
    if json_result.success:
        data = json_result.data
        if isinstance(data, dict) and "name" in data:
            # OpenAI function call format
            result_data = {
                "name": data["name"],
                "arguments": data.get("arguments", {})
            }
            # If arguments is a string, try to parse it
            if isinstance(result_data["arguments"], str):
                try:
                    result_data["arguments"] = json.loads(result_data["arguments"])
                except json.JSONDecodeError:
                    pass
            
            return ParseResult(
                success=True,
                data=result_data,
                fixed=json_result.fixed,
                original=original,
                warnings=json_result.warnings
            )
    
    # Try plain function call format: func_name(arg1=val1, arg2=val2)
    pattern = r'(\w+)\((.*?)\)'
    match = re.search(pattern, text)
    
    if match:
        func_name = match.group(1)
        args_str = match.group(2)
        
        # Parse arguments
        arguments = {}
        if args_str.strip():
            # Simple key=value parsing
            arg_pairs = re.findall(r'(\w+)\s*=\s*([^,]+)', args_str)
            for key, value in arg_pairs:
                value = value.strip()
                # Try to parse value as JSON
                try:
                    arguments[key] = json.loads(value)
                except json.JSONDecodeError:
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        arguments[key] = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        arguments[key] = value[1:-1]
                    else:
                        arguments[key] = value
        
        return ParseResult(
            success=True,
            data={"name": func_name, "arguments": arguments},
            fixed=True,
            original=original,
            warnings=["Parsed plain function call format"]
        )
    
    return ParseResult(
        success=False,
        error="Could not parse function call from text",
        original=original
    )


def format_function_result(result: Any, name: Optional[str] = None) -> Dict[str, Any]:
    """Format a function result for returning to the LLM.
    
    Args:
        result: Function execution result
        name (str, optional): Function name
        
    Returns:
        Dict: Formatted function result
    """
    formatted = {
        "role": "function",
        "content": json.dumps(result) if not isinstance(result, str) else result
    }
    
    if name:
        formatted["name"] = name
    
    return formatted
