"""Parsing utilities for LLM output processing and validation.

This module provides comprehensive tools for parsing and validating LLM outputs.

Enums:
    ParseMode - Parsing strictness mode (STRICT, LENIENT, BEST_EFFORT)
    ValidationLevel - Validation strictness level
    
Data Classes:
    ParseResult - Result from parsing operation with data and metadata
    ValidationResult - Result from validation with errors and warnings
    
JSON Parsing:
    extract_json() - Extract JSON from text with markdown/text artifacts
    parse_json() - Parse JSON with automatic fixing
    fix_json() - Fix common JSON formatting issues
    extract_json_array() - Extract and validate JSON array
    extract_json_object() - Extract and validate JSON object
    ensure_json_output() - Extract JSON with fallback default
    ensure_list_output() - Extract list with fallback default
    ensure_dict_output() - Extract dict with fallback default
    
Schema Validation:
    validate_json_schema() - Validate against JSON Schema
    
Pydantic Integration:
    parse_to_pydantic() - Parse text to Pydantic model instance
    pydantic_to_schema() - Convert Pydantic model to JSON Schema
    validate_pydantic() - Validate data against Pydantic model
    pydantic_to_function() - Convert Pydantic model to function definition
    
Function Calling / Tool Use:
    format_function_call() - Format function definition for LLMs
    format_tool_call() - Format tool definition (OpenAI format)
    parse_function_call() - Parse function call from LLM output
    format_function_result() - Format function result for LLM
    
Code and Text Extraction:
    extract_code_blocks() - Extract code blocks from markdown
    extract_xml_tag() - Extract content from XML-style tags
    extract_markdown_sections() - Extract sections by heading
    extract_list_items() - Extract list items from markdown
    parse_markdown_table() - Parse markdown table to dicts
    
Output Validation:
    validate_output() - Comprehensive output validation
    retry_parse_with_fixes() - Retry parsing with progressive fixes
    
Utilities:
    clean_llm_output() - Clean common LLM output artifacts
    
Usage Examples:
    # Common imports
    from kerb.parsing import parse_json, extract_code_blocks
    
    # Submodule imports for specialized use
    from kerb.parsing.json import fix_json
    from kerb.parsing.pydantic import parse_to_pydantic
    from kerb.parsing.validation import validate_output
    
    For splitting by delimiter, use: text.split(delimiter)
    For text truncation, use the preprocessing module:
        from kerb.preprocessing import truncate_text
"""

# Core types (always available at top level)
from .types import (
    ParseMode,
    ValidationLevel,
    ParseResult,
    ValidationResult,
)

# Submodules (for specialized imports)
from . import json, schema, pydantic, functions, code, text, validation, utilities

# Most commonly used functions (top-level convenience imports)
from .json import (
    extract_json,
    parse_json,
    fix_json,
    extract_json_array,
    extract_json_object,
    ensure_json_output,
    ensure_list_output,
    ensure_dict_output,
)

from .schema import validate_json_schema

from .pydantic import (
    parse_to_pydantic,
    pydantic_to_schema,
    validate_pydantic,
    pydantic_to_function,
)

from .functions import (
    format_function_call,
    format_tool_call,
    parse_function_call,
    format_function_result,
)

from .code import extract_code_blocks

from .text import (
    extract_xml_tag,
    extract_markdown_sections,
    extract_list_items,
    parse_markdown_table,
)

from .validation import (
    validate_output,
    retry_parse_with_fixes,
)

from .utilities import clean_llm_output

__all__ = [
    # Enums
    "ParseMode",
    "ValidationLevel",
    
    # Data classes
    "ParseResult",
    "ValidationResult",
    
    # Submodules
    "json",
    "schema",
    "pydantic",
    "functions",
    "code",
    "text",
    "validation",
    "utilities",
    
    # JSON parsing
    "extract_json",
    "parse_json",
    "fix_json",
    "extract_json_array",
    "extract_json_object",
    "ensure_json_output",
    "ensure_list_output",
    "ensure_dict_output",
    
    # Schema validation
    "validate_json_schema",
    
    # Pydantic integration
    "parse_to_pydantic",
    "pydantic_to_schema",
    "validate_pydantic",
    "pydantic_to_function",
    
    # Function calling / tool use
    "format_function_call",
    "format_tool_call",
    "parse_function_call",
    "format_function_result",
    
    # Code and text extraction
    "extract_code_blocks",
    "extract_xml_tag",
    "extract_markdown_sections",
    "extract_list_items",
    "parse_markdown_table",
    
    # Output validation
    "validate_output",
    "retry_parse_with_fixes",
    
    # Utilities
    "clean_llm_output",
]

