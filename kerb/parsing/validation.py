"""Output validation and fixing utilities.

This module provides comprehensive validation of LLM outputs and retry mechanisms
with progressive fixes.
"""

from typing import Any, Callable, Dict, Optional, Type

from .code import extract_code_blocks
from .json import (extract_json, extract_json_array, extract_json_object,
                   fix_json)
from .schema import validate_json_schema
from .types import ParseMode, ParseResult, ValidationResult


def validate_output(
    text: str,
    output_type: str,
    schema: Optional[Dict[str, Any]] = None,
    model_class: Optional[Type] = None,
    custom_validator: Optional[Callable[[Any], bool]] = None,
) -> ValidationResult:
    """Validate LLM output against expected format.

    Args:
        text (str): LLM output text
        output_type (str): Expected type ('json', 'json_array', 'json_object', 'pydantic', 'code', etc.)
        schema (Dict, optional): JSON Schema for validation
        model_class (Type, optional): Pydantic model class for validation
        custom_validator (Callable, optional): Custom validation function

    Returns:
        ValidationResult: Validation result with errors/warnings
    """
    errors = []
    warnings = []
    data = None

    # Parse based on type
    if output_type == "json":
        result = extract_json(text)
        if not result.success:
            errors.append(result.error)
        else:
            data = result.data
            warnings.extend(result.warnings)

    elif output_type == "json_array":
        result = extract_json_array(text)
        if not result.success:
            errors.append(result.error)
        else:
            data = result.data
            warnings.extend(result.warnings)

    elif output_type == "json_object":
        result = extract_json_object(text)
        if not result.success:
            errors.append(result.error)
        else:
            data = result.data
            warnings.extend(result.warnings)

    elif output_type == "pydantic":
        if not model_class:
            errors.append("model_class required for pydantic validation")
        else:
            from .pydantic import parse_to_pydantic

            result = parse_to_pydantic(text, model_class)
            if not result.success:
                errors.append(result.error)
            else:
                data = result.data
                warnings.extend(result.warnings)

    elif output_type == "code":
        blocks = extract_code_blocks(text)
        if not blocks:
            errors.append("No code blocks found in output")
        else:
            data = blocks

    else:
        errors.append(f"Unknown output_type: {output_type}")

    # Schema validation
    if schema and data and not errors:
        schema_result = validate_json_schema(
            data if not hasattr(data, "model_dump") else data.model_dump(), schema
        )
        if not schema_result.valid:
            errors.extend(schema_result.errors)

    # Custom validation
    if custom_validator and data and not errors:
        try:
            is_valid = custom_validator(data)
            if not is_valid:
                errors.append("Custom validation failed")
        except Exception as e:
            errors.append(f"Custom validation error: {str(e)}")

    return ValidationResult(
        valid=len(errors) == 0, errors=errors, warnings=warnings, data=data
    )


def retry_parse_with_fixes(
    text: str, parser_func: Callable[[str], ParseResult], max_attempts: int = 3
) -> ParseResult:
    """Retry parsing with increasingly aggressive fixes.

    Args:
        text (str): Text to parse
        parser_func (Callable): Parser function to use
        max_attempts (int): Maximum retry attempts

    Returns:
        ParseResult: Final parse result
    """
    modes = [ParseMode.STRICT, ParseMode.LENIENT, ParseMode.BEST_EFFORT]

    for i, mode in enumerate(modes[:max_attempts]):
        # If parser supports mode parameter
        try:
            result = parser_func(text, mode=mode)
            if result.success:
                return result
        except TypeError:
            # Parser doesn't support mode parameter
            result = parser_func(text)
            if result.success:
                return result

        # If not last attempt, try fixing
        if i < max_attempts - 1:
            if isinstance(text, str) and ("{" in text or "[" in text):
                fix_result = fix_json(text)
                if fix_result.success:
                    return fix_result

    return result
