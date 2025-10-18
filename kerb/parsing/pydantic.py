"""Pydantic model integration for parsing.

This module provides functions for parsing LLM outputs to Pydantic models,
validation, and schema conversion.
"""

from typing import Any, Dict, Optional, Type

from .json import extract_json
from .types import ParseMode, ParseResult, ValidationResult


def parse_to_pydantic(
    text: str, model_class: Type, mode: ParseMode = ParseMode.LENIENT
) -> ParseResult:
    """Parse text to a Pydantic model instance.

    Args:
        text (str): Text containing JSON data
        model_class: Pydantic model class to parse into
        mode (ParseMode): Parsing mode

    Returns:
        ParseResult: Parsed Pydantic model instance

    Examples:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> parse_to_pydantic('{"name": "John", "age": 30}', User)
        ParseResult(success=True, data=User(name='John', age=30), ...)
    """
    try:
        # Check if it's a Pydantic model
        from pydantic import BaseModel

        if not issubclass(model_class, BaseModel):
            return ParseResult(
                success=False,
                error=f"{model_class.__name__} is not a Pydantic model",
                original=text,
            )
    except ImportError:
        return ParseResult(
            success=False,
            error="pydantic package not installed. Install with: pip install pydantic",
            original=text,
        )

    # Extract JSON
    json_result = extract_json(text, mode)

    if not json_result.success:
        return ParseResult(
            success=False,
            error=f"Could not extract JSON: {json_result.error}",
            original=text,
            warnings=json_result.warnings,
        )

    # Parse to Pydantic model
    try:
        instance = model_class.model_validate(json_result.data)
        return ParseResult(
            success=True,
            data=instance,
            fixed=json_result.fixed,
            original=text,
            warnings=json_result.warnings,
        )
    except Exception as e:
        return ParseResult(
            success=False,
            error=f"Pydantic validation error: {str(e)}",
            original=text,
            warnings=json_result.warnings,
        )


def pydantic_to_schema(model_class: Type) -> Dict[str, Any]:
    """Convert a Pydantic model to JSON Schema.

    Args:
        model_class: Pydantic model class

    Returns:
        Dict: JSON Schema representation
    """
    try:
        from pydantic import BaseModel

        if not issubclass(model_class, BaseModel):
            raise ValueError(f"{model_class.__name__} is not a Pydantic model")

        return model_class.model_json_schema()
    except ImportError:
        raise ImportError(
            "pydantic package not installed. Install with: pip install pydantic"
        )


def validate_pydantic(data: Dict[str, Any], model_class: Type) -> ValidationResult:
    """Validate data against a Pydantic model.

    Args:
        data: Data to validate (typically a dict)
        model_class: Pydantic model class

    Returns:
        ValidationResult: Validation result with any errors
    """
    try:
        from pydantic import BaseModel, ValidationError

        if not issubclass(model_class, BaseModel):
            return ValidationResult(
                valid=False,
                errors=[f"{model_class.__name__} is not a Pydantic model"],
                data=data,
            )

        instance = model_class.model_validate(data)
        return ValidationResult(valid=True, data=instance)
    except ImportError:
        return ValidationResult(
            valid=False,
            errors=[
                "pydantic package not installed. Install with: pip install pydantic"
            ],
            data=data,
        )
    except Exception as e:
        errors = []
        if hasattr(e, "errors"):
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        else:
            errors = [str(e)]

        return ValidationResult(valid=False, errors=errors, data=data)


def pydantic_to_function(
    model_class: Type, name: Optional[str] = None, description: Optional[str] = None
) -> Dict[str, Any]:
    """Convert a Pydantic model to a function calling definition.

    Args:
        model_class: Pydantic model class
        name (str, optional): Function name (defaults to model class name)
        description (str, optional): Function description (defaults to model docstring)

    Returns:
        Dict: Function calling definition
    """
    try:
        from pydantic import BaseModel

        if not issubclass(model_class, BaseModel):
            raise ValueError(f"{model_class.__name__} is not a Pydantic model")

        schema = model_class.model_json_schema()

        func_name = name or model_class.__name__
        func_desc = description or model_class.__doc__ or f"Parameters for {func_name}"

        # Extract properties and required fields
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Import format_function_call to avoid circular dependency
        from .functions import format_function_call

        return format_function_call(func_name, func_desc, properties, required)
    except ImportError:
        raise ImportError(
            "pydantic package not installed. Install with: pip install pydantic"
        )
