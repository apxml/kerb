"""JSON Schema validation utilities.

This module provides functions for validating data against JSON Schema definitions.
"""

from typing import Any, Dict

from .types import ValidationResult


def validate_json_schema(data: Any, schema: Dict[str, Any]) -> ValidationResult:
    """Validate data against a JSON Schema.
    
    Args:
        data: Data to validate (typically a dict or list)
        schema: JSON Schema definition
        
    Returns:
        ValidationResult: Validation result with any errors
        
    Examples:
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> validate_json_schema({"name": "John"}, schema)
        ValidationResult(valid=True, errors=[], data={'name': 'John'})
    """
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=schema)
        return ValidationResult(valid=True, data=data)
    except ImportError:
        return ValidationResult(
            valid=False,
            errors=["jsonschema package not installed. Install with: pip install jsonschema"],
            data=data
        )
    except jsonschema.exceptions.ValidationError as e:
        return ValidationResult(
            valid=False,
            errors=[str(e)],
            data=data
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            errors=[f"Validation error: {str(e)}"],
            data=data
        )
