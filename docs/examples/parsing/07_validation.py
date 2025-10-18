"""Output Validation and Schema Validation

This example demonstrates comprehensive validation of LLM outputs against
JSON schemas, custom validators, and output type validation.

Main concepts:
- Validating JSON against schemas
- Custom validation functions
- Output type validation
- Comprehensive validation workflows
"""

from kerb.parsing import (
    validate_output,
    validate_json_schema,
    parse_json,
    ParseMode
)


def simulate_llm_validation_responses() -> dict:
    """Generate LLM responses for validation examples."""
    return {
        "user_data": """{
    "username": "alice123",
    "email": "alice@example.com",
    "age": 28,
    "roles": ["user", "moderator"]
}""",
        
        "invalid_email": """{
    "username": "bob456",
    "email": "invalid-email",
    "age": 35,
    "roles": ["user"]
}""",
        
        "api_response": """{
    "status": "success",
    "data": {
        "items": [
            {"id": 1, "name": "Item 1", "price": 19.99},
            {"id": 2, "name": "Item 2", "price": 29.99}
        ],
        "total": 2
    },
    "timestamp": "2024-01-15T10:30:00Z"
}""",
        
        "config_data": """{
    "api_key": "sk-test123",
    "endpoint": "https://api.example.com",
    "timeout": 30,
    "retries": 3,
    "features": {
        "cache": true,
        "logging": true
    }
}""",
    }


def create_user_schema() -> dict:
    """Create JSON Schema for user data validation."""
    return {
        "type": "object",
        "properties": {
            "username": {
                "type": "string",
                "minLength": 3,
                "maxLength": 50,
                "pattern": "^[a-z0-9_]+$"
            },
            "email": {
                "type": "string",
                "format": "email"
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150
            },
            "roles": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1
            }
        },
        "required": ["username", "email", "age", "roles"],
        "additionalProperties": False
    }


def create_api_response_schema() -> dict:
    """Create JSON Schema for API response validation."""
    return {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "error", "pending"]
            },
            "data": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                                "price": {"type": "number"}
                            },
                            "required": ["id", "name", "price"]
                        }
                    },
                    "total": {"type": "integer"}
                },
                "required": ["items", "total"]
            },
            "timestamp": {
                "type": "string",
                "format": "date-time"
            }
        },
        "required": ["status", "data"]
    }


def main():
    """Run output validation examples."""
    
    print("="*80)
    print("OUTPUT VALIDATION AND SCHEMA VALIDATION")
    print("="*80)
    
    responses = simulate_llm_validation_responses()
    
    # Example 1: Basic JSON Schema validation
    print("\nExample 1: Basic JSON Schema Validation")
    print("-"*80)
    
    text = responses["user_data"]
    print(f"User Data:\n{text}\n")
    
    # Parse JSON first
    parse_result = parse_json(text)
    if parse_result.success:
        data = parse_result.data
        
        # Validate against schema
        schema = create_user_schema()
        validation_result = validate_json_schema(data, schema)
        
        print(f"Valid: {validation_result.valid}")
        if validation_result.valid:
            print("Data passes all schema validations")
        else:
            print(f"Errors: {validation_result.errors}")
    
    # Example 2: Schema validation failure
    print("\n\nExample 2: Schema Validation Failure")
    print("-"*80)
    
    text = responses["invalid_email"]
    print(f"Invalid User Data:\n{text}\n")
    
    parse_result = parse_json(text)
    if parse_result.success:
        data = parse_result.data
        
        schema = create_user_schema()
        validation_result = validate_json_schema(data, schema)
        
        print(f"Valid: {validation_result.valid}")
        if not validation_result.valid:
            print("Validation Errors:")
            for error in validation_result.errors:
                print(f"  - {error}")
    
    # Example 3: Complex nested schema validation
    print("\n\nExample 3: Complex Nested Schema Validation")
    print("-"*80)
    
    text = responses["api_response"]
    print(f"API Response:\n{text[:150]}...\n")
    
    parse_result = parse_json(text)
    if parse_result.success:
        data = parse_result.data
        
        schema = create_api_response_schema()
        validation_result = validate_json_schema(data, schema)
        
        print(f"Valid: {validation_result.valid}")
        if validation_result.valid:
            print("API response structure is valid")
            print(f"Status: {data['status']}")
            print(f"Items count: {data['data']['total']}")
        else:
            print(f"Errors: {validation_result.errors}")
    
    # Example 4: Output type validation
    print("\n\nExample 4: Output Type Validation")
    print("-"*80)
    
    text = responses["user_data"]
    
    # Validate as JSON object
    result = validate_output(text, output_type="json_object")
    
    print(f"Validation as JSON object:")
    print(f"  Valid: {result.valid}")
    if result.valid:
        print(f"  Data type: {type(result.data).__name__}")
        print(f"  Keys: {list(result.data.keys())}")
    
    # Example 5: Custom validation function
    print("\n\nExample 5: Custom Validation Function")
    print("-"*80)
    
    def validate_api_key(data: dict) -> bool:
        """Custom validator for API configuration."""
        if "api_key" not in data:
            return False
        
        api_key = data["api_key"]
        
        # Check API key format
        if not api_key.startswith("sk-"):
            return False
        
        if len(api_key) < 10:
            return False
        
        return True
    
    text = responses["config_data"]
    print(f"Config Data:\n{text}\n")
    
    parse_result = parse_json(text)
    if parse_result.success:
        data = parse_result.data
        
        result = validate_output(
            text,
            output_type="json_object",
            custom_validator=validate_api_key
        )
        
        print(f"Custom Validation Result:")
        print(f"  Valid: {result.valid}")
        if result.valid:
            print("  API key format is valid")
        else:
            print(f"  Errors: {result.errors}")
    
    # Example 6: Multi-stage validation
    print("\n\nExample 6: Multi-Stage Validation Workflow")
    print("-"*80)
    
    text = responses["user_data"]
    
    # Stage 1: Parse
    print("Stage 1: Parse JSON")
    parse_result = parse_json(text, mode=ParseMode.LENIENT)
    print(f"  Parse Success: {parse_result.success}")
    
    if parse_result.success:
        data = parse_result.data
        
        # Stage 2: Schema validation
        print("\nStage 2: Schema Validation")
        schema = create_user_schema()
        schema_result = validate_json_schema(data, schema)
        print(f"  Schema Valid: {schema_result.valid}")
        
        if schema_result.valid:
            # Stage 3: Business logic validation
            print("\nStage 3: Business Logic Validation")
            
            business_rules_pass = True
            
            # Check minimum age
            if data["age"] < 18:
                print("  Warning: User is under 18")
                business_rules_pass = False
            
            # Check role validity
            valid_roles = ["user", "moderator", "admin"]
            invalid_roles = [r for r in data["roles"] if r not in valid_roles]
            if invalid_roles:
                print(f"  Error: Invalid roles: {invalid_roles}")
                business_rules_pass = False
            
            if business_rules_pass:
                print("  All business rules passed")
    
    # Example 7: Batch validation
    print("\n\nExample 7: Batch Validation")
    print("-"*80)
    
    test_cases = [
        ("Valid User", responses["user_data"]),
        ("Invalid Email", responses["invalid_email"]),
    ]
    
    schema = create_user_schema()
    
    results = []
    for name, text in test_cases:
        parse_result = parse_json(text)
        if parse_result.success:
            validation_result = validate_json_schema(parse_result.data, schema)
            results.append((name, validation_result.valid))
        else:
            results.append((name, False))
    
    print("Batch Validation Results:")
    for name, valid in results:
        status = "PASS" if valid else "FAIL"
        print(f"  [{status}] {name}")
    
    # Example 8: Schema generation for documentation
    print("\n\nExample 8: Schema as Documentation")
    print("-"*80)
    
    schema = create_user_schema()
    
    print("User Data Requirements:")
    print(f"  Required fields: {schema.get('required', [])}")
    print("\n  Field Specifications:")
    for field, spec in schema["properties"].items():
        print(f"    - {field}:")
        print(f"        Type: {spec.get('type', 'any')}")
        if "minLength" in spec:
            print(f"        Min Length: {spec['minLength']}")
        if "maximum" in spec:
            print(f"        Maximum: {spec['maximum']}")
        if "pattern" in spec:
            print(f"        Pattern: {spec['pattern']}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
