"""
Structured Output Example
=========================

This example demonstrates generating structured data and JSON responses.

Main concepts:
- Using JSON mode for structured output
- Parsing JSON responses reliably
- Generating data models and schemas
- Extracting structured information from text
- Working with response_format parameter
"""

import json
from typing import Dict, Any
from kerb.generation import generate, ModelName, GenerationConfig
from kerb.parsing import extract_json


def example_json_mode_basic():
    """Generate JSON output using JSON mode."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic JSON Mode")
    print("="*80)
    
    prompt = """Generate a JSON object representing a Python package with the following fields:

# %%
# Setup and Imports
# -----------------
    - name: package name
    - version: semantic version
    - description: short description
    - dependencies: list of dependency names"""
    
    config = GenerationConfig(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.7
    )
    
    print("\nPrompt: Generate a Python package info JSON\n")
    
    response = generate(prompt, config=config)
    
    print("Raw Response:")
    print(response.content)
    
    # Parse JSON
    try:
        data = json.loads(response.content)
        print("\nParsed JSON:")
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")


def example_structured_data_extraction():
    """Extract structured data from unstructured text."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Structured Data Extraction")
    print("="*80)
    
    text = """
    Python 3.11 was released on October 24, 2022. It includes several performance 
    improvements and new features like exception groups and the tomllib module.
    """
    
    prompt = f"""Extract the following information from the text and return as JSON:
    - language: programming language name
    - version: version number
    - release_date: release date
    - features: list of mentioned features
    
    Text: {text}
    
    Return only valid JSON."""
    
    config = GenerationConfig(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.3
    )
    
    response = generate(prompt, config=config)
    
    print("Source Text:")
    print(text.strip())
    print("\nExtracted Data:")
    
    data = extract_json(response.content)
    if data:
        print(json.dumps(data, indent=2))
    else:
        print("Failed to extract JSON")



# %%
# Example Schema Generation
# -------------------------

def example_schema_generation():
    """Generate data that follows a specific schema."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Schema-Based Generation")
    print("="*80)
    
    schema = {
        "type": "object",
        "properties": {
            "function_name": {"type": "string"},
            "parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"}
                    }
                }
            },
            "return_type": {"type": "string"},
            "description": {"type": "string"}
        }
    }
    
    prompt = f"""Generate a JSON object for a Python function that calculates the factorial of a number.
    Follow this schema exactly:
    {json.dumps(schema, indent=2)}
    
    Return only valid JSON."""
    
    config = GenerationConfig(
        model="gpt-4o-mini",
        response_format={"type": "json_object"}
    )
    
    response = generate(prompt, config=config)
    
    print("Generated Function Specification:")
    data = extract_json(response.content)
    if data:
        print(json.dumps(data, indent=2))


def example_multiple_entities():
    """Extract multiple structured entities from text."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Multiple Entity Extraction")
    print("="*80)
    
    text = """
    FastAPI is a modern web framework for Python. It was created by Sebastian Ramirez.
    Django is a high-level Python web framework created by Adrian Holovaty and Simon Willison.
    Flask is a micro web framework written in Python, created by Armin Ronacher.
    """
    
    prompt = f"""Extract information about each framework mentioned in the text.
    Return a JSON object with a 'frameworks' key containing an array of objects.
    Each object should have: name, type, language, and creators (array).
    
    Text: {text}
    
    Return only valid JSON."""
    
    config = GenerationConfig(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.2
    )
    
    response = generate(prompt, config=config)
    
    print("Extracted Frameworks:")
    data = extract_json(response.content)
    if data and "frameworks" in data:
        for framework in data["frameworks"]:
            print(f"\n  {framework.get('name', 'Unknown')}:")
            print(f"    Type: {framework.get('type', 'N/A')}")
            print(f"    Language: {framework.get('language', 'N/A')}")
            print(f"    Creators: {', '.join(framework.get('creators', []))}")



# %%
# Example Validation And Correction
# ---------------------------------

def example_validation_and_correction():
    """Generate and validate structured output."""
    print("\n" + "="*80)
    print("EXAMPLE 5: JSON Validation and Correction")
    print("="*80)
    
    prompt = """Generate a JSON object representing a user profile with:
    - username (string, lowercase, no spaces)
    - email (valid email format)
    - age (integer, 18-100)
    - interests (array of strings)
    - created_at (ISO 8601 timestamp)
    
    Make up realistic data. Return only valid JSON."""
    
    config = GenerationConfig(
        model="gpt-4o-mini",
        response_format={"type": "json_object"}
    )
    
    response = generate(prompt, config=config)
    
    print("Generated User Profile:")
    
    try:
        data = json.loads(response.content)
        
        # Validate
        is_valid = True
        issues = []
        
        if not isinstance(data.get("username"), str) or " " in data.get("username", ""):
            issues.append("Username contains spaces")
            is_valid = False
        
        if not isinstance(data.get("age"), int) or not (18 <= data.get("age", 0) <= 100):
            issues.append("Age out of range")
            is_valid = False
        
        if not isinstance(data.get("interests"), list):
            issues.append("Interests not a list")
            is_valid = False
        
        print(json.dumps(data, indent=2))
        
        print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
    
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")


def example_code_to_json():
    """Convert code into structured JSON representation."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Code to JSON Conversion")
    print("="*80)
    
    code = """
    def calculate_bmi(weight: float, height: float) -> float:
        '''Calculate Body Mass Index.
        
        Args:
            weight: Weight in kilograms
            height: Height in meters
        
        Returns:
            BMI value
        '''
        return weight / (height ** 2)
    """
    
    prompt = f"""Analyze this Python function and convert it to JSON with:
    - name: function name
    - docstring: the docstring text
    - parameters: array of {{name, type, description}}
    - return_type: return type annotation
    - body: brief description of what it does
    
    Function:
    {code}
    
    Return only valid JSON."""
    
    config = GenerationConfig(
        model="gpt-4o-mini",
        response_format={"type": "json_object"}
    )
    
    response = generate(prompt, config=config)
    
    print("Original Code:")
    print(code)
    print("\nJSON Representation:")
    
    data = extract_json(response.content)
    if data:
        print(json.dumps(data, indent=2))



# %%
# Main
# ----

def main():
    """Run all structured output examples."""
    print("\n" + "#"*80)
    print("# STRUCTURED OUTPUT EXAMPLES")
    print("#"*80)
    
    try:
        example_json_mode_basic()
    except Exception as e:
        print(f"\nExample 1 Error: {e}")
    
    try:
        example_structured_data_extraction()
    except Exception as e:
        print(f"\nExample 2 Error: {e}")
    
    try:
        example_schema_generation()
    except Exception as e:
        print(f"\nExample 3 Error: {e}")
    
    try:
        example_multiple_entities()
    except Exception as e:
        print(f"\nExample 4 Error: {e}")
    
    try:
        example_validation_and_correction()
    except Exception as e:
        print(f"\nExample 5 Error: {e}")
    
    try:
        example_code_to_json()
    except Exception as e:
        print(f"\nExample 6 Error: {e}")
    
    print("\n" + "#"*80)
    print("# Examples completed")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
