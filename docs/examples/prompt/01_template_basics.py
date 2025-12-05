"""
Basic prompt template usage for LLM developers.
===============================================

This example demonstrates:
- Variable substitution in prompt templates
- Template validation and error handling
- Custom delimiters for different LLM formats
- Nested variable support for structured data
"""

from kerb.prompt import (
    render_template,
    render_template_safe,
    validate_template,
    extract_template_variables
)


def basic_template_rendering():
    """Demonstrate basic template rendering for LLM prompts."""
    print("=" * 80)
    print("BASIC TEMPLATE RENDERING")
    print("=" * 80)
    
    # Simple system prompt with variables
    template = """You are a {{role}} assistant specialized in {{domain}}.

# %%
# Setup and Imports
# -----------------
Your task is to help users with {{task}}."""
    
    variables = {
        "role": "helpful",
        "domain": "Python programming",
        "task": "code review and debugging"
    }
    
    result = render_template(template, variables)
    print("\nTemplate:")
    print(template)
    print("\nRendered:")
    print(result)


def user_query_templates():
    """Use templates for user queries with dynamic data."""
    print("\n" + "=" * 80)
    print("USER QUERY TEMPLATES")
    print("=" * 80)
    
    # Template for code analysis requests
    template = """Analyze the following {{language}} code:

```{{language}}
{{code}}
```

Focus on:
- {{focus_area}}
- Performance implications
- Best practices"""
    
    variables = {
        "language": "python",
        "code": "def process(data):\\n    return [x*2 for x in data]",
        "focus_area": "Time complexity"
    }
    
    result = render_template(template, variables)
    print("\nCode Analysis Prompt:")
    print(result)



# %%
# Nested Variable Templates
# -------------------------

def nested_variable_templates():
    """Use nested variables for structured prompt data."""
    print("\n" + "=" * 80)
    print("NESTED VARIABLE TEMPLATES")
    print("=" * 80)
    
    # Template with nested user data
    template = """Generate a personalized response for:
Name: {{user.name}}
Role: {{user.role}}
Experience: {{user.experience}} years
Interests: {{user.interests}}

Context: {{context}}"""
    
    variables = {
        "user": {
            "name": "Alice Johnson",
            "role": "Senior Developer",
            "experience": 5,
            "interests": "Machine Learning, Cloud Architecture"
        },
        "context": "Recommending learning resources"
    }
    
    result = render_template(template, variables)
    print("\nPersonalized Prompt:")
    print(result)


def custom_delimiters():
    """Use custom delimiters for different LLM prompt formats."""
    print("\n" + "=" * 80)
    print("CUSTOM DELIMITERS")
    print("=" * 80)
    
    # Some LLM frameworks use different placeholder syntax
    # Example 1: Using {var} instead of {{var}}
    template = "Translate '{text}' from {source_lang} to {target_lang}"
    variables = {
        "text": "Hello, world!",
        "source_lang": "English",
        "target_lang": "Spanish"
    }
    
    result = render_template(template, variables, delimiters=("{", "}"))
    print("\nSingle brace delimiters:")
    print(result)
    
    # Example 2: Using %var% syntax
    template2 = "System: %system_prompt%\nUser: %user_query%"
    variables2 = {
        "system_prompt": "You are a helpful assistant",
        "user_query": "What is Python?"
    }
    
    result2 = render_template(template2, variables2, delimiters=("%", "%"))
    print("\nPercent delimiters:")
    print(result2)



# %%
# Safe Rendering With Fallbacks
# -----------------------------

def safe_rendering_with_fallbacks():
    """Handle missing variables gracefully in production."""
    print("\n" + "=" * 80)
    print("SAFE RENDERING WITH FALLBACKS")
    print("=" * 80)
    
    # Production scenario: not all variables might be available
    template = """System: {{system_role}}
User: {{user_name}}
Context: {{context}}
History: {{chat_history}}"""
    
    # Partial variables - some might be missing
    variables = {
        "system_role": "AI Assistant",
        "user_name": "Bob"
        # context and chat_history are missing
    }
    
    # Use safe rendering with default values
    result = render_template_safe(template, variables, default="[Not provided]")
    print("\nTemplate with missing variables:")
    print(result)


def template_validation():
    """Validate templates before rendering."""
    print("\n" + "=" * 80)
    print("TEMPLATE VALIDATION")
    print("=" * 80)
    
    template = """You are analyzing {{model_type}} model performance.
Dataset: {{dataset_name}}
Metrics: {{metrics}}
Baseline: {{baseline_score}}"""
    
    # Check if all variables are available
    available_vars = {
        "model_type": "transformer",
        "dataset_name": "IMDB",
        "metrics": "accuracy, F1-score"
        # baseline_score is missing
    }
    
    is_valid, missing = validate_template(template, available_vars)
    
    print(f"\nTemplate is valid: {is_valid}")
    if not is_valid:
        print(f"Missing variables: {missing}")
        print("\nYou should provide values for these variables before rendering.")



# %%
# Extract Required Variables
# --------------------------

def extract_required_variables():
    """Extract and inspect required variables from templates."""
    print("\n" + "=" * 80)
    print("EXTRACT REQUIRED VARIABLES")
    print("=" * 80)
    
    # Useful for template documentation and validation
    template = """Generate {{output_format}} documentation for:
Function: {{function_name}}
Parameters: {{parameters}}
Returns: {{return_type}}
Description: {{description}}"""
    
    required_vars = extract_template_variables(template)
    
    print("\nTemplate:")
    print(template)
    print(f"\nRequired variables: {required_vars}")
    print("\nThis is useful for:")
    print("- Documenting template requirements")
    print("- Building UI forms for prompt configuration")
    print("- Validating configuration files")


def main():
    """Run all template examples."""
    print("\n" + "=" * 80)
    print("PROMPT TEMPLATE EXAMPLES FOR LLM DEVELOPERS")
    print("=" * 80)
    
    basic_template_rendering()
    user_query_templates()
    nested_variable_templates()
    custom_delimiters()
    safe_rendering_with_fallbacks()
    template_validation()
    extract_required_variables()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
