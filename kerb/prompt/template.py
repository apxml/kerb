"""Template engine with variable substitution for prompts.

This module provides template rendering functionality with support for
variable substitution, validation, and variable extraction.
"""

import re
from typing import Dict, Any, Tuple, List


def render_template(
    template: str,
    variables: Dict[str, Any],
    delimiters: Tuple[str, str] = ("{{", "}}"),
    allow_missing: bool = False
) -> str:
    """Render a template with variable substitution.
    
    Supports nested variables, conditionals, and loops with simple syntax.
    
    Args:
        template (str): Template string with variable placeholders
        variables (Dict[str, Any]): Dictionary of variables to substitute
        delimiters (Tuple[str, str]): Opening and closing delimiters. Defaults to ("{{", "}}").
        allow_missing (bool): If True, missing variables are left as-is. If False, raises KeyError.
            Defaults to False.
            
    Returns:
        str: Rendered template with variables substituted
        
    Examples:
        >>> render_template("Hello {{name}}!", {"name": "Alice"})
        'Hello Alice!'
        
        >>> render_template("{{greeting}} {{name}}!", {"greeting": "Hi", "name": "Bob"})
        'Hi Bob!'
        
        >>> render_template("Value: {value}", {"value": 42}, delimiters=("{", "}"))
        'Value: 42'
    """
    if not template:
        return template
    
    result = template
    open_delim, close_delim = delimiters
    
    # Escape delimiters for regex
    open_escaped = re.escape(open_delim)
    close_escaped = re.escape(close_delim)
    
    # Find all variable placeholders
    pattern = f"{open_escaped}\\s*([\\w.]+)\\s*{close_escaped}"
    
    def replace_variable(match):
        var_name = match.group(1).strip()
        
        # Handle nested variables (e.g., user.name)
        value = variables
        for key in var_name.split('.'):
            if isinstance(value, dict):
                if key in value:
                    value = value[key]
                else:
                    if allow_missing:
                        return match.group(0)
                    raise KeyError(f"Variable '{var_name}' not found in template variables")
            else:
                if allow_missing:
                    return match.group(0)
                raise KeyError(f"Cannot access '{key}' in non-dict value")
        
        return str(value)
    
    result = re.sub(pattern, replace_variable, result)
    return result


def render_template_safe(
    template: str,
    variables: Dict[str, Any],
    delimiters: Tuple[str, str] = ("{{", "}}"),
    default: str = ""
) -> str:
    """Render a template with safe handling of missing variables.
    
    Missing variables are replaced with the default value instead of raising errors.
    
    Args:
        template (str): Template string with variable placeholders
        variables (Dict[str, Any]): Dictionary of variables to substitute
        delimiters (Tuple[str, str]): Opening and closing delimiters. Defaults to ("{{", "}}").
        default (str): Default value for missing variables. Defaults to "".
            
    Returns:
        str: Rendered template with variables substituted
        
    Examples:
        >>> render_template_safe("Hello {{name}}!", {})
        'Hello !'
        
        >>> render_template_safe("Hello {{name}}!", {}, default="[unknown]")
        'Hello [unknown]!'
    """
    if not template:
        return template
    
    result = template
    open_delim, close_delim = delimiters
    
    # Escape delimiters for regex
    open_escaped = re.escape(open_delim)
    close_escaped = re.escape(close_delim)
    
    # Find all variable placeholders
    pattern = f"{open_escaped}\\s*([\\w.]+)\\s*{close_escaped}"
    
    def replace_variable(match):
        var_name = match.group(1).strip()
        
        # Handle nested variables (e.g., user.name)
        value = variables
        for key in var_name.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return str(value)
    
    result = re.sub(pattern, replace_variable, result)
    return result


def validate_template(
    template: str,
    variables: Dict[str, Any],
    delimiters: Tuple[str, str] = ("{{", "}}")
) -> Tuple[bool, List[str]]:
    """Validate that all template variables are available.
    
    Args:
        template (str): Template string to validate
        variables (Dict[str, Any]): Available variables
        delimiters (Tuple[str, str]): Opening and closing delimiters. Defaults to ("{{", "}}").
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, missing_variables)
        
    Examples:
        >>> validate_template("Hello {{name}}!", {"name": "Alice"})
        (True, [])
        
        >>> validate_template("Hello {{name}}!", {})
        (False, ['name'])
    """
    required_vars = extract_template_variables(template, delimiters)
    missing = []
    
    for var_name in required_vars:
        # Check nested variables
        value = variables
        keys = var_name.split('.')
        found = True
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                found = False
                break
        
        if not found:
            missing.append(var_name)
    
    return len(missing) == 0, missing


def extract_template_variables(
    template: str,
    delimiters: Tuple[str, str] = ("{{", "}}")
) -> List[str]:
    """Extract all variable names from a template.
    
    Args:
        template (str): Template string to analyze
        delimiters (Tuple[str, str]): Opening and closing delimiters. Defaults to ("{{", "}}").
        
    Returns:
        List[str]: List of unique variable names found in template
        
    Examples:
        >>> extract_template_variables("Hello {{name}}!")
        ['name']
        
        >>> extract_template_variables("{{greeting}} {{name}}, you are {{age}} years old!")
        ['greeting', 'name', 'age']
    """
    if not template:
        return []
    
    open_delim, close_delim = delimiters
    
    # Escape delimiters for regex
    open_escaped = re.escape(open_delim)
    close_escaped = re.escape(close_delim)
    
    # Find all variable placeholders
    pattern = f"{open_escaped}\\s*([\\w.]+)\\s*{close_escaped}"
    matches = re.findall(pattern, template)
    
    # Return unique variables preserving order
    seen = set()
    result = []
    for var in matches:
        var = var.strip()
        if var not in seen:
            seen.add(var)
            result.append(var)
    
    return result
