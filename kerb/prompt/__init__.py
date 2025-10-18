"""Prompt management utilities for LLM applications.

This module provides comprehensive prompt engineering tools:
- Template engine with variable substitution
- Prompt versioning and A/B testing
- Few-shot example management
- Prompt compression and optimization
"""

# Core template functionality (most commonly used)
from .template import (
    render_template,
    render_template_safe,
    validate_template,
    extract_template_variables,
)

# Versioning classes and functions
from .versioning import (
    PromptVersion,
    PromptRegistry,
    create_version,
    register_prompt,
    get_prompt,
    list_versions,
    compare_versions,
    select_version,
)

# Few-shot examples
from .examples import (
    FewShotExample,
    ExampleSelector,
    create_example,
    select_examples,
    format_examples,
)

# Optimization utilities
from .optimization import (
    compress_prompt,
    optimize_whitespace,
    analyze_prompt,
)

# Submodules for specialized imports
from . import template
from . import versioning
from . import examples
from . import optimization


__all__ = [
    # Core template functions
    "render_template",
    "render_template_safe",
    "validate_template",
    "extract_template_variables",
    
    # Versioning
    "PromptVersion",
    "PromptRegistry",
    "create_version",
    "register_prompt",
    "get_prompt",
    "list_versions",
    "compare_versions",
    "select_version",
    
    # Few-shot examples
    "FewShotExample",
    "ExampleSelector",
    "create_example",
    "select_examples",
    "format_examples",
    
    # Optimization
    "compress_prompt",
    "optimize_whitespace",
    "analyze_prompt",
    
    # Submodules
    "template",
    "versioning",
    "examples",
    "optimization",
]

