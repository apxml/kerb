"""Prompt management utilities for LLM applications.

This module provides comprehensive prompt engineering tools:
- Template engine with variable substitution
- Prompt versioning and A/B testing
- Few-shot example management
- Prompt compression and optimization
"""

# Submodules for specialized imports
from . import examples, optimization, template, versioning
# Few-shot examples
from .examples import (ExampleSelector, FewShotExample, create_example,
                       format_examples, select_examples)
# Optimization utilities
from .optimization import analyze_prompt, compress_prompt, optimize_whitespace
# Core template functionality (most commonly used)
from .template import (extract_template_variables, render_template,
                       render_template_safe, validate_template)
# Versioning classes and functions
from .versioning import (PromptRegistry, PromptVersion, compare_versions,
                         create_version, get_prompt, list_versions,
                         register_prompt, select_version)

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
