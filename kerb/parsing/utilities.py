"""Utility functions for parsing.

This module provides general utility functions for cleaning and
preprocessing LLM outputs.
"""

import re


def clean_llm_output(text: str) -> str:
    """Clean common artifacts from LLM outputs.

    Removes:
    - Markdown code blocks
    - Leading/trailing whitespace
    - Common prefixes like "Here is..." or "Sure, here's..."

    Args:
        text (str): Raw LLM output

    Returns:
        str: Cleaned text
    """
    # Remove markdown code blocks first
    text = re.sub(r"```(?:\w+)?\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Remove common prefixes (more comprehensive patterns)
    prefixes = [
        r"^Sure,?\s+here(?:\'s| is)\s+(?:the|a|an)?\s*",
        r"^Here(?:\'s| is)\s+(?:the|a|an)?\s*",
        r"^(?:OK|Okay),?\s+",
    ]

    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE | re.MULTILINE)

    return text.strip()
