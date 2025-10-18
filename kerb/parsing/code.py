"""Code extraction utilities.

This module provides functions for extracting code blocks from markdown and
other text formats.
"""

import re
from typing import Dict, List, Optional


def extract_code_blocks(
    text: str, language: Optional[str] = None
) -> List[Dict[str, str]]:
    """Extract code blocks from markdown text.

    Args:
        text (str): Markdown text containing code blocks
        language (str, optional): Filter by language (e.g., 'python', 'json')

    Returns:
        List[Dict]: List of code blocks with 'language' and 'code' keys

    Examples:
        >>> extract_code_blocks('```python\\nprint("hello")\\n```')
        [{'language': 'python', 'code': 'print("hello")'}]
    """
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    blocks = []
    for lang, code in matches:
        if language is None or lang.lower() == language.lower():
            blocks.append({"language": lang or "unknown", "code": code.strip()})

    return blocks
