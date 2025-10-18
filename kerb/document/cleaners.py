"""Text cleaning utilities.

This module provides functions for cleaning and normalizing text:
- General text cleaning
- Newline normalization
"""

import re


def clean_text(
    text: str,
    normalize_whitespace: bool = True,
    remove_urls: bool = False,
    remove_emails: bool = False,
    remove_special_chars: bool = False,
    lowercase: bool = False,
) -> str:
    """Clean and normalize text.

    Args:
        text (str): Text to clean
        normalize_whitespace (bool): Normalize whitespace to single spaces
        remove_urls (bool): Remove URLs
        remove_emails (bool): Remove email addresses
        remove_special_chars (bool): Remove special characters
        lowercase (bool): Convert to lowercase

    Returns:
        str: Cleaned text

    Examples:
        >>> text = "Check   out https://example.com  for more info!"
        >>> clean_text(text, normalize_whitespace=True, remove_urls=True)
        'Check out for more info!'
    """
    cleaned = text

    if remove_urls:
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        cleaned = re.sub(r"www\.\S+", "", cleaned)

    if remove_emails:
        cleaned = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", cleaned
        )

    if remove_special_chars:
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-\'"()]', "", cleaned)

    if normalize_whitespace:
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()

    if lowercase:
        cleaned = cleaned.lower()

    return cleaned


def remove_extra_newlines(text: str, max_consecutive: int = 2) -> str:
    """Remove excessive newlines from text.

    Args:
        text (str): Text to process
        max_consecutive (int): Maximum consecutive newlines to keep

    Returns:
        str: Text with limited newlines

    Examples:
        >>> remove_extra_newlines("Hello\\n\\n\\n\\nWorld", max_consecutive=2)
        'Hello\\n\\nWorld'
    """
    pattern = r"\n{" + str(max_consecutive + 1) + r",}"
    replacement = "\n" * max_consecutive
    return re.sub(pattern, replacement, text)
