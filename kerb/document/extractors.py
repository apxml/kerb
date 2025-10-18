"""Text extraction utilities.

This module provides functions for extracting text from various formats:
- HTML text extraction
- Markdown stripping
- Sentence splitting
- Paragraph splitting
"""

import re
from typing import List


def extract_text_from_html(html: str, remove_scripts: bool = True) -> str:
    """Extract plain text from HTML content.

    Args:
        html (str): HTML content
        remove_scripts (bool): Remove script and style tags

    Returns:
        str: Extracted plain text

    Examples:
        >>> html = '<html><body><p>Hello World</p></body></html>'
        >>> extract_text_from_html(html)
        'Hello World'
    """
    text = html

    # Remove script and style tags
    if remove_scripts:
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
        )

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def strip_markdown(text: str) -> str:
    """Remove Markdown formatting from text.

    Args:
        text (str): Markdown text

    Returns:
        str: Plain text without Markdown formatting

    Examples:
        >>> strip_markdown("# Hello **World**")
        'Hello World'
    """
    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bold and italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)  # Bold italic
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # Bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)  # Italic
    text = re.sub(r"__(.+?)__", r"\1", text)  # Bold
    text = re.sub(r"_(.+?)_", r"\1", text)  # Italic

    # Remove inline code
    text = re.sub(r"`(.+?)`", r"\1", text)

    # Remove code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove links
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", text)

    # Remove blockquotes
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)

    # Remove list markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Args:
        text (str): Text to split

    Returns:
        List[str]: List of sentences

    Examples:
        >>> split_into_sentences("Hello world. This is a test!")
        ['Hello world.', 'This is a test!']
    """
    # Simple sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs.

    Args:
        text (str): Text to split

    Returns:
        List[str]: List of paragraphs

    Examples:
        >>> split_into_paragraphs("Para 1\\n\\nPara 2\\n\\nPara 3")
        ['Para 1', 'Para 2', 'Para 3']
    """
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]
