"""Format-specific text preprocessing utilities.

This module provides preprocessing functions for specific document formats:
- PDF text preprocessing
- HTML text preprocessing
- Markdown preprocessing
"""

import re


def preprocess_pdf_text(text: str) -> str:
    """Preprocess text extracted from PDF.

    PDFs often have formatting artifacts like broken lines, extra spaces, etc.

    Args:
        text (str): Text extracted from PDF

    Returns:
        str: Cleaned text

    Examples:
        >>> pdf_text = "This is a sen-\\ntence with line break."
        >>> preprocess_pdf_text(pdf_text)
        'This is a sentence with line break.'
    """
    # Fix hyphenated line breaks
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Fix line breaks mid-sentence
    text = re.sub(r"(\w)\s*\n\s*(\w)", r"\1 \2", text)

    # Normalize whitespace (inline instead of removed function)
    text = " ".join(text.split())

    return text


def preprocess_html_text(html: str) -> str:
    """Preprocess HTML to extract clean text.

    Args:
        html (str): HTML content

    Returns:
        str: Cleaned text

    Examples:
        >>> html = '<div>Hello <span>World</span></div>'
        >>> preprocess_html_text(html)
        'Hello World'
    """
    from .extractors import extract_text_from_html

    text = extract_text_from_html(html)
    text = " ".join(text.split())
    return text


def preprocess_markdown(text: str, keep_structure: bool = True) -> str:
    """Preprocess Markdown text.

    Args:
        text (str): Markdown text
        keep_structure (bool): Keep headings and structure markers

    Returns:
        str: Processed text

    Examples:
        >>> md = "# Title\\n\\nSome **bold** text"
        >>> preprocess_markdown(md, keep_structure=False)
        'Title\\n\\nSome bold text'
    """
    from .extractors import strip_markdown

    if not keep_structure:
        text = strip_markdown(text)

    text = " ".join(text.split())
    return text
