"""Metadata extraction utilities.

This module provides functions for extracting metadata and structured information:
- File metadata extraction
- Document statistics
- URL extraction
- Email extraction
- Date extraction
- Phone number extraction
"""

import re
from pathlib import Path
from typing import Dict, List, Any


def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from a file.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        Dict[str, Any]: Extracted metadata
        
    Examples:
        >>> metadata = extract_metadata("document.pdf")
        >>> print(metadata['size'], metadata['created'])
    """
    path = Path(file_path)
    
    metadata = {
        "filename": path.name,
        "extension": path.suffix.lstrip('.'),
        "size": path.stat().st_size,
        "created": path.stat().st_ctime,
        "modified": path.stat().st_mtime,
    }
    
    return metadata


def extract_document_stats(text: str) -> Dict[str, int]:
    """Extract statistics from document text.
    
    Args:
        text (str): Document text
        
    Returns:
        Dict[str, int]: Document statistics
        
    Examples:
        >>> stats = extract_document_stats("Hello world. This is a test.")
        >>> print(stats['word_count'], stats['sentence_count'])
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    paragraphs = text.split('\n\n')
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "paragraph_count": len([p for p in paragraphs if p.strip()]),
        "line_count": text.count('\n') + 1,
    }


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text.
    
    Args:
        text (str): Text to extract URLs from
        
    Returns:
        List[str]: List of URLs
        
    Examples:
        >>> extract_urls("Visit https://example.com and www.test.com")
        ['https://example.com', 'www.test.com']
    """
    url_pattern = r'https?://\S+|www\.\S+'
    return re.findall(url_pattern, text)


def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text.
    
    Args:
        text (str): Text to extract emails from
        
    Returns:
        List[str]: List of email addresses
        
    Examples:
        >>> extract_emails("Contact us at info@example.com or sales@test.org")
        ['info@example.com', 'sales@test.org']
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


def extract_dates(text: str) -> List[str]:
    """Extract dates from text (simple patterns).
    
    Args:
        text (str): Text to extract dates from
        
    Returns:
        List[str]: List of potential date strings
        
    Examples:
        >>> extract_dates("Meeting on 2024-01-15 and 01/20/2024")
        ['2024-01-15', '01/20/2024']
    """
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
    ]
    
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    
    return dates


def extract_phone_numbers(text: str) -> List[str]:
    """Extract phone numbers from text (US format).
    
    Args:
        text (str): Text to extract phone numbers from
        
    Returns:
        List[str]: List of phone numbers
        
    Examples:
        >>> extract_phone_numbers("Call (555) 123-4567 or 555-987-6543")
        ['(555) 123-4567', '555-987-6543']
    """
    phone_patterns = [
        r'\(\d{3}\)\s*\d{3}-\d{4}',  # (555) 123-4567
        r'\d{3}-\d{3}-\d{4}',  # 555-123-4567
        r'\d{3}\.\d{3}\.\d{4}',  # 555.123.4567
    ]
    
    phones = []
    for pattern in phone_patterns:
        phones.extend(re.findall(pattern, text))
    
    return phones
