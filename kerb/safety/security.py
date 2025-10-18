"""Security and privacy utility functions.

This module provides functions for input sanitization, character escaping,
URL validation, file upload checking, and data exfiltration detection.
"""

import re
from typing import Optional, List

from .enums import ContentCategory
from .types import SafetyResult
from .pii import detect_url


def sanitize_input(
    text: str,
    remove_html: bool = True,
    remove_scripts: bool = True,
    max_length: Optional[int] = None
) -> str:
    """Clean and sanitize user input.
    
    Args:
        text: User input to sanitize
        remove_html: Whether to remove HTML tags
        remove_scripts: Whether to remove script tags
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
        
    Examples:
        >>> input_text = "<script>alert('xss')</script>Hello"
        >>> sanitized = sanitize_input(input_text)
        >>> print(sanitized)
        "Hello"
    """
    sanitized = text
    
    # Remove script tags
    if remove_scripts:
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML tags
    if remove_html:
        sanitized = re.sub(r'<[^>]+>', '', sanitized)
    
    # Trim to max length
    if max_length:
        sanitized = sanitized[:max_length]
    
    # Remove leading/trailing whitespace
    sanitized = sanitized.strip()
    
    return sanitized


def escape_special_chars(
    text: str,
    escape_html: bool = True,
    escape_sql: bool = True
) -> str:
    """Escape potentially dangerous characters.
    
    Args:
        text: Text to escape
        escape_html: Whether to escape HTML special chars
        escape_sql: Whether to escape SQL special chars
        
    Returns:
        Escaped text
    """
    escaped = text
    
    if escape_html:
        html_escapes = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
        }
        for char, escape in html_escapes.items():
            escaped = escaped.replace(char, escape)
    
    if escape_sql:
        # Basic SQL escaping (for demonstration)
        escaped = escaped.replace("'", "''")
        escaped = escaped.replace(";", "")
        escaped = escaped.replace("--", "")
    
    return escaped


def validate_url_safety(
    url: str,
    allow_http: bool = True,
    blocked_domains: Optional[List[str]] = None
) -> SafetyResult:
    """Check if URL is safe.
    
    Args:
        url: URL to validate
        allow_http: Whether to allow HTTP (vs HTTPS only)
        blocked_domains: List of blocked domains
        
    Returns:
        SafetyResult with URL safety assessment
    """
    issues = []
    score = 1.0
    
    # Check protocol
    if not allow_http and url.startswith('http://'):
        issues.append("HTTP not allowed (use HTTPS)")
        score -= 0.3
    
    if not (url.startswith('http://') or url.startswith('https://')):
        issues.append("Invalid protocol (must be HTTP/HTTPS)")
        score -= 0.5
    
    # Check for blocked domains
    if blocked_domains:
        domain_match = re.search(r'://([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            for blocked in blocked_domains:
                if blocked in domain:
                    issues.append(f"Blocked domain: {blocked}")
                    score -= 0.8
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'javascript:',
        r'data:',
        r'file://',
        r'\.\./',  # Path traversal
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            issues.append(f"Suspicious pattern detected: {pattern}")
            score -= 0.5
    
    score = max(0.0, score)
    safe = len(issues) == 0 and score >= 0.5
    
    return SafetyResult(
        safe=safe,
        score=score,
        reason="; ".join(issues) if issues else None,
        details={'issues': issues, 'url': url}
    )


def check_file_upload(
    filename: str,
    allowed_extensions: Optional[List[str]] = None,
    blocked_extensions: Optional[List[str]] = None
) -> SafetyResult:
    """Validate uploaded file content.
    
    Args:
        filename: Name of uploaded file
        allowed_extensions: List of allowed extensions
        blocked_extensions: List of blocked extensions
        
    Returns:
        SafetyResult with file upload assessment
    """
    issues = []
    score = 1.0
    
    # Extract extension
    ext = filename.split('.')[-1].lower() if '.' in filename else ''
    
    # Check allowed extensions
    if allowed_extensions:
        if ext not in [e.lower() for e in allowed_extensions]:
            issues.append(f"Extension not allowed: {ext}")
            score -= 0.8
    
    # Check blocked extensions
    if blocked_extensions:
        if ext in [e.lower() for e in blocked_extensions]:
            issues.append(f"Extension blocked: {ext}")
            score -= 0.8
    
    # Check for dangerous extensions
    dangerous_extensions = ['exe', 'bat', 'sh', 'cmd', 'com', 'pif', 'scr', 'vbs', 'js']
    if ext in dangerous_extensions:
        issues.append(f"Dangerous file type: {ext}")
        score -= 1.0
    
    # Check for path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        issues.append("Path traversal attempt detected")
        score -= 1.0
    
    score = max(0.0, score)
    safe = len(issues) == 0 and score >= 0.5
    
    return SafetyResult(
        safe=safe,
        score=score,
        reason="; ".join(issues) if issues else None,
        details={'issues': issues, 'filename': filename, 'extension': ext}
    )


def detect_data_exfiltration(
    text: str,
    threshold: float = 0.5
) -> SafetyResult:
    """Detect data exfiltration attempts.
    
    Args:
        text: Text to check
        threshold: Detection sensitivity (0.0-1.0)
        
    Returns:
        SafetyResult with exfiltration detection
    """
    exfil_patterns = [
        r'send\s+(?:this|data|information)\s+to',
        r'email\s+(?:this|data|information)\s+to',
        r'post\s+(?:this|data|information)\s+to',
        r'upload\s+(?:this|data|information)\s+to',
        r'transmit\s+(?:this|data|information)',
        r'leak\s+(?:this|data|information)',
    ]
    
    text_lower = text.lower()
    matches = sum(1 for pattern in exfil_patterns if re.search(pattern, text_lower, re.IGNORECASE))
    
    # Also check for multiple URLs (potential data drop sites)
    urls = detect_url(text)
    if len(urls) > 3:
        matches += 1
    
    score = max(0.0, 1.0 - (matches / (len(exfil_patterns) + 1)))
    safe = score >= threshold and matches == 0
    
    reason = f"Detected potential data exfiltration ({matches} indicators)" if not safe else None
    
    return SafetyResult(
        safe=safe,
        score=score,
        category=ContentCategory.MALICIOUS,
        confidence=0.75,
        reason=reason,
        details={'matched_patterns': matches, 'url_count': len(urls)}
    )
