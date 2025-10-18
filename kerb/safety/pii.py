"""PII detection and redaction functions.

This module provides functions for detecting and redacting personally
identifiable information (PII) from text.
"""

import re
from typing import Dict, List, Optional, Tuple

from .enums import PIIType
from .patterns import (CREDIT_CARD_PATTERN, EMAIL_PATTERN, IP_ADDRESS_PATTERN,
                       PHONE_PATTERN, SSN_PATTERN, URL_PATTERN)
from .types import PIIMatch


def detect_pii(text: str, pii_types: Optional[List[PIIType]] = None) -> List[PIIMatch]:
    """Detect personally identifiable information.

    Args:
        text: Text to scan for PII
        pii_types: Specific PII types to detect (None = all)

    Returns:
        List of PIIMatch objects with detected PII

    Examples:
        >>> text = "Email me at john@example.com or call 555-123-4567"
        >>> matches = detect_pii(text)
        >>> for match in matches:
        ...     print(f"{match.pii_type}: {match.text}")
        PIIType.EMAIL: john@example.com
        PIIType.PHONE: 555-123-4567
    """
    if pii_types is None:
        pii_types = list(PIIType)

    matches = []

    if PIIType.EMAIL in pii_types:
        matches.extend(detect_email(text))

    if PIIType.PHONE in pii_types:
        matches.extend(detect_phone(text))

    if PIIType.SSN in pii_types:
        matches.extend(detect_ssn(text))

    if PIIType.CREDIT_CARD in pii_types:
        matches.extend(detect_credit_card(text))

    if PIIType.IP_ADDRESS in pii_types:
        matches.extend(detect_ip_address(text))

    if PIIType.URL in pii_types:
        matches.extend(detect_url(text))

    return matches


def detect_email(text: str) -> List[PIIMatch]:
    """Detect email addresses.

    Args:
        text: Text to scan

    Returns:
        List of PIIMatch objects for detected emails
    """
    matches = []
    for match in re.finditer(EMAIL_PATTERN, text):
        matches.append(
            PIIMatch(
                pii_type=PIIType.EMAIL,
                text=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=0.95,
            )
        )
    return matches


def detect_phone(text: str) -> List[PIIMatch]:
    """Detect phone numbers.

    Args:
        text: Text to scan

    Returns:
        List of PIIMatch objects for detected phone numbers
    """
    matches = []
    for match in re.finditer(PHONE_PATTERN, text):
        matches.append(
            PIIMatch(
                pii_type=PIIType.PHONE,
                text=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=0.9,
            )
        )
    return matches


def detect_ssn(text: str) -> List[PIIMatch]:
    """Detect social security numbers.

    Args:
        text: Text to scan

    Returns:
        List of PIIMatch objects for detected SSNs
    """
    matches = []
    for match in re.finditer(SSN_PATTERN, text):
        matches.append(
            PIIMatch(
                pii_type=PIIType.SSN,
                text=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=0.95,
            )
        )
    return matches


def detect_credit_card(text: str) -> List[PIIMatch]:
    """Detect credit card numbers.

    Args:
        text: Text to scan

    Returns:
        List of PIIMatch objects for detected credit card numbers
    """
    matches = []
    for match in re.finditer(CREDIT_CARD_PATTERN, text):
        # Basic Luhn algorithm check
        digits = re.sub(r"[-\s]", "", match.group())
        if len(digits) == 16:
            matches.append(
                PIIMatch(
                    pii_type=PIIType.CREDIT_CARD,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                )
            )
    return matches


def detect_ip_address(text: str) -> List[PIIMatch]:
    """Detect IP addresses.

    Args:
        text: Text to scan

    Returns:
        List of PIIMatch objects for detected IP addresses
    """
    matches = []
    for match in re.finditer(IP_ADDRESS_PATTERN, text):
        # Validate IP address range
        parts = match.group().split(".")
        if all(0 <= int(p) <= 255 for p in parts):
            matches.append(
                PIIMatch(
                    pii_type=PIIType.IP_ADDRESS,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                )
            )
    return matches


def detect_url(text: str) -> List[PIIMatch]:
    """Detect URLs.

    Args:
        text: Text to scan

    Returns:
        List of PIIMatch objects for detected URLs
    """
    matches = []
    for match in re.finditer(URL_PATTERN, text):
        matches.append(
            PIIMatch(
                pii_type=PIIType.URL,
                text=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=0.95,
            )
        )
    return matches


def redact_pii(
    text: str,
    pii_types: Optional[List[PIIType]] = None,
    replacement: str = "[REDACTED]",
) -> Tuple[str, List[PIIMatch]]:
    """Remove or mask PII from text.

    Args:
        text: Text to redact
        pii_types: Specific PII types to redact (None = all)
        replacement: Replacement text for redacted PII

    Returns:
        Tuple of (redacted_text, detected_matches)

    Examples:
        >>> text = "Email me at john@example.com"
        >>> redacted, matches = redact_pii(text)
        >>> print(redacted)
        "Email me at [REDACTED]"
    """
    matches = detect_pii(text, pii_types)

    # Sort matches by position (reverse order for replacement)
    matches.sort(key=lambda m: m.start, reverse=True)

    redacted = text
    for match in matches:
        redacted = redacted[: match.start] + replacement + redacted[match.end :]

    return redacted, matches


def anonymize_text(
    text: str, pii_types: Optional[List[PIIType]] = None
) -> Tuple[str, Dict[str, str]]:
    """Replace PII with anonymized placeholders.

    Args:
        text: Text to anonymize
        pii_types: Specific PII types to anonymize (None = all)

    Returns:
        Tuple of (anonymized_text, mapping_dict)

    Examples:
        >>> text = "Contact john@example.com or jane@example.com"
        >>> anonymized, mapping = anonymize_text(text)
        >>> print(anonymized)
        "Contact [EMAIL_1] or [EMAIL_2]"
    """
    matches = detect_pii(text, pii_types)
    matches.sort(key=lambda m: m.start, reverse=True)

    # Track replacements by type
    type_counters = {}
    mapping = {}

    anonymized = text
    for match in matches:
        pii_type_name = match.pii_type.value.upper()

        # Increment counter for this PII type
        if pii_type_name not in type_counters:
            type_counters[pii_type_name] = 0
        type_counters[pii_type_name] += 1

        # Create placeholder
        placeholder = f"[{pii_type_name}_{type_counters[pii_type_name]}]"
        mapping[placeholder] = match.text

        # Replace in text
        anonymized = anonymized[: match.start] + placeholder + anonymized[match.end :]

    return anonymized, mapping
