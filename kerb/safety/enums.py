"""Safety-related enumerations.

This module defines all enum types used in the safety subpackage.
"""

from enum import Enum


class SafetyLevel(Enum):
    """Safety check strictness level."""

    PERMISSIVE = "permissive"  # Minimal filtering
    MODERATE = "moderate"  # Balanced approach
    STRICT = "strict"  # Maximum safety


class ContentCategory(Enum):
    """Content classification categories."""

    SAFE = "safe"
    TOXICITY = "toxicity"
    SEXUAL = "sexual"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SELF_HARM = "self_harm"
    PROFANITY = "profanity"
    SPAM = "spam"
    MALICIOUS = "malicious"


class PIIType(Enum):
    """Types of personally identifiable information."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    URL = "url"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    ACCOUNT_NUMBER = "account_number"


class ToxicityLevel(Enum):
    """Toxicity severity levels."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    SEVERE = 4
