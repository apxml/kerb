"""Safety utilities for LLM applications.

This module provides comprehensive tools for safety and security in LLM applications.

Enums:
    SafetyLevel - Safety check strictness level (PERMISSIVE, MODERATE, STRICT)
    ContentCategory - Content classification categories
    PIIType - Types of personally identifiable information
    ToxicityLevel - Toxicity severity levels

Data Classes:
    SafetyResult - Result from safety check with score and metadata
    PIIMatch - Detected PII with type, location, and confidence
    ModerationResult - Comprehensive moderation check result
    Guardrail - Custom safety guardrail definition

Content Moderation:
    moderate_content() - Check content against multiple safety categories
    check_toxicity() - Detect toxic, hateful, or harmful content
    check_sexual_content() - Detect sexual or adult content
    check_violence() - Detect violent content
    check_hate_speech() - Detect hate speech or discrimination
    check_self_harm() - Detect self-harm related content
    check_profanity() - Detect profane or offensive language

PII Detection & Redaction:
    detect_pii() - Detect personally identifiable information
    redact_pii() - Remove or mask PII from text
    detect_email() - Detect email addresses
    detect_phone() - Detect phone numbers
    detect_ssn() - Detect social security numbers
    detect_credit_card() - Detect credit card numbers
    detect_ip_address() - Detect IP addresses
    detect_url() - Detect URLs
    anonymize_text() - Replace PII with anonymized placeholders

Prompt Injection Detection:
    detect_prompt_injection() - Detect prompt injection attempts
    detect_jailbreak() - Detect jailbreak attempts
    detect_system_prompt_leak() - Detect attempts to leak system prompts
    detect_role_confusion() - Detect role confusion attacks
    check_input_safety() - Comprehensive input safety check

Output Validation & Filtering:
    validate_output() - Validate LLM output against safety rules
    filter_output() - Filter or sanitize LLM output
    check_output_safety() - Comprehensive output safety check
    ensure_safe_json() - Validate JSON output for safety
    detect_code_injection() - Detect code injection in outputs

Guardrails & Policies:
    create_guardrail() - Create a custom safety guardrail
    apply_guardrails() - Apply multiple guardrails to content
    check_content_policy() - Check against custom content policy
    validate_against_rules() - Validate content against rule set

Security & Privacy:
    sanitize_input() - Clean and sanitize user input
    escape_special_chars() - Escape potentially dangerous characters
    validate_url_safety() - Check if URL is safe
    check_file_upload() - Validate uploaded file content
    detect_data_exfiltration() - Detect data exfiltration attempts

Pattern Matching & Classification:
    match_patterns() - Match text against safety patterns
    classify_content() - Classify content into safety categories
    score_content() - Score content for safety risk
    extract_entities() - Extract sensitive entities from text

Submodules:
    moderation - Content moderation functions
    pii - PII detection and redaction
    injection - Prompt injection and jailbreak detection
    validation - Output validation and filtering
    guardrails - Custom guardrails and policies
    security - Security and privacy utilities
    classification - Content classification and pattern matching
"""

# Submodules for specialized usage
from . import (classification, guardrails, injection, moderation, pii,
               security, validation)
from .classification import (classify_content, extract_entities,
                             match_patterns, score_content)
# Core types and enums
from .enums import ContentCategory, PIIType, SafetyLevel, ToxicityLevel
from .guardrails import (apply_guardrails, check_content_policy,
                         create_guardrail, validate_against_rules)
from .injection import (check_input_safety, detect_jailbreak,
                        detect_prompt_injection, detect_role_confusion,
                        detect_system_prompt_leak)
# Most commonly used functions from each submodule
from .moderation import (check_hate_speech, check_profanity, check_self_harm,
                         check_sexual_content, check_toxicity, check_violence,
                         moderate_content)
from .pii import (anonymize_text, detect_credit_card, detect_email,
                  detect_ip_address, detect_phone, detect_pii, detect_ssn,
                  detect_url, redact_pii)
from .security import (check_file_upload, detect_data_exfiltration,
                       escape_special_chars, sanitize_input,
                       validate_url_safety)
from .types import Guardrail, ModerationResult, PIIMatch, SafetyResult
from .validation import (check_output_safety, detect_code_injection,
                         ensure_safe_json, filter_output, validate_output)

__all__ = [
    # Enums
    "SafetyLevel",
    "ContentCategory",
    "PIIType",
    "ToxicityLevel",
    # Data classes
    "SafetyResult",
    "PIIMatch",
    "ModerationResult",
    "Guardrail",
    # Submodules
    "moderation",
    "pii",
    "injection",
    "validation",
    "guardrails",
    "security",
    "classification",
    # Content moderation
    "moderate_content",
    "check_toxicity",
    "check_sexual_content",
    "check_violence",
    "check_hate_speech",
    "check_self_harm",
    "check_profanity",
    # PII detection & redaction
    "detect_pii",
    "redact_pii",
    "detect_email",
    "detect_phone",
    "detect_ssn",
    "detect_credit_card",
    "detect_ip_address",
    "detect_url",
    "anonymize_text",
    # Prompt injection detection
    "detect_prompt_injection",
    "detect_jailbreak",
    "detect_system_prompt_leak",
    "detect_role_confusion",
    "check_input_safety",
    # Output validation & filtering
    "validate_output",
    "filter_output",
    "check_output_safety",
    "ensure_safe_json",
    "detect_code_injection",
    # Guardrails & policies
    "create_guardrail",
    "apply_guardrails",
    "check_content_policy",
    "validate_against_rules",
    # Security & privacy
    "sanitize_input",
    "escape_special_chars",
    "validate_url_safety",
    "check_file_upload",
    "detect_data_exfiltration",
    # Pattern matching & classification
    "match_patterns",
    "classify_content",
    "score_content",
    "extract_entities",
]
