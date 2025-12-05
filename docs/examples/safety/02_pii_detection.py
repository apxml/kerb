"""
PII Detection and Redaction Example
===================================

This example demonstrates how to detect and redact personally identifiable
information (PII) from user inputs and LLM outputs to protect privacy.

Main concepts:
- Detecting various PII types (email, phone, SSN, credit cards)
- Redacting PII from text
- Anonymizing text with placeholders
- Protecting user privacy in LLM conversations
"""

from kerb.safety import (
    detect_pii,
    redact_pii,
    anonymize_text,
    detect_email,
    detect_phone,
    detect_credit_card,
    PIIType
)


def main():
    """Run PII detection and redaction example."""
    
    print("="*80)
    print("PII DETECTION AND REDACTION EXAMPLE")
    print("="*80)
    
    # Example 1: Basic PII detection
    print("\nExample 1: Basic PII Detection")
    print("-"*80)
    
    user_message = "My email is john.doe@example.com and phone is 555-123-4567"
    matches = detect_pii(user_message)
    
    print(f"Text: {user_message}")
    print(f"Found {len(matches)} PII instances:")
    for match in matches:
        print(f"  - {match.pii_type.value}: {match.text} (confidence: {match.confidence:.2f})")
    
    # Example 2: Redacting PII
    print("\n\nExample 2: Redacting PII")
    print("-"*80)
    
    sensitive_text = "Contact me at jane@company.com or call 555-987-6543"
    redacted = redact_pii(sensitive_text)
    
    print(f"Original: {sensitive_text}")
    print(f"Redacted: {redacted}")
    
    # Example 3: Anonymizing with placeholders
    print("\n\nExample 3: Anonymizing with Placeholders")
    print("-"*80)
    
    customer_query = "I need help with my account john.smith@email.com, card 4532-1234-5678-9010"
    anonymized = anonymize_text(customer_query)
    
    print(f"Original: {customer_query}")
    print(f"Anonymized: {anonymized}")
    
    # Example 4: Specific PII type detection
    print("\n\nExample 4: Specific PII Type Detection")
    print("-"*80)
    
    mixed_text = """

# %%
# Setup and Imports
# -----------------
    Customer details:
    Email: support@example.com
    Phone: (555) 123-4567
    SSN: 123-45-6789
    Card: 4532 1234 5678 9010
    Website: https://example.com
    IP: 192.168.1.1
    """
    
    print("Detecting specific PII types:")
    
    # Detect emails
    emails = detect_email(mixed_text)
    print(f"\nEmails ({len(emails)}):")
    for match in emails:
        print(f"  - {match.text}")
    
    # Detect phones
    phones = detect_phone(mixed_text)
    print(f"\nPhones ({len(phones)}):")
    for match in phones:
        print(f"  - {match.text}")
    
    # Detect credit cards
    cards = detect_credit_card(mixed_text)
    print(f"\nCredit Cards ({len(cards)}):")
    for match in cards:
        print(f"  - {match.text}")
    
    # Example 5: Selective PII detection
    print("\n\nExample 5: Selective PII Detection")
    print("-"*80)
    
    text = "Email me at bob@test.com, call 555-0000, or visit http://mysite.com"
    
    # Only detect emails and phones, ignore URLs
    matches = detect_pii(text, pii_types=[PIIType.EMAIL, PIIType.PHONE])
    
    print(f"Text: {text}")
    print(f"Detecting only: Email and Phone")
    print(f"Found {len(matches)} instances:")
    for match in matches:
        print(f"  - {match.pii_type.value}: {match.text}")
    
    # Example 6: LLM conversation privacy protection
    print("\n\nExample 6: LLM Conversation Privacy Protection")
    print("-"*80)
    
    conversations = [
        {
            "user": "I'm having trouble logging in with john@example.com",
            "assistant": "I'll help you with account john@example.com"
        },
        {
            "user": "Call me at 555-1234 if you need more info",
            "assistant": "I've noted your phone number 555-1234 for follow-up"
        }
    ]
    
    print("Protecting conversation history:")
    for i, conv in enumerate(conversations, 1):
        print(f"\nConversation {i}:")
        
        # Detect PII in user input
        user_pii = detect_pii(conv["user"])
        print(f"  User: {conv['user']}")
        if user_pii:
            print(f"  WARNING: User shared {len(user_pii)} PII item(s)")
            anonymized_user = anonymize_text(conv["user"])
            print(f"  Stored as: {anonymized_user}")
        
        # Check if assistant leaked PII
        assistant_pii = detect_pii(conv["assistant"])
        print(f"  Assistant: {conv['assistant']}")
        if assistant_pii:
            print(f"  ALERT: Assistant exposed {len(assistant_pii)} PII item(s)")
            print(f"  Should redact before logging!")
            redacted_response = redact_pii(conv["assistant"])
            print(f"  Logged as: {redacted_response}")
    
    # Example 7: Custom redaction for logging
    print("\n\nExample 7: Custom Redaction for Logging")
    print("-"*80)
    
    log_entry = "User alice@company.com attempted login from IP 10.0.0.5 using card ending in 9010"
    
    # Original (unsafe for logs)
    print(f"Unsafe log: {log_entry}")
    
    # Redacted for security
    safe_log = redact_pii(log_entry)
    print(f"Safe log: {safe_log}")
    
    # Anonymized for analytics
    analytics_log = anonymize_text(log_entry)
    print(f"Analytics log: {analytics_log}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
