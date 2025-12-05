"""
Content Filtering for Dataset Quality
=====================================

=====================================

This example demonstrates filtering techniques to ensure high-quality training data
for LLMs by removing low-quality, inappropriate, or irrel    print(f"\nOriginal dataset:")
    for i, text in enumerate(multilingual_dataset, 1):
        print(f"{i}. {text}")
    
    # Keep only ASCII (filter each text)
    ascii_only = [filter_non_ascii(text) for text in multilingual_dataset if filter_non_ascii(text).strip()]
    
    print(f"\nASCII-only filtered: {len(ascii_only)} samples")
    for text in ascii_only:
        print(f"  - {text}")ent.

Main concepts:
- Length-based filtering for quality control
- Pattern-based filtering to remove unwanted content
- PII removal for privacy compliance
- Quality metrics for dataset curation
- ASCII/encoding filtering

Use case: Curating high-quality datasets from web scrapes or user content
"""

from kerb.preprocessing import (
    filter_by_length,
    filter_by_pattern,
    filter_pii,
    filter_non_ascii,
    filter_by_quality,
    remove_urls,
    remove_emails,
    remove_phone_numbers,
)


def main():
    """Run content filtering examples."""
    
    print("="*80)
    print("CONTENT FILTERING FOR DATASET QUALITY")
    print("="*80)
    
    # Example 1: Length-based filtering
    print("\n" + "-"*80)
    print("Example 1: Length-Based Filtering")
    print("-"*80)
    
    # Simulated dataset with varying quality
    raw_dataset = [
        "ok",  # Too short
        "This is a good quality example sentence for training.",
        "Another quality sentence with meaningful content.",
        "x",  # Too short
        "Medium length text here.",
        "This is an extremely long sentence that goes on and on and on with repetitive content that doesn't add much value and could potentially cause issues during training because it's unnecessarily verbose and contains redundant information that could be better expressed more concisely.",  # Too long
        "Good example text.",
        "",  # Empty
    ]
    
    print(f"\nOriginal dataset: {len(raw_dataset)} samples")
    for i, text in enumerate(raw_dataset, 1):
        print(f"{i}. [{len(text):3d} chars] {text[:50]}...")
    
    # Filter by character length
    filtered_chars = filter_by_length(
        raw_dataset,
        min_length=10,
        max_length=200,
        unit="chars"
    )
    
    print(f"\nFiltered by chars (10-200): {len(filtered_chars)} samples")
    for text in filtered_chars:
        print(f"  - {text[:60]}...")
    
    # Filter by word count
    filtered_words = filter_by_length(
        raw_dataset,
        min_length=3,
        max_length=20,
        unit="words"
    )
    
    print(f"\nFiltered by words (3-20): {len(filtered_words)} samples")
    for text in filtered_words:
        print(f"  - {text[:60]}...")
    
    # Example 2: Pattern-based filtering
    print("\n" + "-"*80)
    print("Example 2: Pattern-Based Filtering")
    print("-"*80)
    
    mixed_dataset = [
        "This is normal text content.",
        "Code example: def foo(): pass",
        "Regular sentence here.",
        "import numpy as np",
        "More normal content for training.",
        "class MyClass: def __init__(self): pass",
        "Final normal sentence.",
    ]
    
    print("\nOriginal dataset:")
    for i, text in enumerate(mixed_dataset, 1):
        print(f"{i}. {text}")
    
    # Filter out code-like content
    code_patterns = r'(def\s+\w+\(|class\s+\w+:|import\s+\w+)'
    
    filtered_no_code = filter_by_pattern(
        mixed_dataset,
        pattern=code_patterns,
        keep_matches=False  # Keep non-matching (remove code)
    )
    
    print(f"\nFiltered (no code): {len(filtered_no_code)} samples")
    for text in filtered_no_code:
        print(f"  - {text}")
    
    # Keep only English text (starting with capital letter)
    english_text = filter_by_pattern(
        mixed_dataset,
        pattern=r'^[A-Z]',
        keep_matches=True
    )
    
    print(f"\nFiltered (English sentences): {len(english_text)} samples")
    for text in english_text:
        print(f"  - {text}")
    
    # Example 3: PII removal
    print("\n" + "-"*80)
    print("Example 3: PII Removal for Privacy")
    print("-"*80)
    
    pii_dataset = [
        "Contact me at john.doe@email.com for more information.",
        "My phone is 555-123-4567 and email is jane@company.com",
        "Visit our website at https://example.com or call (555) 987-6543",
        "This text has no PII information.",
        "You can reach support@help.com or visit http://help.example.org",
    ]
    
    print("\nOriginal dataset with PII:")
    for i, text in enumerate(pii_dataset, 1):
        print(f"{i}. {text}")
    
    # Remove PII
    print("\nRemoving emails:")
    for text in pii_dataset:
        cleaned = remove_emails(text)
        if cleaned != text:
            print(f"  Before: {text}")
            print(f"  After:  {cleaned}")
    
    print("\nRemoving phone numbers:")
    for text in pii_dataset:
        cleaned = remove_phone_numbers(text)
        if cleaned != text:
            print(f"  Before: {text}")
            print(f"  After:  {cleaned}")
    
    print("\nRemoving URLs:")
    for text in pii_dataset:
        cleaned = remove_urls(text)
        if cleaned != text:
            print(f"  Before: {text}")
            print(f"  After:  {cleaned}")
    
    # Combined PII removal
    print("\nCombined PII removal:")
    for text in pii_dataset:
        cleaned = remove_emails(remove_phone_numbers(remove_urls(text)))
        print(f"  Original: {text}")
        print(f"  Cleaned:  {cleaned}")
        print()
    
    # Example 4: ASCII filtering
    print("\n" + "-"*80)
    print("Example 4: Character Encoding Filtering")
    print("-"*80)
    
    multilingual_dataset = [
        "This is pure ASCII text.",
        "This has some émojis and àccents.",
        "日本語のテキスト (Japanese text)",
        "Regular English sentence.",
        "Текст на русском языке",
        "Another ASCII example.",
    ]
    
    print("\nOriginal dataset:")
    for i, text in enumerate(multilingual_dataset, 1):
        print(f"{i}. {text}")
    
    # Keep only ASCII (filter each text)
    ascii_only = [filter_non_ascii(text) for text in multilingual_dataset if filter_non_ascii(text).strip()]
    
    print(f"\nASCII-only filtered: {len(ascii_only)} samples")
    for text in ascii_only:
        print(f"  - {text}")
    
    # Example 5: Quality-based filtering
    print("\n" + "-"*80)
    print("Example 5: Quality Metrics Filtering")
    print("-"*80)
    
    quality_dataset = [
        "asdfjkl qwerty zxcvbn",  # Low quality - gibberish
        "This is a well-formed sentence with proper grammar.",
        "!!!!!!",  # Low quality - only punctuation
        "Another high-quality example with meaningful content.",
        "aaaaaaaaaaaaaaaaaaa",  # Low quality - repetitive
        "The quick brown fox jumps over the lazy dog.",
        "123 456 789",  # Low quality - only numbers
    ]
    
    print("\nOriginal dataset:")
    for i, text in enumerate(quality_dataset, 1):
        print(f"{i}. {text}")
    
    # Filter by quality score
    high_quality = filter_by_quality(
        quality_dataset,
        min_score=0.5  # Minimum quality threshold
    )
    
    print(f"\nHigh quality filtered: {len(high_quality)} samples")
    for text in high_quality:
        print(f"  - {text}")
    
    # Example 6: Multi-stage filtering pipeline
    print("\n" + "-"*80)
    print("Example 6: Multi-Stage Filtering Pipeline")
    print("-"*80)
    
    raw_web_scrape = [
        "x",  # Too short
        "This is good content from a blog post about machine learning.",
        "Contact us: sales@spam.com for amazing deals!!!",
        "日本語",  # Non-ASCII
        "A well-written article about natural language processing.",
        "",  # Empty
        "def code(): pass",  # Code
        "Great tutorial on deep learning with practical examples.",
        "asdfjkl qwerty",  # Low quality
        "Click here: http://suspicious-link.com",
        "Another excellent resource for learning AI.",
    ]
    
    print(f"\nOriginal dataset: {len(raw_web_scrape)} samples")
    
    # Stage 1: Length filter
    stage1 = filter_by_length(raw_web_scrape, min_length=20, unit="chars")
    print(f"After length filter: {len(stage1)} samples")
    
    # Stage 2: Remove code
    stage2 = filter_by_pattern(stage1, pattern=r'def\s+\w+\(', keep_matches=False)
    print(f"After code filter: {len(stage2)} samples")
    
    # Stage 3: ASCII only
    stage3 = [filter_non_ascii(text) for text in stage2 if filter_non_ascii(text).strip()]
    print(f"After ASCII filter: {len(stage3)} samples")
    
    # Stage 4: Quality filter
    stage4 = filter_by_quality(stage3, min_score=0.5)
    print(f"After quality filter: {len(stage4)} samples")
    
    print("\nFinal filtered dataset:")
    for i, text in enumerate(stage4, 1):
        print(f"{i}. {text}")
    
    print("\n" + "="*80)
    print("CONTENT FILTERING COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Use length filters to remove too-short or too-long samples")
    print("2. Pattern filters help remove specific content types")
    print("3. Always remove PII for privacy compliance")
    print("4. Quality filters improve overall dataset quality")
    print("5. Multi-stage pipelines provide comprehensive filtering")
    print("6. Balance filtering strictness with dataset size needs")


if __name__ == "__main__":
    main()
