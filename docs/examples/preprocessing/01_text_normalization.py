"""
Text Normalization for Training Data
====================================

This example demonstrates text normalization techniques essential for preparing
clean, consistent training data for LLMs.

Main concepts:
- Cleaning raw web-scraped or user-generated content
- Unicode normalization for consistency
- Removing unwanted elements (URLs, emails, special chars)
- Normalizing whitespace and quotes
- Different normalization levels for different use cases

Use case: Preparing web-scraped data for LLM fine-tuning
"""

from kerb.preprocessing import (
    normalize_text,
    normalize_whitespace,
    normalize_unicode,
    normalize_quotes,
    clean_html,
    clean_markdown,
    remove_urls,
    remove_emails,
    NormalizationLevel,
    NormalizationConfig
)


def main():
    """Run text normalization examples."""
    
    print("="*80)
    print("TEXT NORMALIZATION FOR TRAINING DATA")
    print("="*80)
    
    # Example 1: Raw web-scraped content with various issues
    print("\n" + "-"*80)
    print("Example 1: Cleaning Web-Scraped Content")
    print("-"*80)
    
    raw_web_content = """

# %%
# Setup and Imports
# -----------------
    Check out our site: https://example.com   
    Contact us at: support@example.com
    
    "Smart quotes" and 'curly apostrophes'  —  dashes everywhere!
    
    Multiple     spaces    and      tabs	here.
    
    
    Too many newlines above!
    """
    
    print("\nRaw content:")
    print(repr(raw_web_content[:100]))
    
    # Clean with standard normalization
    cleaned = normalize_text(
        raw_web_content,
        level=NormalizationLevel.STANDARD,
        lowercase=False,
        remove_urls=True,
        remove_emails=True
    )
    
    print("\nCleaned content:")
    print(cleaned)
    
    # Example 2: Using NormalizationConfig (recommended approach)
    print("\n" + "-"*80)
    print("Example 2: Using NormalizationConfig")
    print("-"*80)
    
    # Create reusable config
    training_config = NormalizationConfig(
        level=NormalizationLevel.STANDARD,
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_extra_spaces=True
    )
    
    user_content = [
        "I LOVE this product!!! 😊 http://promo.link",
        "Contact me: john.doe@email.com for more info",
        "This   has    weird     spacing",
    ]
    
    print("\nProcessing user-generated content for training:")
    for i, text in enumerate(user_content, 1):
        normalized = normalize_text(text, config=training_config)
        print(f"\n{i}. Original: {text}")
        print(f"   Normalized: {normalized}")
    
    # Example 3: Different normalization levels
    print("\n" + "-"*80)
    print("Example 3: Normalization Levels")
    print("-"*80)
    
    messy_text = "Check THIS out!!! Special chars: @#$% and quotes: \"hello\""
    
    levels = [
        NormalizationLevel.MINIMAL,
        NormalizationLevel.STANDARD,
        NormalizationLevel.AGGRESSIVE
    ]
    
    print(f"\nOriginal: {messy_text}")
    for level in levels:
        result = normalize_text(messy_text, level=level, lowercase=False)
        print(f"{level.value:12s}: {result}")
    
    # Example 4: HTML and Markdown cleaning
    print("\n" + "-"*80)
    print("Example 4: Structured Content Cleaning")
    print("-"*80)
    
    html_content = "<p>This is <strong>bold</strong> and <em>italic</em> text.</p>"
    markdown_content = "# Header\n\nThis is **bold** and *italic* text."
    
    print(f"\nHTML: {html_content}")
    print(f"Cleaned: {clean_html(html_content)}")
    
    print(f"\nMarkdown: {markdown_content}")
    print(f"Cleaned: {clean_markdown(markdown_content)}")
    
    # Example 5: Unicode normalization
    print("\n" + "-"*80)
    print("Example 5: Unicode Normalization")
    print("-"*80)
    
    # Same visual appearance, different unicode representations
    text1 = "café"  # e + combining acute accent
    text2 = "café"  # precomposed character
    
    print(f"\nText 1 bytes: {text1.encode('utf-8')}")
    print(f"Text 2 bytes: {text2.encode('utf-8')}")
    print(f"Are they equal? {text1 == text2}")
    
    # Normalize both
    norm1 = normalize_unicode(text1)
    norm2 = normalize_unicode(text2)
    
    print(f"\nAfter normalization:")
    print(f"Normalized 1: {norm1.encode('utf-8')}")
    print(f"Normalized 2: {norm2.encode('utf-8')}")
    print(f"Are they equal now? {norm1 == norm2}")
    
    # Example 6: Preparing chat/instruction data
    print("\n" + "-"*80)
    print("Example 6: Instruction Data Preparation")
    print("-"*80)
    
    # Simulated instruction-response pairs with noise
    instruction_pairs = [
        {
            "instruction": "  Explain   what  AI  is.  ",
            "response": "AI (Artificial Intelligence) is... Visit: http://learn-more.com"
        },
        {
            "instruction": "How do I cook pasta???",
            "response": "To cook pasta: \n\n\n1. Boil water\n2. Add pasta\n\n\nContact: chef@cooking.com"
        }
    ]
    
    # Clean instruction config
    instruction_config = NormalizationConfig(
        level=NormalizationLevel.STANDARD,
        lowercase=False,  # Keep case for instructions
        remove_urls=True,
        remove_emails=True,
        remove_extra_spaces=True
    )
    
    print("\nCleaning instruction-response pairs:")
    for i, pair in enumerate(instruction_pairs, 1):
        clean_instruction = normalize_text(pair["instruction"], config=instruction_config)
        clean_response = normalize_text(pair["response"], config=instruction_config)
        
        print(f"\nPair {i}:")
        print(f"  Instruction: {clean_instruction}")
        print(f"  Response: {clean_response}")
    
    # Example 7: Whitespace-only normalization
    print("\n" + "-"*80)
    print("Example 7: Preserving Content, Normalizing Whitespace")
    print("-"*80)
    
    code_snippet = """

# %%
#   Hello
# -------

    def   hello():
        print("world")
        
        
        return   True
    """
    
    print("\nOriginal code:")
    print(repr(code_snippet))
    
    # Just normalize whitespace, keep everything else
    normalized_code = normalize_whitespace(code_snippet)
    print("\nWhitespace-normalized code:")
    print(repr(normalized_code))
    
    print("\n" + "="*80)
    print("TEXT NORMALIZATION COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Use NormalizationConfig for consistent, reusable settings")
    print("2. Choose normalization level based on use case")
    print("3. Always normalize unicode for consistency")
    print("4. Remove PII (emails, URLs) from training data when appropriate")
    print("5. Different content types may need different normalization strategies")


if __name__ == "__main__":
    main()
