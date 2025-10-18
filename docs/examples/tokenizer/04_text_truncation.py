"""Text Truncation Example

This example demonstrates how to truncate text to fit within token limits,
which is essential for LLM applications that need to stay within context
windows, manage API costs, or prepare fixed-size inputs.

Main concepts:
- Truncating text to token limits
- Preserving beginning vs. end of text
- Using custom ellipsis markers
- Smart truncation strategies
"""

from kerb.tokenizer import (
    count_tokens,
    truncate_to_token_limit,
    Tokenizer
)


def main():
    """Run text truncation examples."""
    
    print("="*80)
    print("TEXT TRUNCATION EXAMPLE")
    print("="*80)
    
    # Example 1: Basic truncation from the end
    print("\n" + "-"*80)
    print("EXAMPLE 1: Basic Truncation (Preserve Beginning)")
    print("-"*80)
    
    long_text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a long piece of text that needs to be truncated to fit within "
        "a specific token limit. We want to preserve the beginning of the text "
        "because it usually contains the most important information in many contexts."
    )
    
    print(f"Original text ({len(long_text)} chars):")
    print(f"  {long_text}")
    print(f"  Original tokens: {count_tokens(long_text, tokenizer=Tokenizer.CL100K_BASE)}")
    
    max_tokens = 20
    truncated = truncate_to_token_limit(
        long_text,
        max_tokens=max_tokens,
        tokenizer=Tokenizer.CL100K_BASE,
        preserve_end=False
    )
    
    print(f"\nTruncated to {max_tokens} tokens (preserve beginning):")
    print(f"  {truncated}")
    print(f"  Truncated tokens: {count_tokens(truncated, tokenizer=Tokenizer.CL100K_BASE)}")
    print(f"  Length: {len(truncated)} chars")
    
    # Example 2: Truncation preserving the end
    print("\n" + "-"*80)
    print("EXAMPLE 2: Truncation (Preserve End)")
    print("-"*80)
    
    text_with_important_end = (
        "Please process the following request. The system should handle various "
        "edge cases and ensure proper error handling. Most importantly, "
        "set the status to APPROVED and the priority to HIGH."
    )
    
    print(f"Original text:")
    print(f"  {text_with_important_end}")
    print(f"  Original tokens: {count_tokens(text_with_important_end, tokenizer=Tokenizer.CL100K_BASE)}")
    
    max_tokens = 15
    truncated_end = truncate_to_token_limit(
        text_with_important_end,
        max_tokens=max_tokens,
        tokenizer=Tokenizer.CL100K_BASE,
        preserve_end=True
    )
    
    print(f"\nTruncated to {max_tokens} tokens (preserve end):")
    print(f"  {truncated_end}")
    print(f"  Truncated tokens: {count_tokens(truncated_end, tokenizer=Tokenizer.CL100K_BASE)}")
    
    # Example 3: Custom ellipsis markers
    print("\n" + "-"*80)
    print("EXAMPLE 3: Custom Ellipsis Markers")
    print("-"*80)
    
    documentation = (
        "This function takes a list of integers as input and returns the sum. "
        "The implementation uses a simple loop to iterate through all elements. "
        "Time complexity is O(n) where n is the length of the input list."
    )
    
    print(f"Original text:")
    print(f"  {documentation}\n")
    
    ellipsis_options = [
        "...",
        " [truncated]",
        " [...more...]",
        " [CONTENT OMITTED]",
    ]
    
    max_tokens = 15
    
    for ellipsis in ellipsis_options:
        truncated = truncate_to_token_limit(
            documentation,
            max_tokens=max_tokens,
            tokenizer=Tokenizer.CL100K_BASE,
            ellipsis=ellipsis
        )
        tokens = count_tokens(truncated, tokenizer=Tokenizer.CL100K_BASE)
        print(f"Ellipsis '{ellipsis}':")
        print(f"  {truncated}")
        print(f"  Tokens: {tokens}\n")
    
    # Example 4: Truncating code
    print("\n" + "-"*80)
    print("EXAMPLE 4: Truncating Code Snippets")
    print("-"*80)
    
    code_snippet = """def calculate_metrics(data):
    # Calculate various metrics
    total = sum(data)
    count = len(data)
    average = total / count if count > 0 else 0
    maximum = max(data) if data else 0
    minimum = min(data) if data else 0
    return {
        'total': total,
        'average': average,
        'max': maximum,
        'min': minimum
    }"""
    
    print(f"Original code ({count_tokens(code_snippet, tokenizer=Tokenizer.CL100K_BASE)} tokens):")
    print(code_snippet)
    
    max_tokens = 30
    truncated_code = truncate_to_token_limit(
        code_snippet,
        max_tokens=max_tokens,
        tokenizer=Tokenizer.CL100K_BASE,
        ellipsis="\n    # ... rest of function"
    )
    
    print(f"\nTruncated to {max_tokens} tokens:")
    print(truncated_code)
    
    # Example 5: Batch truncation for multiple documents
    print("\n" + "-"*80)
    print("EXAMPLE 5: Batch Document Truncation")
    print("-"*80)
    
    documents = [
        "Short document that doesn't need truncation.",
        "This is a medium-length document that contains several sentences and provides detailed information about a topic. " * 3,
        "A",
        "Another very long document with extensive details. " * 10,
    ]
    
    max_tokens = 25
    print(f"Truncating {len(documents)} documents to max {max_tokens} tokens each:\n")
    
    for i, doc in enumerate(documents, 1):
        original_tokens = count_tokens(doc, tokenizer=Tokenizer.CL100K_BASE)
        truncated = truncate_to_token_limit(
            doc,
            max_tokens=max_tokens,
            tokenizer=Tokenizer.CL100K_BASE
        )
        new_tokens = count_tokens(truncated, tokenizer=Tokenizer.CL100K_BASE)
        
        print(f"Document {i}:")
        print(f"  Original: {original_tokens} tokens, {len(doc)} chars")
        print(f"  Truncated: {new_tokens} tokens, {len(truncated)} chars")
        print(f"  Preview: {truncated[:70]}...")
        
        if original_tokens <= max_tokens:
            print(f"  Status: No truncation needed")
        else:
            print(f"  Status: Truncated ({original_tokens - new_tokens} tokens removed)")
        print()
    
    # Example 6: Smart truncation for summaries
    print("\n" + "-"*80)
    print("EXAMPLE 6: Smart Truncation for Document Summaries")
    print("-"*80)
    
    article = """Breaking News: New AI Model Released
    
A new artificial intelligence model has been released today by researchers at TechLab.
The model, called SuperAI-2024, shows significant improvements over previous versions.

Key features include:
- 40% faster inference speed
- 25% better accuracy on benchmark tests
- Support for 15 additional languages
- Reduced training costs

The research team stated that this model represents a major step forward in AI capabilities.
Industry experts are already testing the model and providing positive feedback.
The model is available for commercial use under a new licensing agreement."""

    print("Original article:")
    print(article)
    print(f"\nOriginal tokens: {count_tokens(article, tokenizer=Tokenizer.CL100K_BASE)}")
    
    # Create different length summaries
    summary_lengths = [50, 30, 15]
    
    print("\nCreating summaries of different lengths:\n")
    
    for max_length in summary_lengths:
        summary = truncate_to_token_limit(
            article,
            max_tokens=max_length,
            tokenizer=Tokenizer.CL100K_BASE,
            ellipsis="... [Read more]"
        )
        tokens = count_tokens(summary, tokenizer=Tokenizer.CL100K_BASE)
        print(f"{max_length}-token summary ({tokens} actual tokens):")
        print(f"  {summary}")
        print()
    
    # Example 7: Preserving critical information
    print("\n" + "-"*80)
    print("EXAMPLE 7: Preserving Critical Information")
    print("-"*80)
    
    # When you need to preserve critical info, restructure before truncation
    log_entry = (
        "2024-10-15 14:30:22 INFO Processing started for batch_id=12345 "
        "with 150 items. System load: 45%. Memory usage: 2.3GB. "
        "Previous batches completed successfully. Current weather: sunny. "
        "ERROR: Failed to process item_id=67890 due to invalid format. "
        "Status: FAILED. Error code: E404."
    )
    
    print("Original log entry:")
    print(f"  {log_entry}")
    print(f"  Tokens: {count_tokens(log_entry, tokenizer=Tokenizer.CL100K_BASE)}")
    
    # Strategy 1: Truncate from beginning (loses important error at end)
    max_tokens = 25
    truncated_beginning = truncate_to_token_limit(
        log_entry,
        max_tokens=max_tokens,
        tokenizer=Tokenizer.CL100K_BASE,
        preserve_end=False
    )
    
    print(f"\nStrategy 1: Truncate from end (preserve beginning):")
    print(f"  {truncated_beginning}")
    print(f"  Problem: Lost the critical ERROR information!")
    
    # Strategy 2: Truncate from end (preserves error)
    truncated_end = truncate_to_token_limit(
        log_entry,
        max_tokens=max_tokens,
        tokenizer=Tokenizer.CL100K_BASE,
        preserve_end=True
    )
    
    print(f"\nStrategy 2: Truncate from beginning (preserve end):")
    print(f"  {truncated_end}")
    print(f"  Better: Preserves the critical ERROR information!")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
