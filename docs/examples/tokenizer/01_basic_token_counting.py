"""Basic Token Counting Example

This example demonstrates the fundamental token counting capabilities
for LLM applications using different tokenizers.

Main concepts:
- Using different tokenizer types (CL100K_BASE, P50K_BASE, approximation)
- Counting tokens in text
- Understanding tokenizer differences
- Choosing the right tokenizer for your model
"""

from kerb.tokenizer import count_tokens, Tokenizer


def main():
    """Run basic token counting examples."""
    
    print("="*80)
    print("BASIC TOKEN COUNTING EXAMPLE")
    print("="*80)
    
    # Sample text
    text = "The quick brown fox jumps over the lazy dog. This is a sample sentence for token counting."
    
    print(f"\nText: {text}")
    print(f"Character count: {len(text)}")
    
    # Example 1: Count tokens with CL100K_BASE (GPT-4, GPT-3.5-turbo)
    print("\n" + "-"*80)
    print("EXAMPLE 1: CL100K_BASE Tokenizer (GPT-4, GPT-3.5-turbo)")
    print("-"*80)
    
    tokens_cl100k = count_tokens(text, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Token count: {tokens_cl100k}")
    print(f"Tokens per character: {tokens_cl100k / len(text):.4f}")
    
    # Example 2: Count tokens with P50K_BASE (Code models)
    print("\n" + "-"*80)
    print("EXAMPLE 2: P50K_BASE Tokenizer (Code models)")
    print("-"*80)
    
    tokens_p50k = count_tokens(text, tokenizer=Tokenizer.P50K_BASE)
    print(f"Token count: {tokens_p50k}")
    print(f"Difference from CL100K_BASE: {tokens_p50k - tokens_cl100k} tokens")
    
    # Example 3: Fast approximation methods
    print("\n" + "-"*80)
    print("EXAMPLE 3: Fast Approximation Methods")
    print("-"*80)
    
    # CHAR_4 approximation (good for GPT-like models)
    tokens_char4 = count_tokens(text, tokenizer=Tokenizer.CHAR_4)
    print(f"CHAR_4 approximation: {tokens_char4} tokens")
    print(f"  Accuracy vs CL100K_BASE: {abs(tokens_char4 - tokens_cl100k) / tokens_cl100k * 100:.1f}% difference")
    
    # CHAR_5 approximation (good for BERT-like models)
    tokens_char5 = count_tokens(text, tokenizer=Tokenizer.CHAR_5)
    print(f"CHAR_5 approximation: {tokens_char5} tokens")
    
    # WORD approximation
    tokens_word = count_tokens(text, tokenizer=Tokenizer.WORD)
    print(f"WORD approximation: {tokens_word} tokens")
    
    # Example 4: Comparing different text types
    print("\n" + "-"*80)
    print("EXAMPLE 4: Token Counting for Different Text Types")
    print("-"*80)
    
    texts = {
        "Simple English": "Hello world! How are you today?",
        "Technical": "def process_data(input_list: List[str]) -> Dict[str, int]:",
        "Code": "const result = await Promise.all(items.map(async item => fetch(item.url)));",
        "Numbers": "1234567890 9876543210 1111111111 2222222222",
        "Mixed": "The API returned 404 error at 2024-10-15T14:30:00Z for user_id=12345",
    }
    
    for text_type, sample_text in texts.items():
        token_count = count_tokens(sample_text, tokenizer=Tokenizer.CL100K_BASE)
        char_count = len(sample_text)
        ratio = token_count / char_count
        print(f"\n{text_type}:")
        print(f"  Text: {sample_text[:60]}...")
        print(f"  Characters: {char_count}, Tokens: {token_count}, Ratio: {ratio:.4f}")
    
    # Example 5: Real-world use case - checking before API call
    print("\n" + "-"*80)
    print("EXAMPLE 5: Pre-API Token Check")
    print("-"*80)
    
    user_input = "Explain quantum computing in simple terms."
    system_prompt = "You are a helpful assistant that explains complex topics simply."
    
    system_tokens = count_tokens(system_prompt, tokenizer=Tokenizer.CL100K_BASE)
    user_tokens = count_tokens(user_input, tokenizer=Tokenizer.CL100K_BASE)
    total_input_tokens = system_tokens + user_tokens
    
    print(f"System prompt tokens: {system_tokens}")
    print(f"User input tokens: {user_tokens}")
    print(f"Total input tokens: {total_input_tokens}")
    
    # Assuming GPT-3.5-turbo with 4096 token limit
    context_limit = 4096
    max_completion_tokens = 500
    available_for_input = context_limit - max_completion_tokens
    
    print(f"\nContext window: {context_limit} tokens")
    print(f"Reserved for completion: {max_completion_tokens} tokens")
    print(f"Available for input: {available_for_input} tokens")
    print(f"Input usage: {total_input_tokens}/{available_for_input} tokens ({total_input_tokens/available_for_input*100:.1f}%)")
    
    if total_input_tokens > available_for_input:
        print("WARNING: Input exceeds available tokens!")
    else:
        print("OK: Input fits within available tokens")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
