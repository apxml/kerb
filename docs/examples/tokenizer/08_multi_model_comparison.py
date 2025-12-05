"""
Multi-Model Comparison Example
==============================

This example demonstrates how to compare token usage across different models
and tokenizers, which is essential for choosing the right model, understanding
tokenization differences, and optimizing costs.

Main concepts:
- Comparing tokenizers (CL100K, P50K, R50K)
- Analyzing tokenization differences
- Understanding model-specific token counts
- Making informed model selection decisions
"""

from kerb.tokenizer import count_tokens, batch_count_tokens, Tokenizer
from typing import List, Dict


def main():
    """Run multi-model comparison examples."""
    
    print("="*80)
    print("MULTI-MODEL COMPARISON EXAMPLE")
    print("="*80)
    
    # Example 1: Basic tokenizer comparison
    print("\n" + "-"*80)
    print("EXAMPLE 1: Comparing Different Tokenizers")
    print("-"*80)
    
    text = "The quick brown fox jumps over the lazy dog. This is a test of tokenization."
    
    tokenizers = [
        (Tokenizer.CL100K_BASE, "GPT-4, GPT-3.5-turbo"),
        (Tokenizer.P50K_BASE, "Code models (Codex)"),
        (Tokenizer.R50K_BASE, "GPT-3 (davinci, curie)"),
        (Tokenizer.CHAR_4, "Fast approximation (4 chars/token)"),
    ]
    
    print(f"Text: {text}")
    print(f"Length: {len(text)} characters\n")
    
    print(f"{'Tokenizer':<25} | {'Models':<30} | {'Tokens':>8} | {'Chars/Token':>12}")
    print("-" * 80)
    
    for tokenizer, models in tokenizers:
        tokens = count_tokens(text, tokenizer=tokenizer)
        ratio = len(text) / tokens if tokens > 0 else 0
        print(f"{tokenizer.value:<25} | {models:<30} | {tokens:>8} | {ratio:>12.2f}")
    
    # Example 2: Comparing across different text types
    print("\n" + "-"*80)
    print("EXAMPLE 2: Token Counts for Different Content Types")
    print("-"*80)
    
    content_samples = {
        "Natural Language": "Machine learning is transforming how we interact with technology.",
        "Code (Python)": "def calculate_sum(numbers: List[int]) -> int:\n    return sum(numbers)",
        "Code (JavaScript)": "const fetchData = async (url) => { const response = await fetch(url); return response.json(); }",
        "JSON": '{"user": "john_doe", "age": 30, "active": true, "tags": ["developer", "python"]}',
        "Numbers": "123456789 987654321 1111111111 2222222222 3333333333",
        "Technical Jargon": "The API endpoint utilizes OAuth2.0 authentication with JWT tokens for authorization.",
    }
    
    comparison_tokenizers = [
        Tokenizer.CL100K_BASE,
        Tokenizer.P50K_BASE,
        Tokenizer.CHAR_4,
    ]
    
    print(f"{'Content Type':<20} | {'Text Preview':<45} | CL100K | P50K | CHAR_4")
    print("-" * 110)
    
    for content_type, sample in content_samples.items():
        preview = sample[:42] + "..." if len(sample) > 45 else sample
        counts = [count_tokens(sample, tokenizer=tok) for tok in comparison_tokenizers]
        print(f"{content_type:<20} | {preview:<45} | {counts[0]:>6} | {counts[1]:>4} | {counts[2]:>6}")
    
    # Example 3: Analyzing tokenization efficiency
    print("\n" + "-"*80)
    print("EXAMPLE 3: Tokenization Efficiency Analysis")
    print("-"*80)
    
    test_texts = [
        "Simple English sentence with common words.",
        "Technical terminology: microservices, containerization, orchestration, deployment.",
        "Mixed123Numbers456And789Words012Combined345Together678.",
        "Code_with_underscores_and_camelCase_and_PascalCase_naming.",
        "Special!@#$%Characters^&*()Mixed-In=With+Text/Content.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}: {text}")
        print(f"Characters: {len(text)}")
        
        cl100k_tokens = count_tokens(text, tokenizer=Tokenizer.CL100K_BASE)
        p50k_tokens = count_tokens(text, tokenizer=Tokenizer.P50K_BASE)
        
        print(f"CL100K_BASE: {cl100k_tokens} tokens ({len(text)/cl100k_tokens:.2f} chars/token)")
        print(f"P50K_BASE:   {p50k_tokens} tokens ({len(text)/p50k_tokens:.2f} chars/token)")
        
        diff = abs(cl100k_tokens - p50k_tokens)
        if diff > 0:
            print(f"Difference: {diff} tokens ({diff/max(cl100k_tokens, p50k_tokens)*100:.1f}%)")
    
    # Example 4: Model selection based on token efficiency
    print("\n" + "-"*80)
    print("EXAMPLE 4: Model Selection Optimization")
    print("-"*80)
    
    # Simulate choosing between models for a specific use case
    use_case_text = """

# %%
# Setup and Imports
# -----------------
    Process this customer support ticket:
    
    Customer: Jane Smith
    Issue: Unable to access account after password reset
    Priority: High
    Account Type: Premium
    
    Previous attempts:
    - Reset password via email (failed)
    - Called support hotline (line busy)
    
    Resolution needed: Restore account access and investigate why password reset failed.
    """
    
    print("Use Case: Customer Support Ticket Processing")
    print(f"Text length: {len(use_case_text)} characters\n")
    
    model_configs = [
        {
            "name": "GPT-4",
            "tokenizer": Tokenizer.CL100K_BASE,
            "cost_per_1k_input": 0.03,
            "cost_per_1k_output": 0.06,
            "context_window": 8192,
        },
        {
            "name": "GPT-3.5-turbo",
            "tokenizer": Tokenizer.CL100K_BASE,
            "cost_per_1k_input": 0.0005,
            "cost_per_1k_output": 0.0015,
            "context_window": 4096,
        },
    ]
    
    # Assume 200 token response
    expected_output_tokens = 200
    
    print(f"{'Model':<15} | {'Input Tokens':>12} | {'Est. Cost':>12} | {'Context Used':>13}")
    print("-" * 60)
    
    for config in model_configs:
        input_tokens = count_tokens(use_case_text, tokenizer=config["tokenizer"])
        input_cost = (input_tokens / 1000) * config["cost_per_1k_input"]
        output_cost = (expected_output_tokens / 1000) * config["cost_per_1k_output"]
        total_cost = input_cost + output_cost
        context_usage = (input_tokens + expected_output_tokens) / config["context_window"] * 100
        
        print(f"{config['name']:<15} | {input_tokens:>12} | ${total_cost:>11.6f} | {context_usage:>12.1f}%")
    
    # Example 5: Batch comparison across models
    print("\n" + "-"*80)
    print("EXAMPLE 5: Batch Processing Comparison")
    print("-"*80)
    
    # Simulate a dataset
    dataset = [
        "How do I reset my password?",
        "What are your business hours?",
        "I need help with my billing.",
        "Can you explain your refund policy?",
        "My order hasn't arrived yet.",
    ] * 10  # 50 items total
    
    print(f"Dataset: {len(dataset)} support queries\n")
    
    tokenizers_to_compare = [
        ("CL100K_BASE (GPT-4/3.5)", Tokenizer.CL100K_BASE),
        ("P50K_BASE (Code models)", Tokenizer.P50K_BASE),
        ("CHAR_4 (Approximation)", Tokenizer.CHAR_4),
    ]
    
    for name, tokenizer in tokenizers_to_compare:
        counts = batch_count_tokens(dataset, tokenizer=tokenizer)
        total = sum(counts)
        avg = total / len(counts)
        min_tokens = min(counts)
        max_tokens = max(counts)
        
        print(f"{name}:")
        print(f"  Total tokens: {total:,}")
        print(f"  Average per query: {avg:.1f}")
        print(f"  Range: {min_tokens} - {max_tokens}")
        print()
    
    # Example 6: Real-world model migration analysis
    print("\n" + "-"*80)
    print("EXAMPLE 6: Model Migration Impact Analysis")
    print("-"*80)
    
    # Simulate migrating from one model to another
    current_workload = {
        "chat_messages": 10000,
        "avg_tokens_per_message": 75,
    }
    
    migration_scenarios = [
        {
            "from": "GPT-4",
            "to": "GPT-3.5-turbo",
            "from_tokenizer": Tokenizer.CL100K_BASE,
            "to_tokenizer": Tokenizer.CL100K_BASE,
            "from_cost": 0.03,
            "to_cost": 0.0005,
        },
    ]
    
    print("Migration Analysis:\n")
    
    for scenario in migration_scenarios:
        print(f"Migration: {scenario['from']} -> {scenario['to']}")
        
        # Since both use same tokenizer, token count stays same
        monthly_tokens = current_workload["chat_messages"] * current_workload["avg_tokens_per_message"]
        
        current_cost = (monthly_tokens / 1000) * scenario["from_cost"]
        new_cost = (monthly_tokens / 1000) * scenario["to_cost"]
        savings = current_cost - new_cost
        savings_percent = (savings / current_cost) * 100
        
        print(f"  Monthly tokens: {monthly_tokens:,}")
        print(f"  Current cost: ${current_cost:.2f}")
        print(f"  New cost: ${new_cost:.2f}")
        print(f"  Monthly savings: ${savings:.2f} ({savings_percent:.1f}%)")
        print(f"  Annual savings: ${savings * 12:.2f}")
    
    # Example 7: Tokenizer characteristic summary
    print("\n" + "-"*80)
    print("EXAMPLE 7: Tokenizer Characteristics Summary")
    print("-"*80)
    
    test_suite = {
        "English": "The quick brown fox jumps over the lazy dog.",
        "Numbers": "0123456789" * 5,
        "Code": "def func(x):\n    return x * 2",
        "Mixed": "User123 logged in at 2024-10-15T14:30:00",
        "Symbols": "!@#$%^&*()_+-=[]{}|;':,.<>?/",
    }
    
    tokenizers_summary = [
        Tokenizer.CL100K_BASE,
        Tokenizer.P50K_BASE,
        Tokenizer.R50K_BASE,
    ]
    
    print("\nTokenization Comparison Matrix:\n")
    
    # Header
    print(f"{'Content Type':<15}", end="")
    for tok in tokenizers_summary:
        print(f" | {tok.value[:12]:>12}", end="")
    print()
    print("-" * 65)
    
    # Data rows
    for content_type, text in test_suite.items():
        print(f"{content_type:<15}", end="")
        for tok in tokenizers_summary:
            tokens = count_tokens(text, tokenizer=tok)
            print(f" | {tokens:>12}", end="")
        print()
    
    print("\nKey Insights:")
    print("- CL100K_BASE: Most efficient for modern GPT models (GPT-4, GPT-3.5-turbo)")
    print("- P50K_BASE: Optimized for code, used in Codex models")
    print("- R50K_BASE: Older encoding for GPT-3 base models")
    print("- CHAR_4: Fast approximation when exact count isn't critical")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
