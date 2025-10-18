"""Batch Token Counting Example

This example demonstrates efficient batch processing of multiple texts
for token counting, which is essential for LLM applications that need
to process documents, analyze large datasets, or prepare training data.

Main concepts:
- Batch counting multiple texts efficiently
- Analyzing token distributions
- Identifying texts that exceed limits
- Processing large document collections
"""

from kerb.tokenizer import count_tokens, batch_count_tokens, Tokenizer
from typing import List, Dict, Tuple


def main():
    """Run batch token counting examples."""
    
    print("="*80)
    print("BATCH TOKEN COUNTING EXAMPLE")
    print("="*80)
    
    # Example 1: Basic batch counting
    print("\n" + "-"*80)
    print("EXAMPLE 1: Basic Batch Processing")
    print("-"*80)
    
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Transformers have revolutionized NLP tasks.",
        "Large language models can generate human-like text.",
    ]
    
    print(f"Processing {len(documents)} documents...")
    
    token_counts = batch_count_tokens(documents, tokenizer=Tokenizer.CL100K_BASE)
    
    print("\nResults:")
    for i, (doc, count) in enumerate(zip(documents, token_counts), 1):
        print(f"{i}. [{count:3d} tokens] {doc}")
    
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(token_counts)
    print(f"\nTotal tokens: {total_tokens}")
    print(f"Average tokens per document: {avg_tokens:.1f}")
    
    # Example 2: Filtering by token limit
    print("\n" + "-"*80)
    print("EXAMPLE 2: Filtering Documents by Token Limit")
    print("-"*80)
    
    # Simulate a dataset of various length documents
    dataset = [
        "Short text.",
        "This is a medium-length text that contains several sentences and covers a topic in moderate detail.",
        "A",
        "This is a longer document that goes into considerable detail about a subject. " * 10,
        "Medium example with some detail here.",
        "Another very long document that would exceed typical token limits. " * 15,
    ]
    
    token_limit = 100
    print(f"Token limit: {token_limit}")
    print(f"Total documents: {len(dataset)}\n")
    
    counts = batch_count_tokens(dataset, tokenizer=Tokenizer.CL100K_BASE)
    
    within_limit = []
    exceeds_limit = []
    
    for doc, count in zip(dataset, counts):
        if count <= token_limit:
            within_limit.append((doc, count))
        else:
            exceeds_limit.append((doc, count))
    
    print(f"Documents within limit: {len(within_limit)}")
    for doc, count in within_limit:
        preview = doc[:50] + "..." if len(doc) > 50 else doc
        print(f"  [{count:3d} tokens] {preview}")
    
    print(f"\nDocuments exceeding limit: {len(exceeds_limit)}")
    for doc, count in exceeds_limit:
        preview = doc[:50] + "..." if len(doc) > 50 else doc
        print(f"  [{count:3d} tokens] {preview}")
        print(f"    -> Exceeds by {count - token_limit} tokens")
    
    # Example 3: Token distribution analysis
    print("\n" + "-"*80)
    print("EXAMPLE 3: Token Distribution Analysis")
    print("-"*80)
    
    # Simulate processing a large collection
    large_dataset = [
        "Short snippet " * i for i in range(1, 21)
    ]
    
    counts = batch_count_tokens(large_dataset, tokenizer=Tokenizer.CL100K_BASE)
    
    # Calculate statistics
    min_tokens = min(counts)
    max_tokens = max(counts)
    avg_tokens = sum(counts) / len(counts)
    
    print(f"Documents analyzed: {len(large_dataset)}")
    print(f"Token statistics:")
    print(f"  Minimum: {min_tokens} tokens")
    print(f"  Maximum: {max_tokens} tokens")
    print(f"  Average: {avg_tokens:.1f} tokens")
    print(f"  Total: {sum(counts)} tokens")
    
    # Create histogram
    print("\nToken distribution (histogram):")
    bucket_size = max_tokens // 5 if max_tokens > 0 else 1
    buckets = {}
    for count in counts:
        bucket = (count // bucket_size) * bucket_size
        buckets[bucket] = buckets.get(bucket, 0) + 1
    
    for bucket in sorted(buckets.keys()):
        bar = "#" * buckets[bucket]
        print(f"  {bucket:3d}-{bucket+bucket_size:3d} tokens: {bar} ({buckets[bucket]})")
    
    # Example 4: Comparing tokenizers on batch data
    print("\n" + "-"*80)
    print("EXAMPLE 4: Comparing Tokenizers on Batch Data")
    print("-"*80)
    
    sample_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "import numpy as np\ndata = np.array([1, 2, 3])",
    ]
    
    tokenizers = [
        Tokenizer.CL100K_BASE,
        Tokenizer.P50K_BASE,
        Tokenizer.CHAR_4,
    ]
    
    print("Comparing tokenizers:\n")
    print(f"{'Text':<45} | {'CL100K':>8} | {'P50K':>8} | {'CHAR_4':>8}")
    print("-" * 80)
    
    for text in sample_texts:
        preview = text[:40] + "..." if len(text) > 40 else text
        counts = {tok: count_tokens(text, tokenizer=tok) for tok in tokenizers}
        print(f"{preview:<45} | {counts[Tokenizer.CL100K_BASE]:>8} | "
              f"{counts[Tokenizer.P50K_BASE]:>8} | {counts[Tokenizer.CHAR_4]:>8}")
    
    # Example 5: Real-world use case - preparing training data
    print("\n" + "-"*80)
    print("EXAMPLE 5: Training Data Preparation")
    print("-"*80)
    
    # Simulate preparing data for fine-tuning
    training_examples = [
        {"prompt": "What is Python?", "completion": "Python is a high-level programming language."},
        {"prompt": "Explain loops", "completion": "Loops are control structures that repeat code."},
        {"prompt": "What is a function?", "completion": "A function is a reusable block of code."},
        {"prompt": "Define variable", "completion": "A variable stores data values."},
        {"prompt": "What is an API?", "completion": "API stands for Application Programming Interface."},
    ]
    
    print(f"Analyzing {len(training_examples)} training examples...\n")
    
    max_tokens_per_example = 100
    valid_examples = []
    invalid_examples = []
    
    for i, example in enumerate(training_examples, 1):
        prompt_tokens = count_tokens(example["prompt"], tokenizer=Tokenizer.CL100K_BASE)
        completion_tokens = count_tokens(example["completion"], tokenizer=Tokenizer.CL100K_BASE)
        total_tokens = prompt_tokens + completion_tokens
        
        print(f"Example {i}:")
        print(f"  Prompt: {example['prompt']}")
        print(f"  Completion: {example['completion']}")
        print(f"  Tokens: {prompt_tokens} + {completion_tokens} = {total_tokens}")
        
        if total_tokens <= max_tokens_per_example:
            valid_examples.append(example)
            print(f"  Status: VALID")
        else:
            invalid_examples.append(example)
            print(f"  Status: EXCEEDS LIMIT by {total_tokens - max_tokens_per_example} tokens")
        print()
    
    print(f"Summary:")
    print(f"  Valid examples: {len(valid_examples)}/{len(training_examples)}")
    print(f"  Invalid examples: {len(invalid_examples)}/{len(training_examples)}")
    
    if valid_examples:
        total_valid_tokens = sum(
            count_tokens(ex["prompt"], tokenizer=Tokenizer.CL100K_BASE) +
            count_tokens(ex["completion"], tokenizer=Tokenizer.CL100K_BASE)
            for ex in valid_examples
        )
        print(f"  Total tokens in valid examples: {total_valid_tokens}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
