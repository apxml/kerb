"""
Dataset Deduplication for Training Quality
==========================================

This example demonstrates deduplication techniques to remove redundant samples
from training datasets, improving model efficiency and reducing overfitting.

Main concepts:
- Exact deduplication for identical samples
- Fuzzy deduplication for near-duplicates
- Semantic deduplication for similar content
- Line and sentence deduplication
- Finding duplicate groups for analysis

Use case: Cleaning large web-scraped datasets before LLM training
"""

from kerb.preprocessing import (
    deduplicate_exact,
    deduplicate_fuzzy,
    deduplicate_semantic,
    deduplicate_lines,
    deduplicate_sentences,
    find_duplicates,
    compute_text_hash,
    DeduplicationMode
)


def main():
    """Run deduplication examples."""
    
    print("="*80)
    print("DATASET DEDUPLICATION FOR TRAINING QUALITY")
    print("="*80)
    
    # Example 1: Exact deduplication
    print("\n" + "-"*80)
    print("Example 1: Exact Deduplication")
    print("-"*80)
    
    # Simulated dataset with exact duplicates
    dataset_with_dupes = [
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "Machine learning is a subset of AI.",  # Exact duplicate
        "Natural language processing is important.",
        "Deep learning uses neural networks.",  # Exact duplicate
        "Transformers revolutionized NLP.",
        "Machine learning is a subset of AI.",  # Another duplicate
    ]
    
    print(f"\nOriginal dataset: {len(dataset_with_dupes)} samples")
    for i, text in enumerate(dataset_with_dupes, 1):
        print(f"{i}. {text}")
    
    # Remove exact duplicates
    deduplicated = deduplicate_exact(dataset_with_dupes, keep_order=True)
    
    print(f"\nAfter exact deduplication: {len(deduplicated)} samples")
    for i, text in enumerate(deduplicated, 1):
        print(f"{i}. {text}")
    
    print(f"\nRemoved {len(dataset_with_dupes) - len(deduplicated)} duplicates")
    
    # Example 2: Fuzzy deduplication
    print("\n" + "-"*80)
    print("Example 2: Fuzzy Deduplication (Near-Duplicates)")
    print("-"*80)
    
    # Dataset with near-duplicates (small variations)
    near_duplicates = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog",  # Missing period
        "the quick brown fox jumps over the lazy dog.",  # Different case
        "Natural language processing is fascinating.",
        "Natural language  processing  is  fascinating.",  # Extra spaces
        "Deep learning models require lots of data.",
        "Deep learning models require lots of data!",  # Different punctuation
    ]
    
    print(f"\nOriginal dataset: {len(near_duplicates)} samples")
    for i, text in enumerate(near_duplicates, 1):
        print(f"{i}. {text}")
    
    # Remove fuzzy duplicates with high similarity threshold
    fuzzy_deduplicated = deduplicate_fuzzy(
        near_duplicates,
        similarity_threshold=0.9,
        keep_order=True
    )
    
    print(f"\nAfter fuzzy deduplication (threshold=0.9): {len(fuzzy_deduplicated)} samples")
    for i, text in enumerate(fuzzy_deduplicated, 1):
        print(f"{i}. {text}")
    
    print(f"\nRemoved {len(near_duplicates) - len(fuzzy_deduplicated)} near-duplicates")
    
    # Example 3: Semantic deduplication
    print("\n" + "-"*80)
    print("Example 3: Semantic Deduplication")
    print("-"*80)
    
    # Dataset with semantically similar content
    semantic_duplicates = [
        "AI is transforming the world.",
        "Artificial intelligence is changing everything.",  # Similar meaning
        "Python is great for data science.",
        "ML is revolutionizing industries.",  # Similar to first
        "Data science benefits from Python.",  # Similar to third
        "Rust is a systems programming language.",
        "AI technology is advancing rapidly.",  # Similar to first two
    ]
    
    print(f"\nOriginal dataset: {len(semantic_duplicates)} samples")
    for i, text in enumerate(semantic_duplicates, 1):
        print(f"{i}. {text}")
    
    # Without embedding function, falls back to fuzzy matching
    semantic_deduplicated = deduplicate_semantic(
        semantic_duplicates,
        similarity_threshold=0.85
    )
    
    print(f"\nAfter semantic deduplication: {len(semantic_deduplicated)} samples")
    for i, text in enumerate(semantic_deduplicated, 1):
        print(f"{i}. {text}")
    
    print(f"\nRemoved {len(semantic_duplicates) - len(semantic_deduplicated)} semantically similar items")
    
    # Example 4: Line deduplication
    print("\n" + "-"*80)
    print("Example 4: Line Deduplication")
    print("-"*80)
    
    # Text with duplicate lines (common in logs, config files)
    text_with_dupe_lines = """import numpy as np

# %%
# Setup and Imports
# -----------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf"""
    
    print("\nOriginal text:")
    print(text_with_dupe_lines)
    
    # Remove duplicate lines
    dedupe_lines = deduplicate_lines(text_with_dupe_lines, keep_order=True)
    
    print("\nAfter line deduplication:")
    print(dedupe_lines)
    
    # Example 5: Sentence deduplication
    print("\n" + "-"*80)
    print("Example 5: Sentence Deduplication")
    print("-"*80)
    
    # Text with duplicate sentences
    text_with_dupe_sentences = (
        "Machine learning is powerful. "
        "It can solve complex problems. "
        "Machine learning is powerful. "
        "Deep learning is a subset of ML. "
        "It can solve complex problems."
    )
    
    print("\nOriginal text:")
    print(text_with_dupe_sentences)
    
    # Remove duplicate sentences
    dedupe_sentences = deduplicate_sentences(text_with_dupe_sentences, keep_order=True)
    
    print("\nAfter sentence deduplication:")
    print(dedupe_sentences)
    
    # Example 6: Finding duplicate groups
    print("\n" + "-"*80)
    print("Example 6: Finding Duplicate Groups for Analysis")
    print("-"*80)
    
    analysis_dataset = [
        "First unique text.",
        "Second unique text.",
        "First unique text.",  # Duplicate of index 0
        "Third unique text.",
        "Second unique text.",  # Duplicate of index 1
        "First unique text.",  # Another duplicate of index 0
        "Fourth unique text.",
    ]
    
    print(f"\nDataset for analysis: {len(analysis_dataset)} samples")
    for i, text in enumerate(analysis_dataset):
        print(f"{i}: {text}")
    
    # Find duplicate groups
    duplicate_groups = find_duplicates(analysis_dataset, mode=DeduplicationMode.EXACT)
    
    print(f"\nFound {len(duplicate_groups)} groups of duplicates:")
    for i, group in enumerate(duplicate_groups, 1):
        print(f"\nGroup {i} (indices {group}):")
        for idx in group:
            print(f"  [{idx}] {analysis_dataset[idx]}")
    
    # Example 7: Hash-based deduplication
    print("\n" + "-"*80)
    print("Example 7: Hash-Based Deduplication")
    print("-"*80)
    
    # Useful for very large datasets - hash for efficient comparison
    large_dataset = [
        "Sample text one.",
        "Sample text two.",
        "Sample text one.",
        "Sample text three.",
    ]
    
    print("\nComputing hashes for dataset:")
    hashes = {}
    for i, text in enumerate(large_dataset):
        text_hash = compute_text_hash(text)
        print(f"{i}: {text_hash[:16]}... - {text}")
        
        if text_hash in hashes:
            print(f"   -> Duplicate of index {hashes[text_hash]}")
        else:
            hashes[text_hash] = i
    
    # Example 8: Real-world scenario - web scraping deduplication
    print("\n" + "-"*80)
    print("Example 8: Multi-Stage Deduplication Pipeline")
    print("-"*80)
    
    web_scraped_data = [
        "Introduction to machine learning and AI technologies.",
        "Introduction to machine learning and AI technologies.",  # Exact
        "Introduction to machine learning and AI technologies",  # Fuzzy (no period)
        "Getting started with deep learning frameworks.",
        "A comprehensive guide to neural networks.",
        "Getting started with deep  learning frameworks.",  # Fuzzy (spacing)
        "Understanding transformers in NLP applications.",
        "A comprehensive guide to neural networks!",  # Fuzzy (punctuation)
        "Introduction to ML and AI technologies.",  # Semantic similar
        "Python programming for data science projects.",
        "Understanding transformers in NLP applications.",  # Exact
    ]
    
    print(f"\nOriginal web-scraped dataset: {len(web_scraped_data)} samples")
    for i, text in enumerate(web_scraped_data, 1):
        print(f"{i:2d}. {text}")
    
    # Stage 1: Exact deduplication
    stage1 = deduplicate_exact(web_scraped_data, keep_order=True)
    print(f"\nAfter exact deduplication: {len(stage1)} samples")
    
    # Stage 2: Fuzzy deduplication
    stage2 = deduplicate_fuzzy(stage1, similarity_threshold=0.9, keep_order=True)
    print(f"After fuzzy deduplication: {len(stage2)} samples")
    
    # Stage 3: Semantic deduplication
    final_dataset = deduplicate_semantic(stage2, similarity_threshold=0.85)
    print(f"After semantic deduplication: {len(final_dataset)} samples")
    
    print("\nFinal deduplicated dataset:")
    for i, text in enumerate(final_dataset, 1):
        print(f"{i}. {text}")
    
    print(f"\nTotal reduction: {len(web_scraped_data)} -> {len(final_dataset)} samples")
    print(f"Removed {len(web_scraped_data) - len(final_dataset)} duplicates ({(len(web_scraped_data) - len(final_dataset)) / len(web_scraped_data) * 100:.1f}%)")
    
    print("\n" + "="*80)
    print("DEDUPLICATION COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Exact deduplication removes identical copies")
    print("2. Fuzzy deduplication handles minor variations")
    print("3. Semantic deduplication removes similar meanings")
    print("4. Line/sentence deduplication useful for specific formats")
    print("5. Multi-stage pipelines provide comprehensive deduplication")
    print("6. Hash-based methods efficient for very large datasets")
    print("7. Deduplication crucial for training data quality")


if __name__ == "__main__":
    main()
