"""Batch Processing for Large-Scale Datasets

This example demonstrates efficient batch processing techniques for preprocessing
large datasets at scale, essential for production LLM training pipelines.

Main concepts:
- Batch processing for efficiency
- Creating reusable preprocessing pipelines
- Combining multiple operations
- Progress tracking for large datasets
- Memory-efficient processing

Use case: Processing millions of documents for LLM training
"""

from kerb.preprocessing import (
    preprocess_batch,
    preprocess_pipeline,
    normalize_text,
    filter_by_length,
    deduplicate_exact,
    remove_urls,
    remove_emails,
    NormalizationConfig,
    NormalizationLevel
)


def main():
    """Run batch processing examples."""
    
    print("="*80)
    print("BATCH PROCESSING FOR LARGE-SCALE DATASETS")
    print("="*80)
    
    # Example 1: Basic batch preprocessing
    print("\n" + "-"*80)
    print("Example 1: Basic Batch Preprocessing")
    print("-"*80)
    
    # Simulated raw dataset
    raw_batch = [
        "  HELLO WORLD  ",
        "  Machine Learning Is Amazing!  ",
        "  DEEP   LEARNING   ROCKS  ",
        "  Natural Language Processing  ",
        "  AI Is The Future  ",
    ]
    
    print(f"\nOriginal batch ({len(raw_batch)} samples):")
    for i, text in enumerate(raw_batch, 1):
        print(f"{i}. '{text}'")
    
    # Process batch with default normalization
    processed = preprocess_batch(raw_batch)
    
    print(f"\nProcessed batch:")
    for i, text in enumerate(processed, 1):
        print(f"{i}. '{text}'")
    
    # Example 2: Custom operation batch processing
    print("\n" + "-"*80)
    print("Example 2: Custom Operations Pipeline")
    print("-"*80)
    
    messy_batch = [
        "Check this: https://example.com and email me@email.com",
        "Another URL: http://test.org with contact: info@test.org",
        "Visit www.site.com or write to support@site.com",
    ]
    
    print(f"\nMessy batch ({len(messy_batch)} samples):")
    for text in messy_batch:
        print(f"  - {text}")
    
    # Define custom operations
    operations = [
        remove_urls,
        remove_emails,
        str.strip,
    ]
    
    # Process with custom operations
    cleaned_batch = preprocess_batch(messy_batch, operations=operations)
    
    print(f"\nCleaned batch:")
    for text in cleaned_batch:
        print(f"  - {text}")
    
    # Example 3: Pipeline composition
    print("\n" + "-"*80)
    print("Example 3: Reusable Pipeline Composition")
    print("-"*80)
    
    # Create a reusable pipeline
    cleanup_pipeline = preprocess_pipeline(
        remove_urls,
        remove_emails,
        str.lower,
        str.strip
    )
    
    test_texts = [
        "  CONTACT: admin@example.com  ",
        "  Visit HTTPS://WEBSITE.COM for more  ",
        "  Email us at SUPPORT@HELP.ORG  ",
    ]
    
    print("\nApplying reusable pipeline:")
    for text in test_texts:
        processed = cleanup_pipeline(text)
        print(f"\nOriginal:  '{text}'")
        print(f"Processed: '{processed}'")
    
    # Example 4: Large dataset simulation
    print("\n" + "-"*80)
    print("Example 4: Processing Large Datasets")
    print("-"*80)
    
    # Simulate a larger dataset
    large_dataset = []
    templates = [
        "Machine learning is transforming {industry}.",
        "Deep learning models are used in {application}.",
        "AI is revolutionizing {field}.",
        "{Technology} is essential for modern applications.",
    ]
    
    industries = ["healthcare", "finance", "education", "retail", "manufacturing"]
    applications = ["computer vision", "NLP", "robotics", "speech recognition"]
    fields = ["medicine", "transportation", "agriculture", "entertainment"]
    technologies = ["Neural networks", "Transformers", "CNNs", "RNNs"]
    
    # Generate synthetic dataset
    for template in templates:
        if "{industry}" in template:
            for industry in industries:
                large_dataset.append(template.format(industry=industry))
        elif "{application}" in template:
            for app in applications:
                large_dataset.append(template.format(application=app))
        elif "{field}" in template:
            for field in fields:
                large_dataset.append(template.format(field=field))
        elif "{Technology}" in template:
            for tech in technologies:
                large_dataset.append(template.format(Technology=tech))
    
    # Add some duplicates and noise
    large_dataset.extend(large_dataset[:5])  # Add duplicates
    large_dataset.append("x")  # Too short
    large_dataset.append("")  # Empty
    
    print(f"\nGenerated dataset: {len(large_dataset)} samples")
    print("\nFirst 5 samples:")
    for text in large_dataset[:5]:
        print(f"  - {text}")
    
    # Configure preprocessing
    config = NormalizationConfig(
        level=NormalizationLevel.STANDARD,
        lowercase=True,
        remove_urls=True,
        remove_emails=True
    )
    
    # Process batch
    print("\nProcessing with normalization config...")
    normalized_batch = [normalize_text(text, config=config) for text in large_dataset]
    
    # Filter by length
    print("Filtering by length...")
    filtered_batch = filter_by_length(normalized_batch, min_length=10, unit="chars")
    
    # Deduplicate
    print("Deduplicating...")
    final_batch = deduplicate_exact(filtered_batch, keep_order=True)
    
    print(f"\nPipeline results:")
    print(f"  Original: {len(large_dataset)} samples")
    print(f"  After normalization: {len(normalized_batch)} samples")
    print(f"  After length filter: {len(filtered_batch)} samples")
    print(f"  After deduplication: {len(final_batch)} samples")
    print(f"  Removed: {len(large_dataset) - len(final_batch)} samples ({(len(large_dataset) - len(final_batch)) / len(large_dataset) * 100:.1f}%)")
    
    # Example 5: Chunked processing for memory efficiency
    print("\n" + "-"*80)
    print("Example 5: Chunked Processing for Memory Efficiency")
    print("-"*80)
    
    # Simulate very large dataset
    very_large_dataset = [
        f"Sample text number {i} with some content." 
        for i in range(100)
    ]
    
    print(f"\nVery large dataset: {len(very_large_dataset)} samples")
    
    # Process in chunks to save memory
    chunk_size = 25
    processed_chunks = []
    
    print(f"Processing in chunks of {chunk_size}:")
    for i in range(0, len(very_large_dataset), chunk_size):
        chunk = very_large_dataset[i:i + chunk_size]
        processed_chunk = preprocess_batch(chunk)
        processed_chunks.extend(processed_chunk)
        print(f"  Processed chunk {i // chunk_size + 1}: {len(chunk)} samples")
    
    print(f"\nTotal processed: {len(processed_chunks)} samples")
    
    # Example 6: Multi-stage batch processing
    print("\n" + "-"*80)
    print("Example 6: Multi-Stage Batch Processing")
    print("-"*80)
    
    raw_web_data = [
        "  Check https://example.com for details! Contact: info@example.com  ",
        "  AMAZING OFFER!!! Visit http://promo.link NOW!!!  ",
        "  Machine learning tutorial: https://learn.ai  ",
        "  Email support@help.com for assistance  ",
        "  Deep learning basics explained  ",
        "  Visit site.com or email contact@site.com  ",
        "  AI fundamentals course  ",
    ]
    
    print(f"\nRaw web data ({len(raw_web_data)} samples):")
    for i, text in enumerate(raw_web_data, 1):
        print(f"{i}. {text[:60]}...")
    
    # Stage 1: Remove URLs and emails
    print("\nStage 1: Removing URLs and emails...")
    stage1_ops = [remove_urls, remove_emails]
    stage1 = preprocess_batch(raw_web_data, operations=stage1_ops)
    
    # Stage 2: Normalize text
    print("Stage 2: Normalizing text...")
    stage2_config = NormalizationConfig(
        level=NormalizationLevel.STANDARD,
        lowercase=True,
        remove_extra_spaces=True
    )
    stage2 = [normalize_text(text, config=stage2_config) for text in stage1]
    
    # Stage 3: Filter and deduplicate
    print("Stage 3: Filtering and deduplicating...")
    stage3 = filter_by_length(stage2, min_length=10, unit="chars")
    final_data = deduplicate_exact(stage3, keep_order=True)
    
    print(f"\nFinal processed data ({len(final_data)} samples):")
    for i, text in enumerate(final_data, 1):
        print(f"{i}. {text}")
    
    print(f"\nReduction: {len(raw_web_data)} -> {len(final_data)} samples")
    
    # Example 7: Performance comparison
    print("\n" + "-"*80)
    print("Example 7: Batch vs Individual Processing")
    print("-"*80)
    
    test_dataset = [f"Sample text {i}" for i in range(50)]
    
    print(f"\nDataset size: {len(test_dataset)} samples")
    
    # Individual processing
    print("\nIndividual processing:")
    individual_results = []
    for text in test_dataset:
        result = normalize_text(text, lowercase=True)
        individual_results.append(result)
    print(f"  Processed {len(individual_results)} samples individually")
    
    # Batch processing
    print("\nBatch processing:")
    batch_results = preprocess_batch(
        test_dataset,
        operations=[lambda x: normalize_text(x, lowercase=True)]
    )
    print(f"  Processed {len(batch_results)} samples in batch")
    
    # Results should be identical
    assert individual_results == batch_results
    print("\n  Results match: Both methods produce identical output")
    print("  Batch processing provides better organization and reusability")
    
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Use preprocess_batch() for efficient batch operations")
    print("2. Create reusable pipelines with preprocess_pipeline()")
    print("3. Process large datasets in chunks for memory efficiency")
    print("4. Combine multiple operations in multi-stage pipelines")
    print("5. Track progress when processing large datasets")
    print("6. Batch processing improves code organization")
    print("7. Essential for production-scale LLM training pipelines")


if __name__ == "__main__":
    main()
