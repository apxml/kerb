"""Data Quality Analysis Example

This example demonstrates how to analyze and ensure quality of fine-tuning datasets.

Main concepts:
- Analyzing dataset statistics (token counts, distribution)
- Checking for data quality issues
- Detecting duplicates
- Analyzing length distribution
- Checking label distribution for classification
- Detecting personally identifiable information (PII)

Use case: Ensuring your fine-tuning dataset meets quality standards,
identifying potential issues before expensive training runs.
"""

from kerb.fine_tuning import (
    analyze_dataset,
    check_data_quality,
    TrainingExample,
    TrainingDataset,
    DatasetFormat,
)
from kerb.fine_tuning.quality import (
    detect_pii,
    check_length_distribution,
    detect_duplicates,
    check_label_distribution,
)


def create_sample_dataset_with_issues():
    """Create a dataset with various quality issues for demonstration."""
    examples = []
    
    # Good quality examples
    for i in range(5):
        examples.append(TrainingExample(
            messages=[
                {"role": "user", "content": f"What is machine learning concept {i}?"},
                {"role": "assistant", "content": f"Machine learning concept {i} is a fundamental technique used in AI systems for pattern recognition and prediction."}
            ],
            label="ml-basics",
            metadata={"quality": "good"}
        ))
    
    # Duplicate examples
    duplicate = TrainingExample(
        messages=[
            {"role": "user", "content": "What is deep learning?"},
            {"role": "assistant", "content": "Deep learning uses neural networks with multiple layers."}
        ],
        label="ml-basics"
    )
    examples.append(duplicate)
    examples.append(duplicate)  # Exact duplicate
    
    # Very short example
    examples.append(TrainingExample(
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hi"}
        ],
        label="ml-basics",
        metadata={"quality": "poor"}
    ))
    
    # Empty content example
    examples.append(TrainingExample(
        messages=[
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""}
        ],
        label="ml-basics"
    ))
    
    # Example with PII (for demonstration - should be removed in real datasets)
    examples.append(TrainingExample(
        messages=[
            {"role": "user", "content": "Contact me at john.doe@example.com or 555-123-4567"},
            {"role": "assistant", "content": "I'll make a note of that contact information."}
        ],
        label="contact-info",
        metadata={"has_pii": True}
    ))
    
    # Long example
    long_text = " ".join(["This is a very long sentence with many words"] * 20)
    examples.append(TrainingExample(
        messages=[
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": "That was a long question."}
        ],
        label="ml-basics"
    ))
    
    # Classification examples with imbalanced labels
    for i in range(10):
        examples.append(TrainingExample(
            prompt=f"Classify this text {i}",
            completion="positive",
            label="positive"
        ))
    
    for i in range(2):
        examples.append(TrainingExample(
            prompt=f"Classify negative {i}",
            completion="negative",
            label="negative"
        ))
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CHAT,
        metadata={"contains_issues": True}
    )


def main():
    """Run data quality analysis example."""
    
    print("="*80)
    print("DATA QUALITY ANALYSIS EXAMPLE")
    print("="*80)
    
    # Step 1: Create sample dataset with issues
    print("\nStep 1: Creating sample dataset with various quality issues")
    dataset = create_sample_dataset_with_issues()
    print(f"Created dataset: {len(dataset)} examples")
    
    # Step 2: Analyze dataset statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    stats = analyze_dataset(dataset)
    print(f"\nTotal examples: {stats.total_examples}")
    print(f"Total tokens: {stats.total_tokens}")
    print(f"Average tokens per example: {stats.avg_tokens_per_example:.2f}")
    print(f"Min tokens: {stats.min_tokens}")
    print(f"Max tokens: {stats.max_tokens}")
    
    if stats.avg_prompt_tokens:
        print(f"Average prompt tokens: {stats.avg_prompt_tokens:.2f}")
    if stats.avg_completion_tokens:
        print(f"Average completion tokens: {stats.avg_completion_tokens:.2f}")
    
    print(f"\nDuplicate count: {stats.duplicate_count}")
    
    if stats.label_distribution:
        print("\nLabel distribution:")
        for label, count in stats.label_distribution.items():
            print(f"  {label}: {count}")
    
    # Step 3: Check data quality issues
    print("\n" + "="*80)
    print("QUALITY CHECKS")
    print("="*80)
    
    quality_report = check_data_quality(dataset)
    print(f"\nTotal examples: {quality_report['total_examples']}")
    print(f"Empty examples: {quality_report['empty_examples']}")
    print(f"Very short examples: {quality_report['short_examples']}")
    print(f"Duplicate examples: {quality_report['duplicate_examples']}")
    print(f"Total issues found: {quality_report['total_issues']}")
    
    if quality_report['issues']:
        print("\nFirst few issues:")
        for issue in quality_report['issues'][:5]:
            print(f"  - {issue}")
    
    # Step 4: Detect duplicates
    print("\n" + "="*80)
    print("DUPLICATE DETECTION")
    print("="*80)
    
    duplicates = detect_duplicates(dataset)
    print(f"\nFound {len(duplicates)} duplicate pairs")
    
    if duplicates:
        print("\nDuplicate examples:")
        for idx1, idx2 in duplicates[:3]:
            print(f"  Example {idx1} duplicates Example {idx2}")
            example = dataset.examples[idx1]
            content = example.get_text_content()[:100]
            print(f"    Content: {content}...")
    
    # Step 5: Check length distribution
    print("\n" + "="*80)
    print("LENGTH DISTRIBUTION")
    print("="*80)
    
    length_stats = check_length_distribution(dataset)
    print(f"\nToken count statistics:")
    print(f"  Count: {length_stats['count']}")
    print(f"  Min: {length_stats['min']} tokens")
    print(f"  Max: {length_stats['max']} tokens")
    print(f"  Mean: {length_stats['mean']:.2f} tokens")
    print(f"  Median: {length_stats['median']} tokens")
    print(f"  25th percentile: {length_stats['p25']} tokens")
    print(f"  75th percentile: {length_stats['p75']} tokens")
    
    # Step 6: Check label distribution
    print("\n" + "="*80)
    print("LABEL DISTRIBUTION")
    print("="*80)
    
    label_stats = check_label_distribution(dataset)
    
    if "message" not in label_stats:
        print(f"\nTotal labeled examples: {label_stats['total_labeled']}")
        print(f"Unique labels: {label_stats['unique_labels']}")
        print(f"Is balanced: {label_stats['is_balanced']}")
        
        print("\nLabel counts:")
        for label, count in label_stats['label_counts'].items():
            percentage = label_stats['label_percentages'][label]
            print(f"  {label}: {count} ({percentage}%)")
        
        print("\nMost common labels:")
        for label, count in label_stats['most_common']:
            print(f"  {label}: {count}")
    else:
        print(f"\n{label_stats['message']}")
    
    # Step 7: Detect PII
    print("\n" + "="*80)
    print("PII DETECTION")
    print("="*80)
    
    pii_found = False
    for i, example in enumerate(dataset.examples):
        text = example.get_text_content()
        pii = detect_pii(text)
        
        if pii:
            if not pii_found:
                print("\nWARNING: Personally Identifiable Information detected!")
                pii_found = True
            
            print(f"\nExample {i}:")
            for pii_type, values in pii.items():
                print(f"  {pii_type}: {values}")
    
    if not pii_found:
        print("\nNo PII detected in dataset.")
    else:
        print("\nRecommendation: Remove or anonymize PII before fine-tuning!")
    
    # Step 8: Quality recommendations
    print("\n" + "="*80)
    print("QUALITY RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    if quality_report['empty_examples'] > 0:
        recommendations.append(f"Remove {quality_report['empty_examples']} empty examples")
    
    if quality_report['short_examples'] > 0:
        recommendations.append(f"Review {quality_report['short_examples']} very short examples")
    
    if quality_report['duplicate_examples'] > 0:
        recommendations.append(f"Remove {quality_report['duplicate_examples']} duplicate examples")
    
    if pii_found:
        recommendations.append("Remove or anonymize PII from examples")
    
    if not label_stats.get('is_balanced', True) and 'message' not in label_stats:
        recommendations.append("Consider balancing dataset (imbalanced labels detected)")
    
    if stats.total_examples < 50:
        recommendations.append("Dataset has fewer than 50 examples - consider adding more data")
    
    if length_stats['max'] > 1000:
        recommendations.append("Some examples are very long - consider truncating or splitting")
    
    if recommendations:
        print("\nRecommended actions:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("\nNo major issues detected! Dataset quality looks good.")
    
    # Step 9: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nDataset size: {len(dataset)} examples")
    print(f"Quality issues: {quality_report['total_issues']}")
    print(f"Duplicates: {quality_report['duplicate_examples']}")
    print(f"Average length: {length_stats['mean']:.2f} tokens")
    print(f"Label balance: {'Balanced' if label_stats.get('is_balanced', False) else 'Imbalanced'}")
    print(f"PII detected: {'Yes - needs attention!' if pii_found else 'No'}")
    print(f"\nRecommendations: {len(recommendations)} action items")


if __name__ == "__main__":
    main()
