"""Production-Ready Preprocessing Pipelines

This example demonstrates how to build robust, reusable preprocessing pipelines
for production LLM training workflows.

Main concepts:
- Building modular preprocessing pipelines
- Combining multiple preprocessing operations
- Creating domain-specific pipelines
- Error handling and validation
- Pipeline composition and reusability
- Performance optimization

Use case: Production data preprocessing for LLM training at scale
"""

from kerb.preprocessing import (
    normalize_text,
    filter_by_length,
    filter_by_quality,
    deduplicate_exact,
    deduplicate_fuzzy,
    detect_language,
    filter_by_language,
    remove_urls,
    remove_emails,
    expand_contractions,
    standardize_numbers,
    preprocess_batch,
    preprocess_pipeline,
    NormalizationConfig,
    NormalizationLevel,
)


def main():
    """Run preprocessing pipeline examples."""
    
    print("="*80)
    print("PRODUCTION-READY PREPROCESSING PIPELINES")
    print("="*80)
    
    # Example 1: Basic pipeline composition
    print("\n" + "-"*80)
    print("Example 1: Basic Pipeline Composition")
    print("-"*80)
    
    # Create a simple pipeline
    basic_pipeline = preprocess_pipeline(
        str.strip,
        str.lower,
    )
    
    test_texts = [
        "  HELLO WORLD  ",
        "  Machine Learning  ",
        "  Deep Learning  ",
    ]
    
    print("\nApplying basic pipeline:")
    for text in test_texts:
        result = basic_pipeline(text)
        print(f"'{text}' -> '{result}'")
    
    # Example 2: Comprehensive cleaning pipeline
    print("\n" + "-"*80)
    print("Example 2: Comprehensive Cleaning Pipeline")
    print("-"*80)
    
    def create_cleaning_pipeline():
        """Create a comprehensive cleaning pipeline."""
        config = NormalizationConfig(
            level=NormalizationLevel.STANDARD,
            lowercase=True,
            remove_urls=True,
            remove_emails=True,
            remove_extra_spaces=True
        )
        
        def clean(text):
            # Step 1: Normalize
            text = normalize_text(text, config=config)
            # Step 2: Expand contractions
            text = expand_contractions(text)
            # Step 3: Standardize numbers
            text = standardize_numbers(text)
            return text
        
        return clean
    
    cleaning_pipeline = create_cleaning_pipeline()
    
    messy_data = [
        "  I'm visiting https://example.com with three friends!  ",
        "  Contact me@email.com for five opportunities  ",
        "  We've got twenty datasets. Check http://data.org  ",
    ]
    
    print("\nApplying cleaning pipeline:")
    for text in messy_data:
        cleaned = cleaning_pipeline(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned:  {cleaned}")
    
    # Example 3: Quality control pipeline
    print("\n" + "-"*80)
    print("Example 3: Quality Control Pipeline")
    print("-"*80)
    
    def quality_control_pipeline(texts, min_length=20, max_length=500):
        """Apply quality control filters."""
        # Stage 1: Length filtering
        stage1 = filter_by_length(texts, min_length=min_length, max_length=max_length, unit="chars")
        print(f"  After length filter: {len(stage1)}/{len(texts)}")
        
        # Stage 2: Quality filtering
        stage2 = filter_by_quality(stage1, min_score=0.5)
        print(f"  After quality filter: {len(stage2)}/{len(texts)}")
        
        # Stage 3: Deduplication
        stage3 = deduplicate_exact(stage2, keep_order=True)
        print(f"  After exact dedup: {len(stage3)}/{len(texts)}")
        
        stage4 = deduplicate_fuzzy(stage3, similarity_threshold=0.9, keep_order=True)
        print(f"  After fuzzy dedup: {len(stage4)}/{len(texts)}")
        
        return stage4
    
    raw_dataset = [
        "x",  # Too short
        "This is a good quality sample with sufficient content.",
        "Another quality sample with meaningful information.",
        "y",  # Too short
        "This is a good quality sample with sufficient content.",  # Duplicate
        "Yet another high-quality sample for the dataset.",
        "asdfjkl qwerty zxcvbn",  # Low quality
        "More quality content for training purposes here.",
    ]
    
    print(f"\nApplying quality control to {len(raw_dataset)} samples:")
    filtered_dataset = quality_control_pipeline(raw_dataset)
    
    print(f"\nFinal dataset ({len(filtered_dataset)} samples):")
    for i, text in enumerate(filtered_dataset, 1):
        print(f"{i}. {text}")
    
    # Example 4: Language-specific pipeline
    print("\n" + "-"*80)
    print("Example 4: Language-Specific Pipeline")
    print("-"*80)
    
    def create_language_pipeline(target_languages=None):
        """Create pipeline for language-specific datasets."""
        if target_languages is None:
            target_languages = ["en"]
        
        def process(texts):
            # Stage 1: Language detection and filtering
            # Filter for each target language
            filtered = []
            for lang in target_languages:
                filtered.extend(filter_by_language(texts, language=lang))
            # Remove duplicates while preserving order
            seen = set()
            filtered = [x for x in filtered if not (x in seen or seen.add(x))]
            print(f"  After language filter: {len(filtered)}/{len(texts)}")
            
            # Stage 2: Cleaning
            config = NormalizationConfig(
                level=NormalizationLevel.STANDARD,
                lowercase=False,  # Preserve case for non-English
                remove_urls=True,
                remove_emails=True
            )
            cleaned = [normalize_text(text, config=config) for text in filtered]
            
            # Stage 3: Deduplication
            deduplicated = deduplicate_exact(cleaned, keep_order=True)
            print(f"  After deduplication: {len(deduplicated)}/{len(texts)}")
            
            return deduplicated
        
        return process
    
    multilingual_data = [
        "Machine learning is transforming industries.",
        "L'apprentissage automatique transforme les industries.",
        "Deep learning requires large datasets.",
        "El aprendizaje profundo requiere grandes conjuntos de datos.",
        "Natural language processing is important.",
        "Machine learning is transforming industries.",  # Duplicate
    ]
    
    print(f"\nProcessing {len(multilingual_data)} multilingual samples:")
    
    # English-only pipeline
    english_pipeline = create_language_pipeline(target_languages=["en"])
    english_data = english_pipeline(multilingual_data)
    
    print(f"\nEnglish dataset ({len(english_data)} samples):")
    for text in english_data:
        print(f"  - {text}")
    
    # Example 5: Domain-specific pipeline (code filtering)
    print("\n" + "-"*80)
    print("Example 5: Domain-Specific Pipeline (No Code)")
    print("-"*80)
    
    from kerb.preprocessing import detect_code
    
    def create_text_only_pipeline():
        """Create pipeline that filters out code."""
        def process(texts):
            # Remove code
            text_only = [text for text in texts if not detect_code(text)]
            print(f"  After code filter: {len(text_only)}/{len(texts)}")
            
            # Clean and normalize
            config = NormalizationConfig(
                level=NormalizationLevel.STANDARD,
                lowercase=True,
                remove_urls=True,
                remove_emails=True
            )
            cleaned = [normalize_text(text, config=config) for text in text_only]
            
            # Quality filter
            quality = filter_by_quality(cleaned, min_score=0.5)
            print(f"  After quality filter: {len(quality)}/{len(texts)}")
            
            return quality
        
        return process
    
    mixed_content = [
        "Machine learning is a powerful technology.",
        "def train_model():\n    return model.fit(data)",
        "Natural language processing enables text understanding.",
        "import numpy as np\nimport pandas as pd",
        "Deep learning uses neural networks effectively.",
        "class Model:\n    def __init__(self):\n        pass",
    ]
    
    print(f"\nFiltering code from {len(mixed_content)} samples:")
    text_pipeline = create_text_only_pipeline()
    text_data = text_pipeline(mixed_content)
    
    print(f"\nText-only dataset ({len(text_data)} samples):")
    for text in text_data:
        print(f"  - {text}")
    
    # Example 6: Complete production pipeline
    print("\n" + "-"*80)
    print("Example 6: Complete Production Pipeline")
    print("-"*80)
    
    class ProductionPreprocessor:
        """Complete preprocessing pipeline for production."""
        
        def __init__(self, config=None):
            self.config = config or NormalizationConfig(
                level=NormalizationLevel.STANDARD,
                lowercase=True,
                remove_urls=True,
                remove_emails=True,
                remove_extra_spaces=True
            )
            self.stats = {}
        
        def process(self, texts, target_language="en"):
            """Process texts through complete pipeline."""
            print(f"\n  Starting with {len(texts)} samples")
            self.stats["original"] = len(texts)
            
            # Stage 1: Basic cleaning
            cleaned = []
            for text in texts:
                try:
                    clean = normalize_text(text, config=self.config)
                    clean = expand_contractions(clean)
                    clean = standardize_numbers(clean)
                    cleaned.append(clean)
                except Exception as e:
                    print(f"    Warning: Skipped sample due to error: {e}")
            
            print(f"  After cleaning: {len(cleaned)}")
            self.stats["cleaned"] = len(cleaned)
            
            # Stage 2: Length filtering
            length_filtered = filter_by_length(
                cleaned,
                min_length=20,
                max_length=1000,
                unit="chars"
            )
            print(f"  After length filter: {len(length_filtered)}")
            self.stats["length_filtered"] = len(length_filtered)
            
            # Stage 3: Language filtering
            if target_language:
                lang_filtered = filter_by_language(
                    length_filtered,
                    language=target_language
                )
                print(f"  After language filter: {len(lang_filtered)}")
                self.stats["lang_filtered"] = len(lang_filtered)
            else:
                lang_filtered = length_filtered
            
            # Stage 4: Code filtering
            text_only = [text for text in lang_filtered if not detect_code(text)]
            print(f"  After code filter: {len(text_only)}")
            self.stats["code_filtered"] = len(text_only)
            
            # Stage 5: Quality filtering
            quality = filter_by_quality(text_only, min_score=0.4)
            print(f"  After quality filter: {len(quality)}")
            self.stats["quality_filtered"] = len(quality)
            
            # Stage 6: Deduplication
            exact_dedup = deduplicate_exact(quality, keep_order=True)
            print(f"  After exact dedup: {len(exact_dedup)}")
            
            final = deduplicate_fuzzy(exact_dedup, similarity_threshold=0.9, keep_order=True)
            print(f"  Final dataset: {len(final)}")
            self.stats["final"] = len(final)
            
            return final
        
        def get_stats(self):
            """Return processing statistics."""
            return self.stats
    
    # Test production pipeline
    production_data = [
        "x",  # Too short
        "Machine learning enables intelligent systems to learn from data.",
        "Visit https://example.com for more information and contact us@email.com",
        "def process(): pass",  # Code
        "Natural language processing helps computers understand human language.",
        "Machine learning enables intelligent systems to learn from data.",  # Duplicate
        "asdfjkl qwerty",  # Low quality
        "Deep learning neural networks revolutionize computer vision tasks.",
        "L'intelligence artificielle est importante.",  # French
        "Artificial intelligence is transforming healthcare and medicine sectors.",
    ]
    
    print(f"\nProcessing production dataset:")
    preprocessor = ProductionPreprocessor()
    final_data = preprocessor.process(production_data, target_language="en")
    
    print(f"\nFinal production dataset ({len(final_data)} samples):")
    for i, text in enumerate(final_data, 1):
        print(f"{i}. {text}")
    
    print("\nProcessing statistics:")
    stats = preprocessor.get_stats()
    for stage, count in stats.items():
        print(f"  {stage:20s}: {count}")
    
    reduction = ((stats["original"] - stats["final"]) / stats["original"]) * 100
    print(f"\nTotal reduction: {reduction:.1f}%")
    
    # Example 7: Batch processing with pipeline
    print("\n" + "-"*80)
    print("Example 7: Efficient Batch Processing")
    print("-"*80)
    
    # Generate larger dataset
    batch_data = [
        f"Sample text number {i} with meaningful content for training."
        for i in range(50)
    ]
    # Add some variety
    batch_data.extend([
        "Visit https://example.com",
        "x",
        "def foo(): pass",
    ])
    
    print(f"\nProcessing batch of {len(batch_data)} samples:")
    
    # Process in chunks for memory efficiency
    chunk_size = 20
    all_results = []
    
    for i in range(0, len(batch_data), chunk_size):
        chunk = batch_data[i:i+chunk_size]
        chunk_preprocessor = ProductionPreprocessor()
        chunk_results = chunk_preprocessor.process(chunk, target_language="en")
        all_results.extend(chunk_results)
        print(f"  Chunk {i//chunk_size + 1}: {len(chunk)} -> {len(chunk_results)} samples")
    
    print(f"\nTotal processed: {len(all_results)} samples")
    print(f"First 3 results:")
    for i, text in enumerate(all_results[:3], 1):
        print(f"{i}. {text}")
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINES COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Build modular, reusable preprocessing pipelines")
    print("2. Combine multiple operations for comprehensive cleaning")
    print("3. Add error handling for robust production use")
    print("4. Track statistics for monitoring and optimization")
    print("5. Create domain-specific pipelines for different use cases")
    print("6. Process in chunks for memory efficiency")
    print("7. Standardize pipelines across training workflows")


if __name__ == "__main__":
    main()
