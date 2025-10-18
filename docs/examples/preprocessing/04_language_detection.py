"""Language Detection for Multilingual Datasets

This example demonstrates language detection and filtering for building
multilingual LLM datasets or filtering to specific languages.

Main concepts:
- Detecting language with confidence scores
- Batch language detection for efficiency
- Filtering datasets by language
- Handling mixed-language content
- Language distribution analysis

Use case: Building language-specific or multilingual training datasets
"""

from kerb.preprocessing import (
    detect_language,
    detect_language_batch,
    is_language,
    filter_by_language,
    get_supported_languages,
    LanguageDetectionMode
)


def main():
    """Run language detection examples."""
    
    print("="*80)
    print("LANGUAGE DETECTION FOR MULTILINGUAL DATASETS")
    print("="*80)
    
    # Example 1: Basic language detection
    print("\n" + "-"*80)
    print("Example 1: Basic Language Detection")
    print("-"*80)
    
    sample_texts = [
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?",
        "Hola, como estas?",
        "Guten Tag, wie geht es Ihnen?",
        "Ciao, come stai?",
        "Привет, как дела?",
        "こんにちは、元気ですか?",
        "你好，你好吗？",
    ]
    
    print("\nDetecting languages:")
    for text in sample_texts:
        result = detect_language(text)
        print(f"'{text}'")
        print(f"  -> Language: {result.language}, Confidence: {result.confidence:.2f}")
    
    # Example 2: Detection modes
    print("\n" + "-"*80)
    print("Example 2: Detection Modes")
    print("-"*80)
    
    test_text = "This is an English sentence for testing different detection modes."
    
    modes = [
        LanguageDetectionMode.FAST,
        LanguageDetectionMode.ACCURATE,
        LanguageDetectionMode.SIMPLE,
    ]
    
    print(f"\nText: '{test_text}'")
    print("\nTesting different modes:")
    for mode in modes:
        result = detect_language(test_text, mode=mode)
        print(f"  {mode.value:10s}: {result.language} (confidence: {result.confidence:.2f})")
    
    # Example 3: Batch detection
    print("\n" + "-"*80)
    print("Example 3: Batch Language Detection")
    print("-"*80)
    
    multilingual_dataset = [
        "Machine learning is transforming industries.",
        "L'apprentissage automatique transforme les industries.",
        "El aprendizaje automático está transformando las industrias.",
        "Deep learning requires large datasets.",
        "L'apprentissage profond nécessite de grands ensembles de données.",
        "Maschinelles Lernen verändert Branchen.",
        "自然语言处理很重要。",
        "Natural language processing is important.",
    ]
    
    print(f"\nProcessing {len(multilingual_dataset)} texts:")
    results = detect_language_batch(multilingual_dataset)
    
    for text, result in zip(multilingual_dataset, results):
        print(f"\n'{text[:50]}...'")
        print(f"  -> {result.language} ({result.confidence:.2f})")
    
    # Count language distribution
    from collections import Counter
    language_counts = Counter(r.language for r in results)
    
    print("\nLanguage distribution:")
    for lang, count in language_counts.most_common():
        print(f"  {lang}: {count} samples")
    
    # Example 4: Filtering by language
    print("\n" + "-"*80)
    print("Example 4: Filtering by Language")
    print("-"*80)
    
    mixed_dataset = [
        "Python is a versatile programming language.",
        "JavaScript est utilisé pour le développement web.",
        "Java is widely used in enterprise applications.",
        "C++ est un langage de programmation performant.",
        "Go is efficient for concurrent programming.",
        "Ruby es conocido por su sintaxis elegante.",
        "Rust provides memory safety without garbage collection.",
    ]
    
    print(f"\nOriginal mixed dataset: {len(mixed_dataset)} samples")
    for i, text in enumerate(mixed_dataset, 1):
        result = detect_language(text)
        print(f"{i}. [{result.language}] {text}")
    
    # Filter for English only
    english_only = filter_by_language(mixed_dataset, language="en")
    
    print(f"\nEnglish-only filtered: {len(english_only)} samples")
    for text in english_only:
        print(f"  - {text}")
    
    # Filter for multiple languages (need to combine results)
    romance_languages = []
    for lang in ["en", "fr", "es"]:
        romance_languages.extend(filter_by_language(mixed_dataset, language=lang))
    # Remove duplicates while preserving order
    seen = set()
    romance_languages = [x for x in romance_languages if not (x in seen or seen.add(x))]
    
    print(f"\nEnglish/French/Spanish filtered: {len(romance_languages)} samples")
    for text in romance_languages:
        result = detect_language(text)
        print(f"  [{result.language}] {text}")
    
    # Example 5: Language checking
    print("\n" + "-"*80)
    print("Example 5: Language Verification")
    print("-"*80)
    
    test_samples = [
        ("Hello world", "en"),
        ("Bonjour le monde", "en"),
        ("Hola mundo", "es"),
        ("Guten Tag", "de"),
    ]
    
    print("\nChecking if texts match expected language:")
    for text, expected_lang in test_samples:
        is_match = is_language(text, expected_lang, threshold=0.5)
        actual = detect_language(text)
        status = "MATCH" if is_match else "NO MATCH"
        print(f"\nText: '{text}'")
        print(f"  Expected: {expected_lang}")
        print(f"  Detected: {actual.language} ({actual.confidence:.2f})")
        print(f"  Status: {status}")
    
    # Example 6: Building language-specific datasets
    print("\n" + "-"*80)
    print("Example 6: Building Language-Specific Datasets")
    print("-"*80)
    
    web_scraped_content = [
        "Artificial intelligence is advancing rapidly.",
        "Die künstliche Intelligenz entwickelt sich schnell.",
        "L'intelligence artificielle progresse rapidement.",
        "Deep learning models are getting more powerful.",
        "Los modelos de aprendizaje profundo son más potentes.",
        "Les modèles d'apprentissage profond deviennent plus puissants.",
        "Machine learning algorithms need quality data.",
        "Gli algoritmi di machine learning necessitano di dati di qualità.",
        "Algorithmen für maschinelles Lernen benötigen qualitativ hochwertige Daten.",
    ]
    
    print(f"\nWeb-scraped content: {len(web_scraped_content)} samples")
    
    # Analyze language distribution
    lang_results = detect_language_batch(web_scraped_content)
    lang_distribution = Counter(r.language for r in lang_results)
    
    print("\nLanguage distribution:")
    for lang, count in sorted(lang_distribution.items(), key=lambda x: -x[1]):
        percentage = (count / len(web_scraped_content)) * 100
        print(f"  {lang}: {count:2d} samples ({percentage:5.1f}%)")
    
    # Create language-specific datasets
    print("\nCreating language-specific training sets:")
    
    languages_to_extract = ["en", "fr", "de"]
    for lang in languages_to_extract:
        lang_dataset = filter_by_language(web_scraped_content, language=lang)
        print(f"\n{lang.upper()} dataset: {len(lang_dataset)} samples")
        for text in lang_dataset:
            print(f"  - {text}")
    
    # Example 7: High-confidence filtering
    print("\n" + "-"*80)
    print("Example 7: Confidence-Based Filtering")
    print("-"*80)
    
    ambiguous_dataset = [
        "OK",  # Ambiguous - could be any language
        "This is clearly English text with sufficient content.",
        "Bonjour",  # Short but French
        "La inteligencia artificial está cambiando el mundo.",
        "AI",  # Very short, ambiguous
        "Machine learning and deep learning are subsets of AI.",
    ]
    
    print("\nDetecting with confidence scores:")
    high_confidence_samples = []
    
    for text in ambiguous_dataset:
        result = detect_language(text)
        print(f"\nText: '{text}'")
        print(f"  Language: {result.language}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        # Only keep high-confidence detections
        if result.confidence >= 0.7:
            high_confidence_samples.append(text)
            print("  -> Kept (high confidence)")
        else:
            print("  -> Filtered (low confidence)")
    
    print(f"\nHigh-confidence dataset: {len(high_confidence_samples)} samples")
    for text in high_confidence_samples:
        print(f"  - {text}")
    
    # Example 8: Supported languages
    print("\n" + "-"*80)
    print("Example 8: Supported Languages")
    print("-"*80)
    
    supported = get_supported_languages()
    print(f"\nTotal supported languages: {len(supported)}")
    print("\nSample of supported languages:")
    for lang in sorted(supported)[:20]:  # Show first 20
        print(f"  - {lang}")
    print("  ...")
    
    print("\n" + "="*80)
    print("LANGUAGE DETECTION COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Use detect_language() for single text analysis")
    print("2. Use detect_language_batch() for efficiency with multiple texts")
    print("3. Filter datasets by language using filter_by_language()")
    print("4. Consider confidence scores for quality assurance")
    print("5. Different modes available: FAST, ACCURATE, SIMPLE")
    print("6. Essential for multilingual dataset curation")
    print("7. Helps ensure language consistency in training data")


if __name__ == "__main__":
    main()
