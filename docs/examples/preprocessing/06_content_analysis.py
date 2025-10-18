"""Content Analysis for Dataset Quality Assessment

This example demonstrates content analysis techniques to assess and improve
training data quality through various metrics and classifications.

Main concepts:
- Content type classification
- Quality metrics calculation
- Readability assessment
- Sentiment analysis
- Code detection
- Statistical analysis (word/sentence counts)

Use case: Analyzing and validating dataset quality before LLM training
"""

from kerb.preprocessing import (
    classify_content_type,
    detect_code,
    detect_sentiment,
    measure_readability,
    count_words,
    count_sentences,
    count_paragraphs,
    ContentType
)


def main():
    """Run content analysis examples."""
    
    print("="*80)
    print("CONTENT ANALYSIS FOR DATASET QUALITY ASSESSMENT")
    print("="*80)
    
    # Example 1: Content type classification
    print("\n" + "-"*80)
    print("Example 1: Content Type Classification")
    print("-"*80)
    
    diverse_samples = [
        "This is regular plain text content for training.",
        "def hello_world():\n    print('Hello, World!')\n    return True",
        '{"name": "John", "age": 30, "city": "New York"}',
        "<html><body><h1>Title</h1><p>Content</p></body></html>",
        "# Markdown Header\n\nThis is **bold** text with a [link](url).",
    ]
    
    print("\nClassifying content types:")
    for i, sample in enumerate(diverse_samples, 1):
        content_type = classify_content_type(sample)
        preview = sample.replace('\n', ' ')[:50]
        print(f"\n{i}. Type: {content_type.value}")
        print(f"   Content: {preview}...")
    
    # Example 2: Code detection
    print("\n" + "-"*80)
    print("Example 2: Code Detection")
    print("-"*80)
    
    mixed_content = [
        "Machine learning is a subset of artificial intelligence.",
        "def train_model(data):\n    return model.fit(data)",
        "The function process() takes two arguments.",
        "import numpy as np\nimport pandas as pd",
        "We use Python for data science projects.",
        "class NeuralNetwork:\n    def __init__(self):\n        pass",
    ]
    
    print("\nDetecting code in mixed content:")
    for text in mixed_content:
        is_code = detect_code(text)
        status = "CODE" if is_code else "TEXT"
        preview = text.replace('\n', ' ')[:50]
        print(f"\n[{status:4s}] {preview}...")
    
    # Separate code from text
    code_samples = [text for text in mixed_content if detect_code(text)]
    text_samples = [text for text in mixed_content if not detect_code(text)]
    
    print(f"\nSeparation results:")
    print(f"  Code samples: {len(code_samples)}")
    print(f"  Text samples: {len(text_samples)}")
    
    # Example 3: Sentiment analysis
    print("\n" + "-"*80)
    print("Example 3: Sentiment Analysis")
    print("-"*80)
    
    review_dataset = [
        "I absolutely love this product! It's amazing and wonderful!",
        "This is the worst experience I've ever had. Terrible service.",
        "The item is okay, nothing special.",
        "Great quality and excellent customer support. Very happy!",
        "Disappointed with the poor quality and bad performance.",
        "It works as expected. Average product.",
    ]
    
    print("\nAnalyzing sentiment:")
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    
    for text in review_dataset:
        sentiment = detect_sentiment(text)
        sentiment_counts[sentiment] += 1
        print(f"\n[{sentiment.upper():8s}] {text}")
    
    print(f"\nSentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(review_dataset)) * 100
        print(f"  {sentiment.capitalize():8s}: {count} ({percentage:.1f}%)")
    
    # Example 4: Readability assessment
    print("\n" + "-"*80)
    print("Example 4: Readability Assessment")
    print("-"*80)
    
    text_samples_readability = [
        "AI is great.",  # Very simple
        "Machine learning algorithms analyze data patterns.",  # Medium
        "The implementation of sophisticated neural network architectures necessitates comprehensive understanding of backpropagation mechanisms.",  # Complex
        "Deep learning is cool.",  # Simple
        "Natural language processing encompasses various computational techniques for analyzing textual information.",  # Medium-complex
    ]
    
    print("\nAssessing readability:")
    for text in text_samples_readability:
        score = measure_readability(text)
        if score > 0.7:
            level = "Easy"
        elif score > 0.4:
            level = "Medium"
        else:
            level = "Complex"
        
        print(f"\n[{level:7s}] Score: {score:.2f}")
        print(f"  {text}")
    
    # Example 5: Statistical analysis
    print("\n" + "-"*80)
    print("Example 5: Statistical Text Analysis")
    print("-"*80)
    
    analysis_samples = [
        "Short text.",
        "This is a medium-length sentence with several words.",
        "First sentence here. Second sentence follows. Third sentence concludes.",
        "Paragraph one has content.\n\nParagraph two has more content.\n\nParagraph three wraps up.",
    ]
    
    print("\nText statistics:")
    for i, text in enumerate(analysis_samples, 1):
        words = count_words(text)
        sentences = count_sentences(text)
        paragraphs = count_paragraphs(text)
        
        preview = text.replace('\n', ' ')[:40]
        print(f"\n{i}. {preview}...")
        print(f"   Words: {words}, Sentences: {sentences}, Paragraphs: {paragraphs}")
    
    # Example 6: Dataset quality metrics
    print("\n" + "-"*80)
    print("Example 6: Dataset Quality Metrics")
    print("-"*80)
    
    training_dataset = [
        "Machine learning enables computers to learn from data.",
        "def func(): pass",  # Code
        "x",  # Too short
        "Deep learning models require large amounts of training data.",
        "asdfjkl qwerty",  # Low quality
        "Natural language processing helps computers understand text.",
        "",  # Empty
        "AI is transforming industries worldwide.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    print(f"\nAnalyzing dataset quality ({len(training_dataset)} samples):")
    
    # Compute metrics
    metrics = {
        "total": len(training_dataset),
        "empty": 0,
        "too_short": 0,
        "code": 0,
        "low_readability": 0,
        "good_quality": 0,
    }
    
    good_samples = []
    
    for text in training_dataset:
        # Empty check
        if not text.strip():
            metrics["empty"] += 1
            continue
        
        # Length check
        if len(text) < 10:
            metrics["too_short"] += 1
            continue
        
        # Code check
        if detect_code(text):
            metrics["code"] += 1
            continue
        
        # Readability check
        readability = measure_readability(text)
        if readability < 0.3:
            metrics["low_readability"] += 1
            continue
        
        # Good quality
        metrics["good_quality"] += 1
        good_samples.append(text)
    
    print("\nQuality metrics:")
    print(f"  Total samples: {metrics['total']}")
    print(f"  Empty: {metrics['empty']}")
    print(f"  Too short: {metrics['too_short']}")
    print(f"  Code: {metrics['code']}")
    print(f"  Low readability: {metrics['low_readability']}")
    print(f"  Good quality: {metrics['good_quality']}")
    print(f"\nQuality rate: {metrics['good_quality'] / metrics['total'] * 100:.1f}%")
    
    print("\nGood quality samples:")
    for i, text in enumerate(good_samples, 1):
        print(f"{i}. {text}")
    
    # Example 7: Content distribution analysis
    print("\n" + "-"*80)
    print("Example 7: Content Distribution Analysis")
    print("-"*80)
    
    large_dataset = []
    
    # Generate diverse content
    text_templates = [
        "Machine learning is used in {}.",
        "Deep learning models can {}.",
        "Natural language processing helps {}.",
    ]
    
    code_templates = [
        "def {}():\n    pass",
        "class {}:\n    def __init__(self):\n        pass",
    ]
    
    applications = ["healthcare", "finance", "robotics", "education"]
    actions = ["classify images", "generate text", "translate languages"]
    helps = ["analyze sentiment", "extract entities", "summarize documents"]
    names = ["process", "analyze", "transform"]
    classes = ["Model", "Processor", "Analyzer"]
    
    for template in text_templates:
        if "{}" in template:
            items = applications if "used in" in template else actions if "can" in template else helps
            for item in items:
                large_dataset.append(template.format(item))
    
    for template in code_templates:
        items = names if "def" in template else classes
        for item in items:
            large_dataset.append(template.format(item))
    
    print(f"\nAnalyzing dataset ({len(large_dataset)} samples):")
    
    # Classify all content
    content_types = {}
    for text in large_dataset:
        ctype = classify_content_type(text)
        content_types[ctype] = content_types.get(ctype, 0) + 1
    
    print("\nContent type distribution:")
    for ctype, count in sorted(content_types.items(), key=lambda x: -x[1]):
        percentage = (count / len(large_dataset)) * 100
        print(f"  {ctype.value:12s}: {count:2d} ({percentage:5.1f}%)")
    
    # Word count statistics
    word_counts = [count_words(text) for text in large_dataset]
    avg_words = sum(word_counts) / len(word_counts)
    min_words = min(word_counts)
    max_words = max(word_counts)
    
    print(f"\nWord count statistics:")
    print(f"  Average: {avg_words:.1f} words")
    print(f"  Min: {min_words} words")
    print(f"  Max: {max_words} words")
    
    print("\n" + "="*80)
    print("CONTENT ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Classify content types to ensure dataset composition")
    print("2. Detect and filter code from text datasets")
    print("3. Analyze sentiment for balanced training data")
    print("4. Assess readability for appropriate complexity")
    print("5. Count statistics for dataset characterization")
    print("6. Calculate quality metrics to guide filtering")
    print("7. Analyze distribution to ensure dataset diversity")


if __name__ == "__main__":
    main()
