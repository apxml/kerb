"""Test Dataset Management Example

This example demonstrates how to create, manage, and use test datasets for
evaluating LLM applications.

Main concepts:
- Creating test datasets programmatically
- Loading and saving datasets
- Organizing examples with metadata
- Iterating over test cases
- Building evaluation datasets
- Managing benchmark datasets

Use cases for LLM developers:
- Creating evaluation benchmarks
- Building regression test suites
- Organizing test cases by category
- Sharing test data across team
- Systematic prompt testing
- Model comparison datasets
"""

from pathlib import Path
from kerb.testing import (
    TestDataset,
    create_dataset,
    load_dataset
)


def main():
    """Run test dataset management examples."""
    
    print("="*80)
    print("TEST DATASET MANAGEMENT EXAMPLE")
    print("="*80)
    
    # Example 1: Creating a basic dataset
    print("\n1. CREATING BASIC DATASET")
    print("-"*80)
    
    dataset = TestDataset(name="qa_basic")
    
    # Add examples
    dataset.add_example(
        input="What is Python?",
        output="Python is a high-level programming language.",
        metadata={"category": "programming", "difficulty": "easy"}
    )
    
    dataset.add_example(
        input="What is machine learning?",
        output="Machine learning is a subset of AI that learns from data.",
        metadata={"category": "ai", "difficulty": "medium"}
    )
    
    dataset.add_example(
        input="Explain neural networks",
        output="Neural networks are computational models inspired by biological neurons.",
        metadata={"category": "ai", "difficulty": "hard"}
    )
    
    print(f"Created dataset: {dataset.name}")
    print(f"Total examples: {len(dataset)}")
    
    # Example 2: Accessing dataset examples
    print("\n2. ACCESSING DATASET EXAMPLES")
    print("-"*80)
    
    # Access by index
    first_example = dataset[0]
    print(f"First example:")
    print(f"  Input: {first_example['input']}")
    print(f"  Output: {first_example['output'][:50]}...")
    print(f"  Metadata: {first_example['metadata']}")
    
    # Iterate over all examples
    print(f"\nAll examples:")
    for i, example in enumerate(dataset):
        category = example['metadata'].get('category', 'unknown')
        print(f"  {i+1}. [{category}] {example['input'][:40]}...")
    
    # Example 3: Creating dataset with convenience function
    print("\n3. CREATING DATASET WITH CONVENIENCE FUNCTION")
    print("-"*80)
    
    code_examples = [
        ("Write a function to calculate factorial", "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"),
        ("Write a function for Fibonacci", "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"),
        ("Write a function to check prime", "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))"),
    ]
    
    metadata_list = [
        {"language": "python", "task": "recursion"},
        {"language": "python", "task": "recursion"},
        {"language": "python", "task": "math"},
    ]
    
    code_dataset = create_dataset(
        name="code_generation",
        examples=code_examples,
        metadata=metadata_list
    )
    
    print(f"Created dataset: {code_dataset.name}")
    print(f"Examples: {len(code_dataset)}")
    
    # Example 4: Saving and loading datasets
    print("\n4. SAVING AND LOADING DATASETS")
    print("-"*80)
    
    # Create temp directory
    temp_dir = Path("temp_test_datasets")
    temp_dir.mkdir(exist_ok=True)
    
    # Save dataset
    save_path = temp_dir / "qa_basic.json"
    dataset.save(save_path)
    print(f"Saved dataset to: {save_path}")
    
    # Load dataset
    loaded_dataset = TestDataset.load(save_path)
    print(f"Loaded dataset: {loaded_dataset.name}")
    print(f"Examples in loaded dataset: {len(loaded_dataset)}")
    
    # Verify content
    assert len(loaded_dataset) == len(dataset)
    print("Verification: Original and loaded datasets match")
    
    # Example 5: Creating domain-specific datasets
    print("\n5. CREATING DOMAIN-SPECIFIC DATASETS")
    print("-"*80)
    
    # Translation dataset
    translation_dataset = TestDataset(name="translation_en_es")
    
    translations = [
        ("Hello", "Hola"),
        ("Good morning", "Buenos dias"),
        ("Thank you", "Gracias"),
        ("How are you?", "Como estas?"),
    ]
    
    for en, es in translations:
        translation_dataset.add_example(
            input=f"Translate to Spanish: {en}",
            output=es,
            metadata={"source_lang": "en", "target_lang": "es"}
        )
    
    print(f"Translation dataset: {len(translation_dataset)} pairs")
    
    # Sentiment classification dataset
    sentiment_dataset = TestDataset(name="sentiment_classification")
    
    sentiments = [
        ("I love this product!", "positive"),
        ("This is terrible", "negative"),
        ("It's okay", "neutral"),
        ("Amazing experience!", "positive"),
    ]
    
    for text, label in sentiments:
        sentiment_dataset.add_example(
            input=f"Classify sentiment: {text}",
            output=label,
            metadata={"task": "classification", "domain": "sentiment"}
        )
    
    print(f"Sentiment dataset: {len(sentiment_dataset)} examples")
    
    # Example 6: Filtering dataset by metadata
    print("\n6. FILTERING BY METADATA")
    print("-"*80)
    
    # Filter examples by category
    ai_examples = [
        ex for ex in dataset
        if ex['metadata'].get('category') == 'ai'
    ]
    
    print(f"AI-related examples: {len(ai_examples)}")
    for ex in ai_examples:
        print(f"  - {ex['input']}")
    
    # Filter by difficulty
    easy_examples = [
        ex for ex in dataset
        if ex['metadata'].get('difficulty') == 'easy'
    ]
    
    print(f"\nEasy examples: {len(easy_examples)}")
    for ex in easy_examples:
        print(f"  - {ex['input']}")
    
    # Example 7: Running tests with dataset
    print("\n7. RUNNING TESTS WITH DATASET")
    print("-"*80)
    
    def mock_llm_classify(prompt: str) -> str:
        """Mock sentiment classifier."""
        if "love" in prompt.lower() or "amazing" in prompt.lower():
            return "positive"
        elif "terrible" in prompt.lower():
            return "negative"
        else:
            return "neutral"
    
    def evaluate_on_dataset(dataset: TestDataset, model_func) -> dict:
        """Evaluate a model on a dataset."""
        correct = 0
        total = 0
        results = []
        
        for example in dataset:
            prediction = model_func(example['input'])
            expected = example['output']
            is_correct = prediction == expected
            
            correct += is_correct
            total += 1
            
            results.append({
                "input": example['input'],
                "expected": expected,
                "predicted": prediction,
                "correct": is_correct
            })
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "results": results
        }
    
    # Run evaluation
    print("Evaluating sentiment classifier...")
    eval_results = evaluate_on_dataset(sentiment_dataset, mock_llm_classify)
    
    print(f"\nResults:")
    print(f"  Accuracy: {eval_results['accuracy']:.2%}")
    print(f"  Correct: {eval_results['correct']}/{eval_results['total']}")
    
    print(f"\nDetailed results:")
    for r in eval_results['results']:
        status = "PASS" if r['correct'] else "FAIL"
        print(f"  [{status}] {r['input'][:40]}")
        print(f"       Expected: {r['expected']}, Predicted: {r['predicted']}")
    
    # Example 8: Creating benchmark datasets
    print("\n8. CREATING BENCHMARK DATASETS")
    print("-"*80)
    
    # Multi-task benchmark
    benchmark = TestDataset(name="multi_task_benchmark")
    
    # Add diverse tasks
    tasks = [
        # Math
        ("Calculate 15 * 23", "345", {"task_type": "math"}),
        ("What is 100 / 4?", "25", {"task_type": "math"}),
        
        # Reasoning
        ("If all birds can fly, and a penguin is a bird, can penguins fly?", 
         "No, the premise is incorrect. Not all birds can fly.", 
         {"task_type": "reasoning"}),
        
        # Code
        ("Write a function to reverse a string", 
         "def reverse(s): return s[::-1]",
         {"task_type": "code"}),
        
        # General knowledge
        ("What is the capital of France?", "Paris", {"task_type": "knowledge"}),
    ]
    
    for input_text, output_text, metadata in tasks:
        benchmark.add_example(input_text, output_text, metadata)
    
    print(f"Benchmark dataset: {len(benchmark)} examples")
    
    # Show task distribution
    task_types = {}
    for ex in benchmark:
        task = ex['metadata'].get('task_type', 'unknown')
        task_types[task] = task_types.get(task, 0) + 1
    
    print("\nTask distribution:")
    for task, count in task_types.items():
        print(f"  {task}: {count} examples")
    
    # Example 9: Dataset versioning
    print("\n9. DATASET VERSIONING")
    print("-"*80)
    
    # Create versioned datasets
    v1_dataset = TestDataset(name="qa_v1")
    v1_dataset.add_example(
        "What is AI?",
        "AI is artificial intelligence.",
        {"version": "1.0"}
    )
    
    v2_dataset = TestDataset(name="qa_v2")
    v2_dataset.add_example(
        "What is AI?",
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
        {"version": "2.0", "enhanced": True}
    )
    
    # Save versions
    v1_path = temp_dir / "qa_v1.json"
    v2_path = temp_dir / "qa_v2.json"
    
    v1_dataset.save(v1_path)
    v2_dataset.save(v2_path)
    
    print(f"Saved dataset versions:")
    print(f"  V1: {v1_path}")
    print(f"  V2: {v2_path}")
    
    # Example 10: Dataset statistics
    print("\n10. DATASET STATISTICS")
    print("-"*80)
    
    def analyze_dataset(dataset: TestDataset) -> dict:
        """Generate statistics for a dataset."""
        stats = {
            "total_examples": len(dataset),
            "avg_input_length": 0,
            "avg_output_length": 0,
            "categories": {}
        }
        
        total_input_len = 0
        total_output_len = 0
        
        for ex in dataset:
            total_input_len += len(ex['input'])
            total_output_len += len(ex['output'])
            
            # Count categories
            category = ex['metadata'].get('category', 'unknown')
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
        
        if len(dataset) > 0:
            stats["avg_input_length"] = total_input_len / len(dataset)
            stats["avg_output_length"] = total_output_len / len(dataset)
        
        return stats
    
    stats = analyze_dataset(dataset)
    
    print(f"Dataset statistics for '{dataset.name}':")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Avg input length: {stats['avg_input_length']:.1f} chars")
    print(f"  Avg output length: {stats['avg_output_length']:.1f} chars")
    print(f"  Categories: {stats['categories']}")
    
    # Cleanup
    print("\n11. CLEANUP")
    print("-"*80)
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print("Cleaned up test datasets directory")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("- Datasets organize test cases systematically")
    print("- Metadata enables filtering and categorization")
    print("- Save/load datasets for sharing and reuse")
    print("- Use datasets for evaluation and benchmarking")
    print("- Version datasets like code")
    print("- Analyze datasets to understand coverage")


if __name__ == "__main__":
    main()
