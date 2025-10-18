"""Dataset Splitting and Balancing Example

This example demonstrates how to split datasets and balance them for fine-tuning.

Main concepts:
- Splitting datasets into train/validation/test sets
- Different split strategies (random, stratified, hash-based)
- Balancing datasets by label distribution
- Sampling and filtering datasets
- Handling imbalanced data

Use case: Preparing a well-balanced training dataset with proper splits
for model evaluation and avoiding overfitting.
"""

from kerb.fine_tuning import (
    prepare_dataset,
    split_dataset,
    balance_dataset,
    sample_dataset,
    filter_dataset,
    TrainingExample,
    TrainingDataset,
    DatasetFormat,
    SplitStrategy,
)
from kerb.core.enums import BalanceMethod


def create_sample_dataset():
    """Create a sample dataset with labels."""
    examples = []
    
    # Category 1: Python basics (15 examples)
    python_topics = [
        ("What is a list?", "A list is an ordered collection: [1, 2, 3]"),
        ("What is a tuple?", "A tuple is immutable: (1, 2, 3)"),
        ("What is a set?", "A set is unordered and unique: {1, 2, 3}"),
        ("What is a dict?", "A dictionary maps keys to values: {'key': 'value'}"),
        ("How to use for loop?", "for i in range(10): print(i)"),
        ("How to use while loop?", "while condition: do_something()"),
        ("What is a function?", "def my_func(): return 42"),
        ("What is a class?", "class MyClass: pass"),
        ("How to handle exceptions?", "try: code() except Exception: handle()"),
        ("What are decorators?", "@decorator modifies function behavior"),
        ("What are generators?", "yield produces values lazily"),
        ("What is list comprehension?", "[x*2 for x in range(10)]"),
        ("What are lambda functions?", "lambda x: x + 1"),
        ("How to import modules?", "import module or from module import func"),
        ("What is __init__?", "Constructor method for classes"),
    ]
    
    for prompt, completion in python_topics:
        examples.append(TrainingExample(
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ],
            label="python-basics",
            metadata={"difficulty": "beginner", "category": "python"}
        ))
    
    # Category 2: Advanced Python (5 examples - imbalanced)
    advanced_topics = [
        ("What are metaclasses?", "Metaclasses define how classes are created"),
        ("What is async/await?", "async def for asynchronous programming"),
        ("What are context managers?", "with statement uses __enter__ and __exit__"),
        ("What is GIL?", "Global Interpreter Lock in CPython"),
        ("What are descriptors?", "Objects defining __get__, __set__, __delete__"),
    ]
    
    for prompt, completion in advanced_topics:
        examples.append(TrainingExample(
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ],
            label="python-advanced",
            metadata={"difficulty": "advanced", "category": "python"}
        ))
    
    # Category 3: Web development (8 examples)
    web_topics = [
        ("What is REST?", "RESTful APIs use HTTP methods"),
        ("What is JSON?", "JavaScript Object Notation for data"),
        ("What is HTTP?", "HyperText Transfer Protocol"),
        ("What are status codes?", "200 OK, 404 Not Found, 500 Error"),
        ("What is CORS?", "Cross-Origin Resource Sharing"),
        ("What is authentication?", "Verifying user identity"),
        ("What is OAuth?", "Open Authorization protocol"),
        ("What are cookies?", "Small data stored in browser"),
    ]
    
    for prompt, completion in web_topics:
        examples.append(TrainingExample(
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ],
            label="web-development",
            metadata={"difficulty": "intermediate", "category": "web"}
        ))
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CHAT,
        metadata={"source": "sample_data"}
    )


def main():
    """Run dataset splitting and balancing example."""
    
    print("="*80)
    print("DATASET SPLITTING AND BALANCING EXAMPLE")
    print("="*80)
    
    # Step 1: Create sample dataset
    print("\nStep 1: Creating sample dataset")
    dataset = create_sample_dataset()
    
    # Count by label
    label_counts = {}
    for example in dataset.examples:
        label = example.label or "unlabeled"
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Total examples: {len(dataset)}")
    print("Label distribution (imbalanced):")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Step 2: Random split
    print("\n" + "="*80)
    print("RANDOM SPLIT STRATEGY")
    print("="*80)
    
    train_ds, val_ds, test_ds = split_dataset(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        strategy=SplitStrategy.RANDOM,
        seed=42
    )
    
    print(f"\nTrain set: {len(train_ds)} examples")
    print(f"Validation set: {len(val_ds)} examples")
    print(f"Test set: {len(test_ds)} examples")
    
    # Step 3: Stratified split (maintains label distribution)
    print("\n" + "="*80)
    print("STRATIFIED SPLIT STRATEGY")
    print("="*80)
    
    train_ds_strat, val_ds_strat, test_ds_strat = split_dataset(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        strategy=SplitStrategy.STRATIFIED,
        seed=42
    )
    
    print(f"\nTrain set: {len(train_ds_strat)} examples")
    print(f"Validation set: {len(val_ds_strat)} examples")
    print(f"Test set: {len(test_ds_strat)} examples")
    
    # Check label distribution in stratified train set
    train_labels = {}
    for example in train_ds_strat.examples:
        label = example.label or "unlabeled"
        train_labels[label] = train_labels.get(label, 0) + 1
    
    print("\nTrain set label distribution (stratified):")
    for label, count in sorted(train_labels.items()):
        print(f"  {label}: {count}")
    
    # Step 4: Balance dataset
    print("\n" + "="*80)
    print("BALANCING DATASET")
    print("="*80)
    
    print("\nOriginal distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Undersample to minority class
    print("\nUndersampling to minority class:")
    balanced_under = balance_dataset(
        dataset,
        method=BalanceMethod.UNDERSAMPLE
    )
    
    under_labels = {}
    for example in balanced_under.examples:
        label = example.label or "unlabeled"
        under_labels[label] = under_labels.get(label, 0) + 1
    
    print(f"Total examples after undersampling: {len(balanced_under)}")
    for label, count in sorted(under_labels.items()):
        print(f"  {label}: {count}")
    
    # Oversample to majority class
    print("\nOversampling to majority class:")
    balanced_over = balance_dataset(
        dataset,
        method=BalanceMethod.OVERSAMPLE
    )
    
    over_labels = {}
    for example in balanced_over.examples:
        label = example.label or "unlabeled"
        over_labels[label] = over_labels.get(label, 0) + 1
    
    print(f"Total examples after oversampling: {len(balanced_over)}")
    for label, count in sorted(over_labels.items()):
        print(f"  {label}: {count}")
    
    # Step 5: Sampling
    print("\n" + "="*80)
    print("DATASET SAMPLING")
    print("="*80)
    
    sampled_ds = sample_dataset(dataset, n=10, seed=42)
    print(f"\nSampled {len(sampled_ds)} examples from {len(dataset)}")
    
    # Step 6: Filtering
    print("\n" + "="*80)
    print("DATASET FILTERING")
    print("="*80)
    
    # Filter only beginner difficulty
    filtered_ds = filter_dataset(
        dataset,
        filter_fn=lambda ex: ex.metadata.get("difficulty") == "beginner"
    )
    
    print(f"\nFiltered to beginner difficulty: {len(filtered_ds)} examples")
    
    # Filter by category
    python_only = filter_dataset(
        dataset,
        filter_fn=lambda ex: ex.metadata.get("category") == "python"
    )
    
    print(f"Filtered to Python category: {len(python_only)} examples")
    
    # Step 7: Complete workflow
    print("\n" + "="*80)
    print("COMPLETE WORKFLOW")
    print("="*80)
    
    print("\n1. Balance dataset")
    balanced = balance_dataset(dataset, method=BalanceMethod.UNDERSAMPLE)
    print(f"   Balanced: {len(balanced)} examples")
    
    print("\n2. Split into train/val/test")
    train, val, test = split_dataset(
        balanced,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        strategy=SplitStrategy.STRATIFIED,
        seed=42
    )
    print(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    print("\n3. Sample from train for quick testing")
    train_sample = sample_dataset(train, n=5, seed=42)
    print(f"   Train sample: {len(train_sample)} examples")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original dataset: {len(dataset)} examples (imbalanced)")
    print(f"Balanced dataset: {len(balanced)} examples")
    print(f"Final train set: {len(train)} examples")
    print(f"Final validation set: {len(val)} examples")
    print(f"Final test set: {len(test)} examples")
    print("\nDataset is ready for fine-tuning!")


if __name__ == "__main__":
    main()
