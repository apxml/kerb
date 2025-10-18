"""Dataset Preparation Example

This example demonstrates how to prepare training data for fine-tuning LLMs.

Main concepts:
- Creating training examples from raw data
- Preparing datasets with validation
- Deduplication and shuffling
- Format specification
- Basic dataset operations

Use case: Converting raw conversational data into a properly formatted
training dataset ready for fine-tuning.
"""

from kerb.fine_tuning import (
    prepare_dataset,
    TrainingExample,
    TrainingDataset,
    DatasetFormat,
    FineTuningProvider,
)


def main():
    """Run dataset preparation example."""
    
    print("="*80)
    print("DATASET PREPARATION EXAMPLE")
    print("="*80)
    
    # Step 1: Raw data (could come from logs, user interactions, etc.)
    print("\nStep 1: Creating raw training data")
    raw_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "How do I create a list in Python?"},
                {"role": "assistant", "content": "You can create a list in Python using square brackets: my_list = [1, 2, 3]"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "What is a dictionary?"},
                {"role": "assistant", "content": "A dictionary is a key-value data structure in Python: my_dict = {'key': 'value'}"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "How do I iterate over a list?"},
                {"role": "assistant", "content": "Use a for loop: for item in my_list: print(item)"}
            ]
        },
        # Intentional duplicate to demonstrate deduplication
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "How do I create a list in Python?"},
                {"role": "assistant", "content": "You can create a list in Python using square brackets: my_list = [1, 2, 3]"}
            ]
        },
    ]
    
    print(f"Raw data: {len(raw_data)} examples")
    
    # Step 2: Prepare dataset with automatic validation and deduplication
    print("\nStep 2: Preparing dataset with validation and deduplication")
    dataset = prepare_dataset(
        data=raw_data,
        format=DatasetFormat.CHAT,
        provider=FineTuningProvider.OPENAI,
        validate=True,
        deduplicate=True,
        shuffle=True
    )
    
    print(f"Prepared dataset: {len(dataset)} examples")
    print(f"Format: {dataset.format.value}")
    print(f"Provider: {dataset.provider.value if dataset.provider else 'Not specified'}")
    
    # Step 3: Create training examples manually for more control
    print("\nStep 3: Creating training examples manually")
    manual_examples = [
        TrainingExample(
            messages=[
                {"role": "user", "content": "Explain decorators"},
                {"role": "assistant", "content": "Decorators are functions that modify other functions in Python."}
            ],
            metadata={"category": "python-basics", "difficulty": "intermediate"}
        ),
        TrainingExample(
            messages=[
                {"role": "user", "content": "What are lambda functions?"},
                {"role": "assistant", "content": "Lambda functions are anonymous functions: lambda x: x + 1"}
            ],
            metadata={"category": "python-basics", "difficulty": "beginner"}
        ),
    ]
    
    manual_dataset = TrainingDataset(
        examples=manual_examples,
        format=DatasetFormat.CHAT,
        provider=FineTuningProvider.OPENAI,
        metadata={"task": "coding_assistant", "version": "1.0"}
    )
    
    print(f"Manual dataset: {len(manual_dataset)} examples")
    
    # Step 4: Combine datasets
    print("\nStep 4: Combining datasets")
    combined_examples = dataset.examples + manual_dataset.examples
    combined_dataset = TrainingDataset(
        examples=combined_examples,
        format=DatasetFormat.CHAT,
        provider=FineTuningProvider.OPENAI,
        metadata={"combined": True}
    )
    
    print(f"Combined dataset: {len(combined_dataset)} examples")
    
    # Step 5: Inspect examples
    print("\nStep 5: Inspecting examples")
    print("-"*80)
    for i, example in enumerate(combined_dataset.examples[:2]):
        print(f"\nExample {i+1}:")
        if example.messages:
            for msg in example.messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:60]
                print(f"  {role}: {content}...")
        if example.metadata:
            print(f"  Metadata: {example.metadata}")
    
    # Step 6: Create completion-format dataset
    print("\n" + "="*80)
    print("COMPLETION FORMAT DATASET")
    print("="*80)
    
    completion_data = [
        {
            "prompt": "Write a function to calculate factorial:",
            "completion": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        },
        {
            "prompt": "Write a function to reverse a string:",
            "completion": "def reverse_string(s):\n    return s[::-1]"
        },
    ]
    
    completion_dataset = prepare_dataset(
        data=completion_data,
        format=DatasetFormat.COMPLETION,
        validate=True,
        deduplicate=False,
        shuffle=False
    )
    
    print(f"\nCompletion dataset: {len(completion_dataset)} examples")
    print(f"Format: {completion_dataset.format.value}")
    
    # Display completion example
    print("\nExample completion format:")
    print("-"*80)
    example = completion_dataset.examples[0]
    print(f"Prompt: {example.prompt[:80]}...")
    print(f"Completion: {example.completion[:80]}...")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Chat format dataset: {len(combined_dataset)} examples")
    print(f"Completion format dataset: {len(completion_dataset)} examples")
    print(f"Total prepared examples: {len(combined_dataset) + len(completion_dataset)}")
    print("\nDataset preparation complete! Ready for fine-tuning.")


if __name__ == "__main__":
    main()
