"""Simple test to verify fine_tuning module works correctly."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kerb.fine_tuning import (DatasetFormat, TrainingDataset, TrainingExample,
                              analyze_dataset, prepare_dataset, split_dataset,
                              to_openai_format, validate_dataset)


def test_basic_functionality():
    """Test basic fine-tuning functionality."""
    print("Testing fine_tuning module...")

    # Create sample data
    data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"},
                {
                    "role": "assistant",
                    "content": "Python is a high-level programming language.",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is JavaScript?"},
                {
                    "role": "assistant",
                    "content": "JavaScript is a programming language for web development.",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Java?"},
                {
                    "role": "assistant",
                    "content": "Java is an object-oriented programming language.",
                },
            ]
        },
    ]

    # Test 1: Prepare dataset
    print("\n1. Testing prepare_dataset...")
    dataset = prepare_dataset(
        data,
        format=DatasetFormat.CHAT,
        validate=True,
        deduplicate=True,
        shuffle=False,  # Keep order for testing
    )
    assert len(dataset) == 3, f"Expected 3 examples, got {len(dataset)}"
    print(f"   ✓ Prepared {len(dataset)} examples")

    # Test 2: Analyze dataset
    print("\n2. Testing analyze_dataset...")
    stats = analyze_dataset(dataset)
    assert stats.total_examples == 3, f"Expected 3 examples, got {stats.total_examples}"
    assert stats.total_tokens > 0, "Expected token count > 0"
    print(f"   ✓ Total tokens: {stats.total_tokens}")
    print(f"   ✓ Avg tokens: {stats.avg_tokens_per_example:.1f}")

    # Test 3: Validate dataset
    print("\n3. Testing validate_dataset...")
    result = validate_dataset(dataset)
    assert result.is_valid, f"Validation failed: {result.errors}"
    assert (
        result.valid_examples == 3
    ), f"Expected 3 valid examples, got {result.valid_examples}"
    print(f"   ✓ Dataset is valid ({result.valid_examples} examples)")

    # Test 4: Split dataset
    print("\n4. Testing split_dataset...")
    # Create larger dataset for splitting (need at least 50 examples for proper split)
    large_data = data * 20  # 60 examples
    large_dataset = prepare_dataset(
        large_data, format=DatasetFormat.CHAT, validate=False, deduplicate=False
    )
    train, val, test = split_dataset(large_dataset, seed=42)
    assert len(train) > 0, "Train set is empty"
    assert len(val) > 0, "Val set is empty"
    assert len(test) > 0, "Test set is empty"
    total_split = len(train) + len(val) + len(test)
    assert total_split == len(
        large_dataset
    ), f"Split sizes don't add up: {total_split} != {len(large_dataset)}"
    print(
        f"   ✓ Split {len(large_dataset)} examples -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
    )

    # Test 5: Convert to OpenAI format
    print("\n5. Testing to_openai_format...")
    openai_data = to_openai_format(dataset)
    assert len(openai_data) == 3, f"Expected 3 examples, got {len(openai_data)}"
    assert "messages" in openai_data[0], "Missing 'messages' key in OpenAI format"
    print(f"   ✓ Converted {len(openai_data)} examples to OpenAI format")

    # Test 6: TrainingExample
    print("\n6. Testing TrainingExample...")
    example = TrainingExample(
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
    )
    text = example.get_text_content()
    assert "Hello" in text and "Hi!" in text, "Text content extraction failed"
    hash1 = example.compute_hash()
    hash2 = example.compute_hash()
    assert hash1 == hash2, "Hash should be consistent"
    print(f"   ✓ TrainingExample works correctly")

    # Test 7: TrainingDataset
    print("\n7. Testing TrainingDataset...")
    custom_dataset = TrainingDataset(
        examples=[example], format=DatasetFormat.CHAT, metadata={"test": "data"}
    )
    assert len(custom_dataset) == 1, "Dataset length incorrect"
    assert custom_dataset[0] == example, "Dataset indexing failed"
    data_list = custom_dataset.to_list()
    assert isinstance(data_list, list), "to_list() should return a list"
    print(f"   ✓ TrainingDataset works correctly")

    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    test_basic_functionality()
