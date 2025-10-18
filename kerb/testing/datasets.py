"""Dataset management for testing."""

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class TestDataset:
    """Dataset for evaluation testing."""

    def __init__(self, name: str, examples: Optional[List[Dict[str, Any]]] = None):
        """Initialize dataset.

        Args:
            name: Dataset name
            examples: List of examples
        """
        self.name = name
        self.examples = examples or []

    def add_example(
        self, input: str, output: str, metadata: Optional[Dict] = None
    ) -> None:
        """Add an example to the dataset."""
        self.examples.append(
            {"input": input, "output": output, "metadata": metadata or {}}
        )

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get example by index."""
        return self.examples[idx]

    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)

    def save(self, filepath: Path) -> None:
        """Save dataset to disk."""
        with open(filepath, "w") as f:
            json.dump({"name": self.name, "examples": self.examples}, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "TestDataset":
        """Load dataset from disk."""
        with open(filepath) as f:
            data = json.load(f)
        return cls(name=data["name"], examples=data["examples"])


def create_dataset(
    name: str, examples: List[Tuple[str, str]], metadata: Optional[List[Dict]] = None
) -> TestDataset:
    """Create a test dataset.

    Args:
        name: Dataset name
        examples: List of (input, output) tuples
        metadata: Optional metadata for each example

    Returns:
        TestDataset instance
    """
    dataset = TestDataset(name)

    for i, (input_text, output_text) in enumerate(examples):
        meta = metadata[i] if metadata and i < len(metadata) else {}
        dataset.add_example(input_text, output_text, meta)

    return dataset


def load_dataset(filepath: Path) -> TestDataset:
    """Load dataset from file.

    Args:
        filepath: Path to dataset file

    Returns:
        TestDataset instance
    """
    return TestDataset.load(filepath)


def split_dataset(
    dataset: TestDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[TestDataset, TestDataset, TestDataset]:
    """Split dataset into train/val/test sets.

    Args:
        dataset: Dataset to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    examples = dataset.examples.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_dataset = TestDataset(f"{dataset.name}_train", examples[:train_end])
    val_dataset = TestDataset(f"{dataset.name}_val", examples[train_end:val_end])
    test_dataset = TestDataset(f"{dataset.name}_test", examples[val_end:])

    return train_dataset, val_dataset, test_dataset


def augment_dataset(
    dataset: TestDataset,
    augmentation_fn: Callable[[str], List[str]],
    augmentation_factor: int = 2,
) -> TestDataset:
    """Augment dataset with variations.

    Args:
        dataset: Original dataset
        augmentation_fn: Function to generate augmented inputs
        augmentation_factor: Number of augmentations per example

    Returns:
        Augmented dataset
    """
    augmented = TestDataset(f"{dataset.name}_augmented")

    for example in dataset:
        # Add original
        augmented.add_example(
            example["input"], example["output"], example.get("metadata")
        )

        # Add augmentations
        augmented_inputs = augmentation_fn(example["input"])
        for aug_input in augmented_inputs[:augmentation_factor]:
            augmented.add_example(
                aug_input,
                example["output"],
                {**example.get("metadata", {}), "augmented": True},
            )

    return augmented
