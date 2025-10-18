"""Dataset preparation and manipulation functions."""

import random
from typing import List, Union, Tuple, Optional, Callable, TYPE_CHECKING
from collections import defaultdict

from .types import (
    TrainingExample,
    TrainingDataset,
    DatasetFormat,
    FineTuningProvider,
    SplitStrategy,
    ValidationResult,
)

if TYPE_CHECKING:
    from kerb.core.enums import BalanceMethod


def prepare_dataset(
    data: Union[List[dict], TrainingDataset],
    format: DatasetFormat = DatasetFormat.CHAT,
    provider: Optional[FineTuningProvider] = None,
    validate: bool = True,
    deduplicate: bool = True,
    shuffle: bool = True
) -> TrainingDataset:
    """Prepare dataset for fine-tuning.
    
    Args:
        data: Raw data as list of dicts or TrainingDataset
        format: Dataset format
        provider: Target provider
        validate: Whether to validate dataset
        deduplicate: Whether to remove duplicates
        shuffle: Whether to shuffle examples
        
    Returns:
        TrainingDataset: Prepared dataset
        
    Examples:
        >>> data = [
        ...     {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]},
        ...     {"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}]}
        ... ]
        >>> dataset = prepare_dataset(data, format=DatasetFormat.CHAT)
    """
    # Import here to avoid circular dependency
    from .validation import validate_dataset as validate_dataset_fn
    
    # Convert to TrainingDataset if needed
    if isinstance(data, TrainingDataset):
        dataset = data
    else:
        examples = []
        for item in data:
            example = TrainingExample(
                messages=item.get("messages"),
                prompt=item.get("prompt"),
                completion=item.get("completion"),
                label=item.get("label"),
                metadata=item.get("metadata", {})
            )
            examples.append(example)
        dataset = TrainingDataset(examples=examples, format=format, provider=provider)
    
    # Deduplicate
    if deduplicate:
        dataset = deduplicate_dataset(dataset)
    
    # Shuffle
    if shuffle:
        dataset = shuffle_dataset(dataset)
    
    # Validate
    if validate:
        result = validate_dataset_fn(dataset)
        if not result.is_valid:
            raise ValueError(f"Dataset validation failed: {result.errors}")
    
    return dataset


def split_dataset(
    dataset: TrainingDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    strategy: SplitStrategy = SplitStrategy.RANDOM,
    seed: Optional[int] = None
) -> Tuple[TrainingDataset, TrainingDataset, TrainingDataset]:
    """Split dataset into train/validation/test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        strategy: Splitting strategy
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    if seed is not None:
        random.seed(seed)
    
    examples = dataset.examples.copy()
    
    if strategy == SplitStrategy.RANDOM:
        random.shuffle(examples)
    elif strategy == SplitStrategy.STRATIFIED:
        # Group by label
        label_groups = defaultdict(list)
        for ex in examples:
            label = ex.label or "unlabeled"
            label_groups[label].append(ex)
        
        # Split each group proportionally
        examples = []
        for label, group in label_groups.items():
            random.shuffle(group)
            examples.extend(group)
    elif strategy == SplitStrategy.HASH:
        # Deterministic split based on content hash
        examples.sort(key=lambda x: x.compute_hash())
    
    n_total = len(examples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Ensure at least 1 example in each split if the dataset is large enough
    if n_total >= 3:
        if n_val == 0 and val_ratio > 0:
            n_val = 1
        if n_train + n_val >= n_total:
            n_train = n_total - n_val - 1
    
    train_examples = examples[:n_train]
    val_examples = examples[n_train:n_train + n_val]
    test_examples = examples[n_train + n_val:]
    
    train_dataset = TrainingDataset(
        examples=train_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "split": "train"}
    )
    val_dataset = TrainingDataset(
        examples=val_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "split": "validation"}
    )
    test_dataset = TrainingDataset(
        examples=test_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "split": "test"}
    )
    
    return train_dataset, val_dataset, test_dataset


def balance_dataset(
    dataset: TrainingDataset,
    method: Union['BalanceMethod', str] = "undersample",
    target_count: Optional[int] = None
) -> TrainingDataset:
    """Balance dataset by label distribution.
    
    Args:
        dataset: Dataset to balance
        method: Balancing method (BalanceMethod enum or string: 'undersample', 'oversample', 'smote', 'none')
        target_count: Target count per label (if None, uses minority class for undersample or majority for oversample)
        
    Returns:
        TrainingDataset: Balanced dataset
        
    Examples:
        >>> # Using enum (recommended)
        >>> from kerb.core.enums import BalanceMethod
        >>> balanced = balance_dataset(dataset, method=BalanceMethod.UNDERSAMPLE)
        
        >>> # Using string (for backward compatibility)
        >>> balanced = balance_dataset(dataset, method="oversample")
    """
    from kerb.core.enums import BalanceMethod, validate_enum_or_string
    
    # Validate and normalize method
    method_val = validate_enum_or_string(method, BalanceMethod, "method")
    if isinstance(method_val, BalanceMethod):
        method_str = method_val.value
    else:
        method_str = method_val
    
    # If method is 'none', return dataset as-is
    if method_str == "none":
        return dataset
    
    # Group by label
    label_groups = defaultdict(list)
    for ex in dataset.examples:
        label = ex.label or "unlabeled"
        label_groups[label].append(ex)
    
    if not label_groups:
        return dataset
    
    # Determine target count
    if target_count is None:
        counts = [len(group) for group in label_groups.values()]
        if method_str == "undersample":
            target_count = min(counts)
        else:  # oversample or smote
            target_count = max(counts)
    
    balanced_examples = []
    for label, group in label_groups.items():
        if method_str == "undersample":
            # Randomly sample down to target
            if len(group) > target_count:
                sampled = random.sample(group, target_count)
            else:
                sampled = group
        elif method_str == "smote":
            # SMOTE-like oversampling (simplified version for text data)
            # For now, just duplicate with slight variations
            sampled = group.copy()
            while len(sampled) < target_count:
                # Add duplicates (in real SMOTE, you'd create synthetic examples)
                sampled.append(random.choice(group))
            if len(sampled) > target_count:
                sampled = sampled[:target_count]
        else:  # oversample
            # Randomly sample with replacement up to target
            sampled = random.choices(group, k=target_count)
        
        balanced_examples.extend(sampled)
    
    random.shuffle(balanced_examples)
    
    return TrainingDataset(
        examples=balanced_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "balanced": True}
    )


def augment_dataset(
    dataset: TrainingDataset,
    augmentation_fn: Callable[[TrainingExample], List[TrainingExample]],
    augment_ratio: float = 0.5
) -> TrainingDataset:
    """Augment dataset with variations.
    
    Args:
        dataset: Dataset to augment
        augmentation_fn: Function that takes example and returns list of augmented examples
        augment_ratio: Proportion of examples to augment
        
    Returns:
        TrainingDataset: Augmented dataset
    """
    augmented_examples = list(dataset.examples)
    
    n_to_augment = int(len(dataset.examples) * augment_ratio)
    examples_to_augment = random.sample(dataset.examples, n_to_augment)
    
    for example in examples_to_augment:
        augmented = augmentation_fn(example)
        augmented_examples.extend(augmented)
    
    return TrainingDataset(
        examples=augmented_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "augmented": True}
    )


def deduplicate_dataset(dataset: TrainingDataset, similarity_threshold: float = 1.0) -> TrainingDataset:
    """Remove duplicate examples from dataset.
    
    Args:
        dataset: Dataset to deduplicate
        similarity_threshold: Threshold for considering examples duplicates (1.0 = exact match)
        
    Returns:
        TrainingDataset: Deduplicated dataset
    """
    seen_hashes = set()
    unique_examples = []
    
    for example in dataset.examples:
        content_hash = example.compute_hash()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_examples.append(example)
    
    return TrainingDataset(
        examples=unique_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "deduplicated": True}
    )


def sample_dataset(dataset: TrainingDataset, n: int, seed: Optional[int] = None) -> TrainingDataset:
    """Sample subset of dataset.
    
    Args:
        dataset: Dataset to sample from
        n: Number of examples to sample
        seed: Random seed
        
    Returns:
        TrainingDataset: Sampled dataset
    """
    if seed is not None:
        random.seed(seed)
    
    if n >= len(dataset.examples):
        return dataset
    
    sampled_examples = random.sample(dataset.examples, n)
    
    return TrainingDataset(
        examples=sampled_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "sampled": n}
    )


def shuffle_dataset(dataset: TrainingDataset, seed: Optional[int] = None) -> TrainingDataset:
    """Shuffle dataset examples.
    
    Args:
        dataset: Dataset to shuffle
        seed: Random seed
        
    Returns:
        TrainingDataset: Shuffled dataset
    """
    if seed is not None:
        random.seed(seed)
    
    shuffled_examples = dataset.examples.copy()
    random.shuffle(shuffled_examples)
    
    return TrainingDataset(
        examples=shuffled_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "shuffled": True}
    )


def filter_dataset(
    dataset: TrainingDataset,
    filter_fn: Callable[[TrainingExample], bool]
) -> TrainingDataset:
    """Filter dataset by custom criteria.
    
    Args:
        dataset: Dataset to filter
        filter_fn: Function that returns True for examples to keep
        
    Returns:
        TrainingDataset: Filtered dataset
    """
    filtered_examples = [ex for ex in dataset.examples if filter_fn(ex)]
    
    return TrainingDataset(
        examples=filtered_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "filtered": True}
    )
