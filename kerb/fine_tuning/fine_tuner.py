"""Fine-tuning implementation for LLM training data preparation and management."""

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List,
                    Optional, Tuple, Union)

if TYPE_CHECKING:
    from kerb.core.enums import BalanceMethod, Device


# ============================================================================
# Enums
# ============================================================================


class FineTuningProvider(Enum):
    """Supported fine-tuning providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    GENERIC = "generic"


class DatasetFormat(Enum):
    """Supported dataset formats."""

    CHAT = "chat"  # Chat format with messages
    COMPLETION = "completion"  # Prompt-completion format
    CLASSIFICATION = "classification"  # Classification tasks
    INSTRUCTION = "instruction"  # Instruction-following format


class SplitStrategy(Enum):
    """Dataset splitting strategies."""

    RANDOM = "random"
    STRATIFIED = "stratified"  # Maintain label distribution
    TEMPORAL = "temporal"  # Split by time/order
    HASH = "hash"  # Deterministic hash-based split


class ValidationLevel(Enum):
    """Validation strictness levels."""

    STRICT = "strict"  # Fail on any issues
    MODERATE = "moderate"  # Warn on minor issues
    LENIENT = "lenient"  # Only fail on critical issues


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TrainingExample:
    """Represents a single training example."""

    messages: Optional[List[Dict[str, str]]] = None  # For chat format
    prompt: Optional[str] = None  # For completion format
    completion: Optional[str] = None  # For completion format
    label: Optional[str] = None  # For classification
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        if self.messages is not None:
            result["messages"] = self.messages
        if self.prompt is not None:
            result["prompt"] = self.prompt
        if self.completion is not None:
            result["completion"] = self.completion
        if self.label is not None:
            result["label"] = self.label
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def get_text_content(self) -> str:
        """Extract all text content from the example."""
        texts = []
        if self.messages:
            for msg in self.messages:
                if "content" in msg:
                    texts.append(msg["content"])
        if self.prompt:
            texts.append(self.prompt)
        if self.completion:
            texts.append(self.completion)
        return " ".join(texts)

    def compute_hash(self) -> str:
        """Compute hash of example content for deduplication."""
        content = self.get_text_content()
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class TrainingDataset:
    """Represents a complete training dataset."""

    examples: List[TrainingExample]
    format: DatasetFormat
    provider: Optional[FineTuningProvider] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries."""
        return [ex.to_dict() for ex in self.examples]


@dataclass
class ValidationResult:
    """Results from dataset validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0

    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)


@dataclass
class DatasetStats:
    """Statistics about a dataset."""

    total_examples: int = 0
    total_tokens: int = 0
    avg_tokens_per_example: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    duplicate_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration for fine-tuning."""

    model: str
    n_epochs: int = 3
    batch_size: Optional[int] = None
    learning_rate_multiplier: Optional[float] = None
    prompt_loss_weight: float = 0.01
    validation_file: Optional[str] = None
    suffix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Dataset Preparation
# ============================================================================


def prepare_dataset(
    data: Union[List[Dict[str, Any]], TrainingDataset],
    format: DatasetFormat = DatasetFormat.CHAT,
    provider: Optional[FineTuningProvider] = None,
    validate: bool = True,
    deduplicate: bool = True,
    shuffle: bool = True,
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
                metadata=item.get("metadata", {}),
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
        result = validate_dataset(dataset)
        if not result.is_valid:
            raise ValueError(f"Dataset validation failed: {result.errors}")

    return dataset


def split_dataset(
    dataset: TrainingDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    strategy: SplitStrategy = SplitStrategy.RANDOM,
    seed: Optional[int] = None,
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
    val_examples = examples[n_train : n_train + n_val]
    test_examples = examples[n_train + n_val :]

    train_dataset = TrainingDataset(
        examples=train_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "split": "train"},
    )
    val_dataset = TrainingDataset(
        examples=val_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "split": "validation"},
    )
    test_dataset = TrainingDataset(
        examples=test_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "split": "test"},
    )

    return train_dataset, val_dataset, test_dataset


def balance_dataset(
    dataset: TrainingDataset,
    method: Union["BalanceMethod", str] = "undersample",
    target_count: Optional[int] = None,
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
        metadata={**dataset.metadata, "balanced": True},
    )


def augment_dataset(
    dataset: TrainingDataset,
    augmentation_fn: Callable[[TrainingExample], List[TrainingExample]],
    augment_ratio: float = 0.5,
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
        metadata={**dataset.metadata, "augmented": True},
    )


def deduplicate_dataset(
    dataset: TrainingDataset, similarity_threshold: float = 1.0
) -> TrainingDataset:
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
        metadata={**dataset.metadata, "deduplicated": True},
    )


def sample_dataset(
    dataset: TrainingDataset, n: int, seed: Optional[int] = None
) -> TrainingDataset:
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
        metadata={**dataset.metadata, "sampled": n},
    )


def shuffle_dataset(
    dataset: TrainingDataset, seed: Optional[int] = None
) -> TrainingDataset:
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
        metadata={**dataset.metadata, "shuffled": True},
    )


def filter_dataset(
    dataset: TrainingDataset, filter_fn: Callable[[TrainingExample], bool]
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
        metadata={**dataset.metadata, "filtered": True},
    )


# ============================================================================
# Format Conversion
# ============================================================================


def to_openai_format(dataset: TrainingDataset) -> List[Dict[str, Any]]:
    """Convert dataset to OpenAI fine-tuning format.

    OpenAI format: {"messages": [{"role": "system/user/assistant", "content": "..."}]}

    Args:
        dataset: Dataset to convert

    Returns:
        List of examples in OpenAI format
    """
    result = []

    for example in dataset.examples:
        if dataset.format == DatasetFormat.CHAT:
            if example.messages:
                result.append({"messages": example.messages})
        elif dataset.format == DatasetFormat.COMPLETION:
            # Convert to chat format
            messages = []
            if example.prompt:
                messages.append({"role": "user", "content": example.prompt})
            if example.completion:
                messages.append({"role": "assistant", "content": example.completion})
            result.append({"messages": messages})

    return result


def to_anthropic_format(dataset: TrainingDataset) -> List[Dict[str, Any]]:
    """Convert dataset to Anthropic fine-tuning format.

    Args:
        dataset: Dataset to convert

    Returns:
        List of examples in Anthropic format
    """
    result = []

    for example in dataset.examples:
        if dataset.format == DatasetFormat.CHAT and example.messages:
            # Anthropic uses similar format to OpenAI
            result.append({"messages": example.messages})
        elif dataset.format == DatasetFormat.COMPLETION:
            messages = []
            if example.prompt:
                messages.append({"role": "user", "content": example.prompt})
            if example.completion:
                messages.append({"role": "assistant", "content": example.completion})
            result.append({"messages": messages})

    return result


def to_google_format(dataset: TrainingDataset) -> List[Dict[str, Any]]:
    """Convert dataset to Google AI fine-tuning format.

    Args:
        dataset: Dataset to convert

    Returns:
        List of examples in Google format
    """
    result = []

    for example in dataset.examples:
        if dataset.format == DatasetFormat.CHAT and example.messages:
            # Google uses 'parts' instead of 'content'
            contents = []
            for msg in example.messages:
                contents.append(
                    {
                        "role": "user" if msg["role"] in ["user", "human"] else "model",
                        "parts": [{"text": msg.get("content", "")}],
                    }
                )
            result.append({"contents": contents})
        elif dataset.format == DatasetFormat.COMPLETION:
            contents = []
            if example.prompt:
                contents.append({"role": "user", "parts": [{"text": example.prompt}]})
            if example.completion:
                contents.append(
                    {"role": "model", "parts": [{"text": example.completion}]}
                )
            result.append({"contents": contents})

    return result


def to_huggingface_format(dataset: TrainingDataset) -> List[Dict[str, Any]]:
    """Convert dataset to HuggingFace format.

    Args:
        dataset: Dataset to convert

    Returns:
        List of examples in HuggingFace format
    """
    result = []

    for example in dataset.examples:
        if dataset.format == DatasetFormat.CHAT and example.messages:
            # HuggingFace often uses 'text' field
            text_parts = []
            for msg in example.messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                text_parts.append(f"{role}: {content}")
            result.append({"text": "\n".join(text_parts)})
        elif dataset.format == DatasetFormat.COMPLETION:
            if example.prompt and example.completion:
                result.append(
                    {"prompt": example.prompt, "completion": example.completion}
                )
        elif dataset.format == DatasetFormat.CLASSIFICATION:
            result.append(
                {
                    "text": example.prompt or example.get_text_content(),
                    "label": example.label,
                }
            )

    return result


def to_generic_format(dataset: TrainingDataset) -> List[Dict[str, Any]]:
    """Convert dataset to generic JSONL format.

    Args:
        dataset: Dataset to convert

    Returns:
        List of examples in generic format
    """
    return [ex.to_dict() for ex in dataset.examples]


def from_csv(
    filepath: str,
    prompt_column: str,
    completion_column: Optional[str] = None,
    label_column: Optional[str] = None,
    format: DatasetFormat = DatasetFormat.COMPLETION,
) -> TrainingDataset:
    """Convert CSV file to training dataset.

    Args:
        filepath: Path to CSV file
        prompt_column: Name of prompt column
        completion_column: Name of completion column
        label_column: Name of label column
        format: Target format

    Returns:
        TrainingDataset
    """
    examples = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get(prompt_column, "")
            completion = row.get(completion_column, "") if completion_column else None
            label = row.get(label_column) if label_column else None

            if format == DatasetFormat.CHAT and completion:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
                example = TrainingExample(messages=messages)
            else:
                example = TrainingExample(
                    prompt=prompt, completion=completion, label=label
                )

            examples.append(example)

    return TrainingDataset(examples=examples, format=format)


def from_json(
    filepath: str, format: DatasetFormat = DatasetFormat.CHAT
) -> TrainingDataset:
    """Convert JSON file to training dataset.

    Args:
        filepath: Path to JSON file
        format: Target format

    Returns:
        TrainingDataset
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    return prepare_dataset(data, format=format, validate=False)


def from_parquet(
    filepath: str, format: DatasetFormat = DatasetFormat.CHAT
) -> TrainingDataset:
    """Convert Parquet file to training dataset.

    Args:
        filepath: Path to Parquet file
        format: Target format

    Returns:
        TrainingDataset

    Note:
        Requires pandas and pyarrow packages
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for Parquet support. Install with: pip install pandas pyarrow"
        )

    df = pd.read_parquet(filepath)
    data = df.to_dict("records")

    return prepare_dataset(data, format=format, validate=False)


# ============================================================================
# JSONL Utilities
# ============================================================================


def write_jsonl(data: Union[List[Dict[str, Any]], TrainingDataset], filepath: str):
    """Write data to JSONL file.

    Args:
        data: Data to write (list of dicts or TrainingDataset)
        filepath: Output file path
    """
    if isinstance(data, TrainingDataset):
        data = data.to_list()

    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Read data from JSONL file.

    Args:
        filepath: Input file path

    Returns:
        List of dictionaries
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(data: Union[List[Dict[str, Any]], TrainingDataset], filepath: str):
    """Append data to JSONL file.

    Args:
        data: Data to append
        filepath: Target file path
    """
    if isinstance(data, TrainingDataset):
        data = data.to_list()

    with open(filepath, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def merge_jsonl(input_files: List[str], output_file: str):
    """Merge multiple JSONL files into one.

    Args:
        input_files: List of input file paths
        output_file: Output file path
    """
    with open(output_file, "w", encoding="utf-8") as outf:
        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as inf:
                for line in inf:
                    outf.write(line)


def validate_jsonl(filepath: str) -> ValidationResult:
    """Validate JSONL file format.

    Args:
        filepath: File path to validate

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    json.loads(line)
                    result.valid_examples += 1
                except json.JSONDecodeError as e:
                    result.add_error(f"Line {i}: Invalid JSON - {e}")
                    result.invalid_examples += 1

        result.total_examples = result.valid_examples + result.invalid_examples

    except FileNotFoundError:
        result.add_error(f"File not found: {filepath}")
    except Exception as e:
        result.add_error(f"Error reading file: {e}")

    return result


def count_jsonl_lines(filepath: str) -> int:
    """Count lines in JSONL file.

    Args:
        filepath: File path

    Returns:
        Number of lines
    """
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def stream_jsonl(
    filepath: str, batch_size: int = 1000
) -> Iterator[List[Dict[str, Any]]]:
    """Stream large JSONL files in batches.

    Args:
        filepath: File path
        batch_size: Number of examples per batch

    Yields:
        Batches of examples
    """
    batch = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                batch.append(json.loads(line))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch


# ============================================================================
# Validation
# ============================================================================


def validate_dataset(
    dataset: TrainingDataset,
    level: ValidationLevel = ValidationLevel.MODERATE,
    max_tokens: Optional[int] = None,
) -> ValidationResult:
    """Validate dataset for fine-tuning.

    Args:
        dataset: Dataset to validate
        level: Validation strictness
        max_tokens: Maximum tokens per example

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True, total_examples=len(dataset))

    if len(dataset.examples) == 0:
        result.add_error("Dataset is empty")
        return result

    # Validate each example
    for i, example in enumerate(dataset.examples):
        # Check format-specific requirements
        if dataset.format == DatasetFormat.CHAT:
            if not example.messages:
                result.add_error(f"Example {i}: Missing messages for chat format")
                result.invalid_examples += 1
                continue

            # Validate message structure
            for j, msg in enumerate(example.messages):
                if "role" not in msg:
                    result.add_error(f"Example {i}, Message {j}: Missing 'role' field")
                if "content" not in msg:
                    result.add_error(
                        f"Example {i}, Message {j}: Missing 'content' field"
                    )

        elif dataset.format == DatasetFormat.COMPLETION:
            if not example.prompt:
                result.add_error(f"Example {i}: Missing prompt for completion format")
                result.invalid_examples += 1
                continue
            if not example.completion:
                if level == ValidationLevel.STRICT:
                    result.add_error(f"Example {i}: Missing completion")
                else:
                    result.add_warning(f"Example {i}: Missing completion")

        # Check token limits if specified
        if max_tokens:
            text = example.get_text_content()
            estimated_tokens = len(text.split())  # Rough estimate
            if estimated_tokens > max_tokens:
                if level == ValidationLevel.STRICT:
                    result.add_error(
                        f"Example {i}: Exceeds token limit ({estimated_tokens} > {max_tokens})"
                    )
                else:
                    result.add_warning(
                        f"Example {i}: May exceed token limit ({estimated_tokens} tokens)"
                    )

        if result.errors and level == ValidationLevel.STRICT:
            result.valid_examples = i
            result.invalid_examples = len(dataset) - i
            return result

    result.valid_examples = len(dataset) - result.invalid_examples

    # Final checks
    if result.valid_examples < 10:
        result.add_warning(
            f"Dataset has only {result.valid_examples} valid examples. Recommended: at least 50-100"
        )

    return result


def validate_format(
    data: List[Dict[str, Any]], provider: FineTuningProvider
) -> ValidationResult:
    """Validate format for specific provider.

    Args:
        data: Data to validate
        provider: Target provider

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True, total_examples=len(data))

    for i, item in enumerate(data):
        if provider == FineTuningProvider.OPENAI:
            if "messages" not in item:
                result.add_error(
                    f"Example {i}: Missing 'messages' field for OpenAI format"
                )
            else:
                for j, msg in enumerate(item["messages"]):
                    if "role" not in msg or "content" not in msg:
                        result.add_error(
                            f"Example {i}, Message {j}: Must have 'role' and 'content'"
                        )

        elif provider == FineTuningProvider.ANTHROPIC:
            if "messages" not in item:
                result.add_error(
                    f"Example {i}: Missing 'messages' field for Anthropic format"
                )

        # Add more provider-specific validation as needed

    result.valid_examples = len(data) - len(result.errors)
    result.invalid_examples = len(result.errors)

    return result


def check_token_limits(
    dataset: TrainingDataset,
    max_tokens: int = 4096,
    tokenizer_name: str = "cl100k_base",
) -> Dict[str, Any]:
    """Check if examples exceed token limits.

    Args:
        dataset: Dataset to check
        max_tokens: Maximum allowed tokens
        tokenizer_name: Tokenizer to use for counting

    Returns:
        Dictionary with statistics about token usage
    """
    try:
        from ..tokenizer import count_tokens
    except ImportError:
        # Fallback to simple word count
        def count_tokens(text, model):
            return len(text.split())

    exceeding = []
    token_counts = []

    for i, example in enumerate(dataset.examples):
        text = example.get_text_content()
        tokens = count_tokens(text, tokenizer_name)
        token_counts.append(tokens)

        if tokens > max_tokens:
            exceeding.append({"index": i, "tokens": tokens})

    return {
        "total_examples": len(dataset),
        "exceeding_limit": len(exceeding),
        "exceeding_examples": exceeding,
        "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
        "max_tokens_found": max(token_counts) if token_counts else 0,
        "min_tokens_found": min(token_counts) if token_counts else 0,
    }


def validate_messages(messages: List[Dict[str, str]]) -> ValidationResult:
    """Validate message structure for chat format.

    Args:
        messages: List of message dictionaries

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)

    valid_roles = {"system", "user", "assistant", "function", "tool"}

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            result.add_error(f"Message {i}: Must be a dictionary")
            continue

        if "role" not in msg:
            result.add_error(f"Message {i}: Missing 'role' field")
        elif msg["role"] not in valid_roles:
            result.add_warning(
                f"Message {i}: Unusual role '{msg['role']}'. Expected: {valid_roles}"
            )

        if "content" not in msg:
            result.add_error(f"Message {i}: Missing 'content' field")

    return result


def estimate_training_tokens(dataset: TrainingDataset) -> int:
    """Estimate total training tokens.

    Args:
        dataset: Dataset to analyze

    Returns:
        Estimated total tokens
    """
    total = 0
    for example in dataset.examples:
        text = example.get_text_content()
        # Rough estimate: 1 token ≈ 4 characters
        total += len(text) // 4

    return total


def estimate_cost(
    dataset: TrainingDataset, model: str = "gpt-3.5-turbo", n_epochs: int = 3
) -> Dict[str, float]:
    """Estimate fine-tuning cost.

    Args:
        dataset: Dataset to train on
        model: Base model name
        n_epochs: Number of training epochs

    Returns:
        Dictionary with cost estimates
    """
    # OpenAI pricing (as of 2024)
    pricing = {
        "gpt-3.5-turbo": {
            "training": 0.008,
            "input": 0.003,
            "output": 0.006,
        },  # per 1K tokens
        "gpt-4": {"training": 0.03, "input": 0.03, "output": 0.06},
    }

    base_model = (
        model.split("-")[0] + "-" + model.split("-")[1] if "-" in model else model
    )
    rates = pricing.get(base_model, pricing["gpt-3.5-turbo"])

    total_tokens = estimate_training_tokens(dataset)
    training_tokens = total_tokens * n_epochs

    training_cost = (training_tokens / 1000) * rates["training"]

    return {
        "total_training_tokens": training_tokens,
        "estimated_training_cost_usd": round(training_cost, 2),
        "cost_per_epoch_usd": round(training_cost / n_epochs, 2),
        "model": model,
        "n_epochs": n_epochs,
    }


def validate_completion_format(prompt: str, completion: str) -> ValidationResult:
    """Validate completion-based format.

    Args:
        prompt: Prompt text
        completion: Completion text

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)

    if not prompt or not prompt.strip():
        result.add_error("Prompt is empty")

    if not completion or not completion.strip():
        result.add_error("Completion is empty")

    # Check for common issues
    if completion.startswith(prompt):
        result.add_warning("Completion appears to include the prompt")

    return result


def validate_chat_format(messages: List[Dict[str, str]]) -> ValidationResult:
    """Validate chat-based format.

    Args:
        messages: List of message dictionaries

    Returns:
        ValidationResult
    """
    return validate_messages(messages)


# ============================================================================
# Data Quality
# ============================================================================


def analyze_dataset(dataset: TrainingDataset) -> DatasetStats:
    """Analyze dataset statistics.

    Args:
        dataset: Dataset to analyze

    Returns:
        DatasetStats with comprehensive statistics
    """
    stats = DatasetStats()
    stats.total_examples = len(dataset)

    token_counts = []
    prompt_tokens = []
    completion_tokens = []
    labels = []

    for example in dataset.examples:
        text = example.get_text_content()
        tokens = len(text.split())  # Rough estimate
        token_counts.append(tokens)

        if example.prompt:
            prompt_tokens.append(len(example.prompt.split()))
        if example.completion:
            completion_tokens.append(len(example.completion.split()))
        if example.label:
            labels.append(example.label)

    if token_counts:
        stats.total_tokens = sum(token_counts)
        stats.avg_tokens_per_example = stats.total_tokens / len(token_counts)
        stats.min_tokens = min(token_counts)
        stats.max_tokens = max(token_counts)

    if prompt_tokens:
        stats.avg_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens)

    if completion_tokens:
        stats.avg_completion_tokens = sum(completion_tokens) / len(completion_tokens)

    if labels:
        stats.label_distribution = dict(Counter(labels))

    # Check for duplicates
    hashes = [ex.compute_hash() for ex in dataset.examples]
    stats.duplicate_count = len(hashes) - len(set(hashes))

    return stats


def check_data_quality(dataset: TrainingDataset) -> Dict[str, Any]:
    """Check dataset for quality issues.

    Args:
        dataset: Dataset to check

    Returns:
        Dictionary with quality metrics and issues
    """
    issues = []

    # Check for empty content
    empty_count = 0
    for i, example in enumerate(dataset.examples):
        text = example.get_text_content().strip()
        if not text:
            empty_count += 1
            issues.append(f"Example {i}: Empty content")

    # Check for very short examples
    short_count = 0
    for i, example in enumerate(dataset.examples):
        text = example.get_text_content()
        if len(text) < 10:
            short_count += 1
            issues.append(f"Example {i}: Very short content ({len(text)} chars)")

    # Check for duplicates
    stats = analyze_dataset(dataset)

    return {
        "total_examples": len(dataset),
        "empty_examples": empty_count,
        "short_examples": short_count,
        "duplicate_examples": stats.duplicate_count,
        "issues": issues[:100],  # Limit to first 100 issues
        "total_issues": len(issues),
    }


def detect_pii(text: str) -> Dict[str, List[str]]:
    """Detect personally identifiable information in text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with detected PII types and examples
    """
    pii = {
        "emails": [],
        "phone_numbers": [],
        "ssn": [],
        "credit_cards": [],
    }

    # Email pattern
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    pii["emails"] = re.findall(email_pattern, text)

    # Phone pattern (simple)
    phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
    pii["phone_numbers"] = re.findall(phone_pattern, text)

    # SSN pattern
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    pii["ssn"] = re.findall(ssn_pattern, text)

    # Credit card pattern (simple)
    cc_pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    pii["credit_cards"] = re.findall(cc_pattern, text)

    return {k: v for k, v in pii.items() if v}


def compute_perplexity(
    dataset: TrainingDataset,
    model_name: str = "gpt2",
    max_examples: Optional[int] = None,
    device: Union["Device", str] = "cpu",
) -> Dict[str, Any]:
    """Compute perplexity distribution for dataset using a HuggingFace model.

    Perplexity measures how well the model predicts the text - lower is better.
    Useful for identifying low-quality or out-of-distribution examples.

    Args:
        dataset: Dataset to analyze
        model_name: HuggingFace model name (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
        max_examples: Maximum number of examples to evaluate (None = all)
        device: Device to run on (Device enum or string: "cpu", "cuda", "cuda:0", "cuda:1", "mps")

    Returns:
        Dictionary with perplexity statistics

    Examples:
        >>> # Using enum (recommended)
        >>> from kerb.core.enums import Device
        >>> stats = compute_perplexity(dataset, model_name="gpt2", device=Device.CUDA)

        >>> # Using string (for backward compatibility)
        >>> stats = compute_perplexity(dataset, model_name="gpt2")
        >>> print(f"Average perplexity: {stats['mean_perplexity']:.2f}")

    Note:
        Requires transformers and torch packages.
        Install with: pip install transformers torch
    """
    from kerb.core.enums import Device, validate_enum_or_string

    try:
        import warnings

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        warnings.filterwarnings("ignore")
    except ImportError:
        return {
            "error": "Required packages not installed",
            "message": "Install with: pip install transformers torch",
        }

    # Validate and normalize device
    device_val = validate_enum_or_string(device, Device, "device")
    if isinstance(device_val, Device):
        device_str = device_val.value
    else:
        device_str = device_val

    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device_str)
        model.eval()

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        perplexities = []
        examples_to_process = (
            dataset.examples[:max_examples] if max_examples else dataset.examples
        )

        with torch.no_grad():
            for example in examples_to_process:
                text = example.get_text_content()
                if not text.strip():
                    continue

                # Tokenize
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )
                inputs = {k: v.to(device_str) for k, v in inputs.items()}

                # Compute loss (negative log-likelihood)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()

                # Perplexity = exp(loss)
                perplexity = torch.exp(torch.tensor(loss)).item()
                perplexities.append(perplexity)

        if not perplexities:
            return {"message": "No valid examples to compute perplexity"}

        # Calculate statistics
        perplexities.sort()
        n = len(perplexities)

        return {
            "model": model_name,
            "examples_evaluated": n,
            "mean_perplexity": sum(perplexities) / n,
            "median_perplexity": perplexities[n // 2],
            "min_perplexity": min(perplexities),
            "max_perplexity": max(perplexities),
            "p25_perplexity": perplexities[n // 4],
            "p75_perplexity": perplexities[3 * n // 4],
            "perplexities": perplexities,
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to compute perplexity with model {model_name}",
        }


def check_length_distribution(dataset: TrainingDataset) -> Dict[str, Any]:
    """Analyze token length distribution.

    Args:
        dataset: Dataset to analyze

    Returns:
        Dictionary with length statistics
    """
    lengths = []
    for example in dataset.examples:
        text = example.get_text_content()
        lengths.append(len(text.split()))

    lengths.sort()
    n = len(lengths)

    return {
        "count": n,
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "mean": sum(lengths) / n if n > 0 else 0,
        "median": lengths[n // 2] if n > 0 else 0,
        "p25": lengths[n // 4] if n > 0 else 0,
        "p75": lengths[3 * n // 4] if n > 0 else 0,
    }


def detect_duplicates(
    dataset: TrainingDataset, threshold: float = 0.95
) -> List[Tuple[int, int]]:
    """Find duplicate or near-duplicate examples.

    Args:
        dataset: Dataset to check
        threshold: Similarity threshold (1.0 = exact match)

    Returns:
        List of (index1, index2) pairs of duplicates
    """
    duplicates = []
    hashes = {}

    for i, example in enumerate(dataset.examples):
        content_hash = example.compute_hash()
        if content_hash in hashes:
            duplicates.append((hashes[content_hash], i))
        else:
            hashes[content_hash] = i

    return duplicates


def check_label_distribution(dataset: TrainingDataset) -> Dict[str, Any]:
    """Analyze label distribution for classification tasks.

    Args:
        dataset: Dataset to analyze

    Returns:
        Dictionary with label statistics
    """
    labels = [ex.label for ex in dataset.examples if ex.label is not None]

    if not labels:
        return {"message": "No labels found in dataset"}

    label_counts = Counter(labels)
    total = len(labels)

    return {
        "total_labeled": total,
        "unique_labels": len(label_counts),
        "label_counts": dict(label_counts),
        "label_percentages": {
            k: round(v / total * 100, 2) for k, v in label_counts.items()
        },
        "most_common": label_counts.most_common(5),
        "is_balanced": (
            max(label_counts.values()) / min(label_counts.values()) < 2
            if label_counts
            else False
        ),
    }


# ============================================================================
# System Prompts
# ============================================================================


def generate_system_prompt(
    task_description: str, examples: Optional[List[str]] = None
) -> str:
    """Generate system prompt from task description.

    Args:
        task_description: Description of the task
        examples: Optional example outputs

    Returns:
        Generated system prompt
    """
    prompt = f"You are an AI assistant specialized in {task_description}."

    if examples:
        prompt += "\n\nHere are some examples of expected outputs:\n"
        for i, example in enumerate(examples[:3], 1):
            prompt += f"{i}. {example}\n"

    prompt += "\nPlease provide accurate, helpful, and relevant responses."

    return prompt


def extract_system_prompts(dataset: TrainingDataset) -> List[str]:
    """Extract system prompts from dataset.

    Args:
        dataset: Dataset to analyze

    Returns:
        List of unique system prompts
    """
    system_prompts = set()

    for example in dataset.examples:
        if example.messages:
            for msg in example.messages:
                if msg.get("role") == "system":
                    system_prompts.add(msg.get("content", ""))

    return list(system_prompts)


def standardize_system_prompts(
    dataset: TrainingDataset, standard_prompt: str
) -> TrainingDataset:
    """Standardize system prompts across dataset.

    Args:
        dataset: Dataset to modify
        standard_prompt: Standard system prompt to use

    Returns:
        Modified dataset
    """
    modified_examples = []

    for example in dataset.examples:
        if example.messages:
            # Remove existing system prompts and add standard one
            messages = [msg for msg in example.messages if msg.get("role") != "system"]
            messages.insert(0, {"role": "system", "content": standard_prompt})

            modified_example = TrainingExample(
                messages=messages, metadata=example.metadata
            )
        else:
            modified_example = example

        modified_examples.append(modified_example)

    return TrainingDataset(
        examples=modified_examples,
        format=dataset.format,
        provider=dataset.provider,
        metadata={**dataset.metadata, "system_prompt_standardized": True},
    )


def optimize_system_prompt(
    task_examples: List[TrainingExample], max_length: int = 200
) -> str:
    """Optimize system prompt based on task examples.

    Args:
        task_examples: Examples of the task
        max_length: Maximum prompt length

    Returns:
        Optimized system prompt
    """
    # Extract common patterns from examples
    # This is a simplified implementation

    if not task_examples:
        return "You are a helpful AI assistant."

    # Analyze first few examples
    sample_texts = [ex.get_text_content()[:500] for ex in task_examples[:5]]

    # Simple heuristic: if examples contain technical terms, make prompt more technical
    technical_terms = ["code", "function", "variable", "class", "API", "algorithm"]
    is_technical = any(
        term in " ".join(sample_texts).lower() for term in technical_terms
    )

    if is_technical:
        prompt = "You are an expert AI assistant specializing in technical and programming tasks. Provide accurate, detailed, and well-structured responses."
    else:
        prompt = "You are a helpful and knowledgeable AI assistant. Provide clear, accurate, and helpful responses."

    return prompt[:max_length]


# ============================================================================
# Training Utilities
# ============================================================================


def create_training_config(
    model: str,
    n_epochs: int = 3,
    batch_size: Optional[int] = None,
    learning_rate_multiplier: Optional[float] = None,
    **kwargs,
) -> TrainingConfig:
    """Create training configuration.

    Args:
        model: Base model name
        n_epochs: Number of training epochs
        batch_size: Batch size (if None, provider determines automatically)
        learning_rate_multiplier: Learning rate multiplier
        **kwargs: Additional configuration options

    Returns:
        TrainingConfig
    """
    return TrainingConfig(
        model=model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate_multiplier=learning_rate_multiplier,
        **kwargs,
    )


def estimate_training_time(
    dataset: TrainingDataset, n_epochs: int = 3, batch_size: int = 8
) -> Dict[str, Any]:
    """Estimate training duration.

    Args:
        dataset: Training dataset
        n_epochs: Number of epochs
        batch_size: Batch size

    Returns:
        Dictionary with time estimates
    """
    n_examples = len(dataset)
    steps_per_epoch = n_examples // batch_size
    total_steps = steps_per_epoch * n_epochs

    # Rough estimates (seconds per step)
    time_per_step = 2.0  # This varies widely by model and hardware

    total_seconds = total_steps * time_per_step
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60

    return {
        "total_examples": n_examples,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_seconds": round(total_seconds),
        "estimated_minutes": round(total_minutes, 1),
        "estimated_hours": round(total_hours, 2),
    }


def calculate_optimal_batch_size(dataset_size: int, gpu_memory_gb: float = 16) -> int:
    """Calculate optimal batch size.

    Args:
        dataset_size: Size of dataset
        gpu_memory_gb: Available GPU memory in GB

    Returns:
        Recommended batch size
    """
    # Simple heuristics
    if gpu_memory_gb >= 40:
        base_batch_size = 32
    elif gpu_memory_gb >= 24:
        base_batch_size = 16
    elif gpu_memory_gb >= 16:
        base_batch_size = 8
    else:
        base_batch_size = 4

    # Adjust for dataset size
    if dataset_size < 100:
        return min(base_batch_size, dataset_size // 4)

    return base_batch_size


def recommend_learning_rate(model: str, dataset_size: int) -> float:
    """Recommend learning rate for fine-tuning.

    Args:
        model: Base model name
        dataset_size: Size of dataset

    Returns:
        Recommended learning rate multiplier
    """
    # Smaller datasets benefit from lower learning rates
    if dataset_size < 100:
        return 0.05
    elif dataset_size < 500:
        return 0.1
    elif dataset_size < 2000:
        return 0.2
    else:
        return 0.3


def create_hyperparameter_grid(
    n_epochs: List[int] = [3, 5, 10],
    batch_sizes: Optional[List[int]] = None,
    learning_rates: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Create hyperparameter search grid.

    Args:
        n_epochs: List of epoch values to try
        batch_sizes: List of batch sizes to try
        learning_rates: List of learning rate multipliers to try

    Returns:
        List of hyperparameter configurations
    """
    if batch_sizes is None:
        batch_sizes = [4, 8, 16]
    if learning_rates is None:
        learning_rates = [0.05, 0.1, 0.2]

    grid = []
    for epochs in n_epochs:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                grid.append(
                    {
                        "n_epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate_multiplier": lr,
                    }
                )

    return grid
