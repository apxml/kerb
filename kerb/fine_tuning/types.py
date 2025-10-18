"""Fine-tuning types, enums, and data classes."""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

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
