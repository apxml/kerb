"""Fine-tuning utilities for preparing and managing LLM training data.

This module provides comprehensive tools for fine-tuning LLMs across multiple providers:

Common Usage:
    from kerb.fine_tuning import prepare_dataset, TrainingDataset
    from kerb.fine_tuning import validate_dataset, analyze_dataset

Submodules:
    types - Core data classes and enums
    dataset - Dataset preparation and manipulation
    formats - Format conversion for different providers
    jsonl - JSONL file utilities
    validation - Dataset validation functions
    quality - Data quality analysis
    prompts - System prompt utilities
    training - Training configuration and optimization

Dataset Preparation:
    prepare_dataset() - Main function to prepare datasets for fine-tuning
    split_dataset() - Split data into train/validation/test sets
    balance_dataset() - Balance dataset by label/category
    augment_dataset() - Augment training data with variations
    deduplicate_dataset() - Remove duplicate examples
    sample_dataset() - Sample subset of dataset
    shuffle_dataset() - Randomize dataset order
    filter_dataset() - Filter dataset by criteria
    
Format Conversion:
    to_openai_format() - Convert to OpenAI fine-tuning format
    to_anthropic_format() - Convert to Anthropic fine-tuning format
    to_google_format() - Convert to Google AI fine-tuning format
    to_huggingface_format() - Convert to HuggingFace format
    to_generic_format() - Convert to generic JSONL format
    from_csv() - Convert CSV to fine-tuning format
    from_json() - Convert JSON to fine-tuning format
    from_parquet() - Convert Parquet to fine-tuning format
    
JSONL Utilities:
    write_jsonl() - Write data to JSONL file
    read_jsonl() - Read data from JSONL file
    append_jsonl() - Append data to JSONL file
    merge_jsonl() - Merge multiple JSONL files
    validate_jsonl() - Validate JSONL file format
    count_jsonl_lines() - Count lines in JSONL file
    stream_jsonl() - Stream large JSONL files
    
Validation:
    validate_dataset() - Validate dataset for fine-tuning
    validate_format() - Validate format for specific provider
    check_token_limits() - Check if examples exceed token limits
    validate_messages() - Validate message structure
    estimate_training_tokens() - Estimate total training tokens
    estimate_cost() - Estimate fine-tuning cost
    validate_completion_format() - Validate completion-based format
    validate_chat_format() - Validate chat-based format
    
Data Quality:
    analyze_dataset() - Analyze dataset statistics
    check_data_quality() - Check for quality issues
    detect_pii() - Detect personally identifiable information
    compute_perplexity() - Compute perplexity with HuggingFace models
    check_length_distribution() - Analyze token length distribution
    detect_duplicates() - Find duplicate or near-duplicate examples
    check_label_distribution() - Analyze label distribution
    
System Prompts:
    generate_system_prompt() - Generate system prompts from examples
    extract_system_prompts() - Extract system prompts from dataset
    standardize_system_prompts() - Standardize system prompts
    optimize_system_prompt() - Optimize system prompt for task
    
Training Utilities:
    create_training_config() - Create training configuration
    estimate_training_time() - Estimate training duration
    calculate_optimal_batch_size() - Calculate optimal batch size
    recommend_learning_rate() - Recommend learning rate
    create_hyperparameter_grid() - Create hyperparameter search grid
    
Data Classes:
    TrainingExample - Single training example
    TrainingDataset - Complete training dataset
    ValidationResult - Validation results
    DatasetStats - Dataset statistics
    TrainingConfig - Training configuration
    
Enums:
    FineTuningProvider - Supported providers
    DatasetFormat - Supported formats
    SplitStrategy - Dataset split strategies
    ValidationLevel - Validation strictness levels
"""

# Core types and enums
from .types import (
    FineTuningProvider,
    DatasetFormat,
    SplitStrategy,
    ValidationLevel,
    TrainingExample,
    TrainingDataset,
    ValidationResult,
    DatasetStats,
    TrainingConfig,
)

# Submodules
from . import (
    dataset,
    formats,
    jsonl,
    validation,
    quality,
    prompts,
    training,
)

# Most commonly used dataset functions
from .dataset import (
    prepare_dataset,
    split_dataset,
    balance_dataset,
    deduplicate_dataset,
    sample_dataset,
    filter_dataset,
)

# Most commonly used format conversion functions
from .formats import (
    to_openai_format,
    to_anthropic_format,
    from_csv,
    from_json,
)

# Most commonly used JSONL utilities
from .jsonl import (
    write_jsonl,
    read_jsonl,
)

# Most commonly used validation functions
from .validation import (
    validate_dataset,
    estimate_cost,
)

# Most commonly used quality functions
from .quality import (
    analyze_dataset,
    check_data_quality,
)

# Most commonly used training utilities
from .training import (
    create_training_config,
    estimate_training_time,
)

__all__ = [
    # Core types
    "FineTuningProvider",
    "DatasetFormat",
    "SplitStrategy",
    "ValidationLevel",
    "TrainingExample",
    "TrainingDataset",
    "ValidationResult",
    "DatasetStats",
    "TrainingConfig",
    
    # Submodules
    "dataset",
    "formats",
    "jsonl",
    "validation",
    "quality",
    "prompts",
    "training",
    
    # Common dataset operations
    "prepare_dataset",
    "split_dataset",
    "balance_dataset",
    "deduplicate_dataset",
    "sample_dataset",
    "filter_dataset",
    
    # Common format conversions
    "to_openai_format",
    "to_anthropic_format",
    "from_csv",
    "from_json",
    
    # Common JSONL operations
    "write_jsonl",
    "read_jsonl",
    
    # Common validation
    "validate_dataset",
    "estimate_cost",
    
    # Common quality checks
    "analyze_dataset",
    "check_data_quality",
    
    # Common training utilities
    "create_training_config",
    "estimate_training_time",
]
