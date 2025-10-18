"""Dataset validation functions for fine-tuning."""

from typing import List, Dict, Any, Optional

from .types import (
    TrainingDataset,
    ValidationResult,
    ValidationLevel,
    DatasetFormat,
    FineTuningProvider,
)


def validate_dataset(
    dataset: TrainingDataset,
    level: ValidationLevel = ValidationLevel.MODERATE,
    max_tokens: Optional[int] = None
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
                    result.add_error(f"Example {i}, Message {j}: Missing 'content' field")
        
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
                    result.add_error(f"Example {i}: Exceeds token limit ({estimated_tokens} > {max_tokens})")
                else:
                    result.add_warning(f"Example {i}: May exceed token limit ({estimated_tokens} tokens)")
        
        if result.errors and level == ValidationLevel.STRICT:
            result.valid_examples = i
            result.invalid_examples = len(dataset) - i
            return result
    
    result.valid_examples = len(dataset) - result.invalid_examples
    
    # Final checks
    if result.valid_examples < 10:
        result.add_warning(f"Dataset has only {result.valid_examples} valid examples. Recommended: at least 50-100")
    
    return result


def validate_format(data: List[Dict[str, Any]], provider: FineTuningProvider) -> ValidationResult:
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
                result.add_error(f"Example {i}: Missing 'messages' field for OpenAI format")
            else:
                for j, msg in enumerate(item["messages"]):
                    if "role" not in msg or "content" not in msg:
                        result.add_error(f"Example {i}, Message {j}: Must have 'role' and 'content'")
        
        elif provider == FineTuningProvider.ANTHROPIC:
            if "messages" not in item:
                result.add_error(f"Example {i}: Missing 'messages' field for Anthropic format")
        
        # Add more provider-specific validation as needed
    
    result.valid_examples = len(data) - len(result.errors)
    result.invalid_examples = len(result.errors)
    
    return result


def check_token_limits(
    dataset: TrainingDataset,
    max_tokens: int = 4096,
    tokenizer_name: str = "cl100k_base"
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
            result.add_warning(f"Message {i}: Unusual role '{msg['role']}'. Expected: {valid_roles}")
        
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
    dataset: TrainingDataset,
    model: str = "gpt-3.5-turbo",
    n_epochs: int = 3
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
        "gpt-3.5-turbo": {"training": 0.008, "input": 0.003, "output": 0.006},  # per 1K tokens
        "gpt-4": {"training": 0.03, "input": 0.03, "output": 0.06},
    }
    
    base_model = model.split("-")[0] + "-" + model.split("-")[1] if "-" in model else model
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
