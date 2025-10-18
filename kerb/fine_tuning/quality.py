"""Data quality analysis functions for fine-tuning datasets."""

import re
from typing import Dict, Any, List, Tuple, Union, TYPE_CHECKING
from collections import Counter

from .types import TrainingDataset, DatasetStats

if TYPE_CHECKING:
    from kerb.core.enums import Device


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
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    pii["emails"] = re.findall(email_pattern, text)
    
    # Phone pattern (simple)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    pii["phone_numbers"] = re.findall(phone_pattern, text)
    
    # SSN pattern
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    pii["ssn"] = re.findall(ssn_pattern, text)
    
    # Credit card pattern (simple)
    cc_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    pii["credit_cards"] = re.findall(cc_pattern, text)
    
    return {k: v for k, v in pii.items() if v}


def compute_perplexity(
    dataset: TrainingDataset,
    model_name: str = "gpt2",
    max_examples: int = None,
    device: Union['Device', str] = "cpu"
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
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError:
        return {
            "error": "Required packages not installed",
            "message": "Install with: pip install transformers torch"
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
        examples_to_process = dataset.examples[:max_examples] if max_examples else dataset.examples
        
        with torch.no_grad():
            for example in examples_to_process:
                text = example.get_text_content()
                if not text.strip():
                    continue
                
                # Tokenize
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
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
            "message": f"Failed to compute perplexity with model {model_name}"
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


def detect_duplicates(dataset: TrainingDataset, threshold: float = 0.95) -> List[Tuple[int, int]]:
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
        "label_percentages": {k: round(v / total * 100, 2) for k, v in label_counts.items()},
        "most_common": label_counts.most_common(5),
        "is_balanced": max(label_counts.values()) / min(label_counts.values()) < 2 if label_counts else False,
    }
