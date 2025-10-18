"""Format conversion functions for different fine-tuning providers."""

import json
import csv
from typing import List, Dict, Any, Optional

from .types import TrainingDataset, TrainingExample, DatasetFormat


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
                contents.append({
                    "role": "user" if msg["role"] in ["user", "human"] else "model",
                    "parts": [{"text": msg.get("content", "")}]
                })
            result.append({"contents": contents})
        elif dataset.format == DatasetFormat.COMPLETION:
            contents = []
            if example.prompt:
                contents.append({"role": "user", "parts": [{"text": example.prompt}]})
            if example.completion:
                contents.append({"role": "model", "parts": [{"text": example.completion}]})
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
                result.append({
                    "prompt": example.prompt,
                    "completion": example.completion
                })
        elif dataset.format == DatasetFormat.CLASSIFICATION:
            result.append({
                "text": example.prompt or example.get_text_content(),
                "label": example.label
            })
    
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
    format: DatasetFormat = DatasetFormat.COMPLETION
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
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get(prompt_column, "")
            completion = row.get(completion_column, "") if completion_column else None
            label = row.get(label_column) if label_column else None
            
            if format == DatasetFormat.CHAT and completion:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
                example = TrainingExample(messages=messages)
            else:
                example = TrainingExample(
                    prompt=prompt,
                    completion=completion,
                    label=label
                )
            
            examples.append(example)
    
    return TrainingDataset(examples=examples, format=format)


def from_json(filepath: str, format: DatasetFormat = DatasetFormat.CHAT) -> TrainingDataset:
    """Convert JSON file to training dataset.
    
    Args:
        filepath: Path to JSON file
        format: Target format
        
    Returns:
        TrainingDataset
    """
    # Import here to avoid circular dependency
    from .dataset import prepare_dataset
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    return prepare_dataset(data, format=format, validate=False)


def from_parquet(filepath: str, format: DatasetFormat = DatasetFormat.CHAT) -> TrainingDataset:
    """Convert Parquet file to training dataset.
    
    Args:
        filepath: Path to Parquet file
        format: Target format
        
    Returns:
        TrainingDataset
        
    Note:
        Requires pandas and pyarrow packages
    """
    # Import here to avoid circular dependency
    from .dataset import prepare_dataset
    
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Parquet support. Install with: pip install pandas pyarrow")
    
    df = pd.read_parquet(filepath)
    data = df.to_dict('records')
    
    return prepare_dataset(data, format=format, validate=False)
