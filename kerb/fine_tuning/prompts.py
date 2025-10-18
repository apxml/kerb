"""System prompt generation and manipulation functions."""

from typing import List, Optional

from .types import TrainingDataset, TrainingExample


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
