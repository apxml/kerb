"""Training configuration and optimization utilities."""

from typing import List, Dict, Any, Optional

from .types import TrainingConfig, TrainingDataset


def create_training_config(
    model: str,
    n_epochs: int = 3,
    batch_size: Optional[int] = None,
    learning_rate_multiplier: Optional[float] = None,
    **kwargs
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
        **kwargs
    )


def estimate_training_time(
    dataset: TrainingDataset,
    n_epochs: int = 3,
    batch_size: int = 8
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
    n_epochs: List[int] = None,
    batch_sizes: Optional[List[int]] = None,
    learning_rates: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """Create hyperparameter search grid.
    
    Args:
        n_epochs: List of epoch values to try
        batch_sizes: List of batch sizes to try
        learning_rates: List of learning rate multipliers to try
        
    Returns:
        List of hyperparameter configurations
    """
    if n_epochs is None:
        n_epochs = [3, 5, 10]
    if batch_sizes is None:
        batch_sizes = [4, 8, 16]
    if learning_rates is None:
        learning_rates = [0.05, 0.1, 0.2]
    
    grid = []
    for epochs in n_epochs:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                grid.append({
                    "n_epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate_multiplier": lr,
                })
    
    return grid
