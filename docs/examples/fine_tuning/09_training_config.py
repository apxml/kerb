"""Training Configuration Example

This example demonstrates how to create optimal training configurations.

Main concepts:
- Creating training configurations
- Calculating optimal batch sizes
- Recommending learning rates
- Creating hyperparameter grids for tuning
- Estimating training time and resources
- Provider-specific configurations
- Best practices for training setup

Use case: Setting up fine-tuning runs with optimal hyperparameters,
planning hyperparameter searches, and understanding training requirements.
"""

from kerb.fine_tuning import (
    create_training_config,
    estimate_training_time,
    TrainingConfig,
    TrainingDataset,
    TrainingExample,
    DatasetFormat,
)
from kerb.fine_tuning.training import (
    calculate_optimal_batch_size,
    recommend_learning_rate,
    create_hyperparameter_grid,
)


def create_sample_dataset(size: int = 100):
    """Create a sample dataset of specified size."""
    examples = []
    
    for i in range(size):
        examples.append(TrainingExample(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question {i}: Explain a concept."},
                {"role": "assistant", "content": f"Answer {i}: Here is the explanation of the concept."}
            ]
        ))
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CHAT
    )


def main():
    """Run training configuration example."""
    
    print("="*80)
    print("TRAINING CONFIGURATION EXAMPLE")
    print("="*80)
    
    # Step 1: Create sample datasets of different sizes
    print("\nStep 1: Creating sample datasets")
    
    datasets = {
        "small": create_sample_dataset(50),
        "medium": create_sample_dataset(500),
        "large": create_sample_dataset(2000),
    }
    
    for name, ds in datasets.items():
        print(f"  {name.capitalize()}: {len(ds)} examples")
    
    # Step 2: Basic training configurations
    print("\n" + "="*80)
    print("BASIC TRAINING CONFIGURATIONS")
    print("="*80)
    
    # Simple configuration
    print("\nSimple configuration (defaults):")
    simple_config = create_training_config(
        model="gpt-3.5-turbo",
        n_epochs=3
    )
    print(f"  Model: {simple_config.model}")
    print(f"  Epochs: {simple_config.n_epochs}")
    print(f"  Batch size: {simple_config.batch_size or 'auto'}")
    print(f"  Learning rate multiplier: {simple_config.learning_rate_multiplier or 'auto'}")
    
    # Custom configuration
    print("\nCustom configuration:")
    custom_config = create_training_config(
        model="gpt-3.5-turbo",
        n_epochs=5,
        batch_size=8,
        learning_rate_multiplier=0.1
    )
    print(f"  Model: {custom_config.model}")
    print(f"  Epochs: {custom_config.n_epochs}")
    print(f"  Batch size: {custom_config.batch_size}")
    print(f"  Learning rate multiplier: {custom_config.learning_rate_multiplier}")
    
    # Step 3: Optimal batch size calculation
    print("\n" + "="*80)
    print("OPTIMAL BATCH SIZE CALCULATION")
    print("="*80)
    
    gpu_configs = [
        (8, "Consumer GPU (8GB)"),
        (16, "Professional GPU (16GB)"),
        (24, "High-end GPU (24GB)"),
        (40, "Data center GPU (40GB)"),
    ]
    
    for name, ds in datasets.items():
        print(f"\n{name.capitalize()} dataset ({len(ds)} examples):")
        print("-"*40)
        
        for gpu_memory, description in gpu_configs:
            batch_size = calculate_optimal_batch_size(len(ds), gpu_memory)
            print(f"  {description}: batch_size={batch_size}")
    
    # Step 4: Learning rate recommendations
    print("\n" + "="*80)
    print("LEARNING RATE RECOMMENDATIONS")
    print("="*80)
    
    models = ["gpt-3.5-turbo", "gpt-4"]
    
    for model in models:
        print(f"\n{model.upper()}:")
        print("-"*40)
        
        for name, ds in datasets.items():
            lr = recommend_learning_rate(model, len(ds))
            print(f"  {name.capitalize()} ({len(ds)} examples): {lr}")
    
    # Step 5: Complete configuration for each dataset
    print("\n" + "="*80)
    print("RECOMMENDED CONFIGURATIONS")
    print("="*80)
    
    for name, ds in datasets.items():
        print(f"\n{name.upper()} DATASET ({len(ds)} examples):")
        print("-"*40)
        
        batch_size = calculate_optimal_batch_size(len(ds), gpu_memory_gb=16)
        lr = recommend_learning_rate("gpt-3.5-turbo", len(ds))
        
        # Adjust epochs based on dataset size
        if len(ds) < 100:
            n_epochs = 5
        elif len(ds) < 1000:
            n_epochs = 3
        else:
            n_epochs = 2
        
        config = create_training_config(
            model="gpt-3.5-turbo",
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate_multiplier=lr
        )
        
        print(f"  Model: {config.model}")
        print(f"  Epochs: {config.n_epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate_multiplier}")
        
        # Estimate training time
        time_est = estimate_training_time(ds, n_epochs=config.n_epochs, batch_size=config.batch_size)
        print(f"\n  Estimated time: {time_est['estimated_hours']:.2f} hours")
        print(f"  Steps per epoch: {time_est['steps_per_epoch']}")
        print(f"  Total steps: {time_est['total_steps']}")
    
    # Step 6: Hyperparameter grid search
    print("\n" + "="*80)
    print("HYPERPARAMETER GRID SEARCH")
    print("="*80)
    
    print("\nCreating hyperparameter grid:")
    grid = create_hyperparameter_grid(
        n_epochs=[3, 5],
        batch_sizes=[4, 8],
        learning_rates=[0.05, 0.1, 0.2]
    )
    
    print(f"Total configurations: {len(grid)}")
    print("\nFirst 5 configurations:")
    print("-"*40)
    for i, config in enumerate(grid[:5], 1):
        print(f"  {i}. epochs={config['n_epochs']}, "
              f"batch_size={config['batch_size']}, "
              f"lr={config['learning_rate_multiplier']}")
    
    # Step 7: Provider-specific configurations
    print("\n" + "="*80)
    print("PROVIDER-SPECIFIC CONFIGURATIONS")
    print("="*80)
    
    # OpenAI configuration
    print("\nOpenAI (gpt-3.5-turbo):")
    print("-"*40)
    openai_config = create_training_config(
        model="gpt-3.5-turbo",
        n_epochs=3,
        batch_size=8,
        learning_rate_multiplier=0.1
    )
    print(f"  Model: {openai_config.model}")
    print(f"  Epochs: {openai_config.n_epochs}")
    print(f"  Batch size: {openai_config.batch_size}")
    print(f"  Learning rate multiplier: {openai_config.learning_rate_multiplier}")
    print("\n  Note: OpenAI uses learning_rate_multiplier (0.05-2.0)")
    
    # GPT-4 configuration (more expensive, use fewer epochs)
    print("\nOpenAI (gpt-4):")
    print("-"*40)
    gpt4_config = create_training_config(
        model="gpt-4",
        n_epochs=2,
        batch_size=4,
        learning_rate_multiplier=0.05
    )
    print(f"  Model: {gpt4_config.model}")
    print(f"  Epochs: {gpt4_config.n_epochs}")
    print(f"  Batch size: {gpt4_config.batch_size}")
    print(f"  Learning rate multiplier: {gpt4_config.learning_rate_multiplier}")
    print("\n  Note: GPT-4 requires fewer epochs and lower learning rate")
    
    # Step 8: Configuration best practices
    print("\n" + "="*80)
    print("CONFIGURATION BEST PRACTICES")
    print("="*80)
    
    print("\n1. Dataset Size Guidelines:")
    print("   - Small (<100): 5 epochs, lr=0.05, smaller batch")
    print("   - Medium (100-1000): 3 epochs, lr=0.1, standard batch")
    print("   - Large (>1000): 2-3 epochs, lr=0.2, larger batch")
    
    print("\n2. Batch Size:")
    print("   - Start with 4-8 for most cases")
    print("   - Increase if you have GPU memory")
    print("   - Decrease if you get OOM errors")
    print("   - Must divide evenly into dataset size")
    
    print("\n3. Learning Rate:")
    print("   - Start with 0.1 (default)")
    print("   - Decrease (0.05) if training is unstable")
    print("   - Increase (0.2-0.3) for larger datasets")
    print("   - Never exceed 2.0")
    
    print("\n4. Epochs:")
    print("   - More epochs != better performance")
    print("   - Watch for overfitting")
    print("   - Use validation set to determine optimal")
    print("   - Start with 3 epochs")
    
    print("\n5. Model Selection:")
    print("   - gpt-3.5-turbo: Cost-effective, fast training")
    print("   - gpt-4: Higher quality, more expensive")
    print("   - Consider base model capabilities")
    
    # Step 9: Configuration validation
    print("\n" + "="*80)
    print("CONFIGURATION VALIDATION")
    print("="*80)
    
    medium_ds = datasets["medium"]
    test_config = create_training_config(
        model="gpt-3.5-turbo",
        n_epochs=3,
        batch_size=8,
        learning_rate_multiplier=0.1
    )
    
    print(f"\nValidating configuration for {len(medium_ds)} examples:")
    print("-"*40)
    print(f"Configuration:")
    print(f"  Epochs: {test_config.n_epochs}")
    print(f"  Batch size: {test_config.batch_size}")
    print(f"  Learning rate: {test_config.learning_rate_multiplier}")
    
    # Check if batch size divides dataset
    if len(medium_ds) % test_config.batch_size != 0:
        print(f"\n  WARNING: Batch size {test_config.batch_size} doesn't divide evenly into {len(medium_ds)}")
        print(f"  Recommendation: Use batch size of {len(medium_ds) // (len(medium_ds) // test_config.batch_size)}")
    else:
        print(f"\n  OK: Batch size divides evenly ({len(medium_ds) // test_config.batch_size} batches)")
    
    # Check learning rate
    if test_config.learning_rate_multiplier < 0.05:
        print("  WARNING: Learning rate very low, training may be slow")
    elif test_config.learning_rate_multiplier > 0.5:
        print("  WARNING: Learning rate high, training may be unstable")
    else:
        print("  OK: Learning rate in recommended range")
    
    # Check epochs
    if test_config.n_epochs > 10:
        print("  WARNING: Many epochs, watch for overfitting")
    else:
        print("  OK: Epoch count reasonable")
    
    # Step 10: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey takeaways:")
    print("  1. Match hyperparameters to dataset size")
    print("  2. Start with conservative settings")
    print("  3. Use validation set to tune")
    print("  4. Consider cost vs performance")
    print("  5. Monitor training for issues")
    print("\nRecommended starting point:")
    print("  - Model: gpt-3.5-turbo")
    print("  - Epochs: 3")
    print("  - Batch size: 8")
    print("  - Learning rate: 0.1")
    print("\nThen adjust based on results!")


if __name__ == "__main__":
    main()
