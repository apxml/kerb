"""Cost Estimation and Validation Example

This example demonstrates how to estimate training costs and validate datasets.

Main concepts:
- Estimating training tokens
- Calculating training costs for different providers
- Validating dataset format and structure
- Checking token limits
- Estimating training time
- Optimal batch size calculation

Use case: Planning and budgeting for fine-tuning runs, ensuring
datasets meet provider requirements before starting expensive training.
"""

from kerb.fine_tuning import (
    prepare_dataset,
    validate_dataset,
    estimate_cost,
    TrainingExample,
    TrainingDataset,
    DatasetFormat,
    ValidationLevel,
)
from kerb.fine_tuning.validation import (
    estimate_training_tokens,
    check_token_limits,
    validate_messages,
    validate_format,
    FineTuningProvider,
)
from kerb.fine_tuning.training import (
    estimate_training_time,
    calculate_optimal_batch_size,
)


def create_sample_training_dataset():
    """Create a sample training dataset."""
    examples = []
    
    # Create diverse examples with varying lengths (reduced for faster execution)
    for i in range(30):  # Reduced from 100 to 30
        # Short examples
        if i < 10:  # Reduced from 30
            user_msg = f"Question {i}: What is Python?"
            assistant_msg = "Python is a high-level programming language."
        # Medium examples
        elif i < 20:  # Reduced from 70
            user_msg = f"Question {i}: Explain how to use for loops in Python with examples."
            assistant_msg = """For loops in Python iterate over sequences. Here's an example:

for i in range(5):
    print(i)

This prints numbers 0 through 4. You can also iterate over lists:

fruits = ['apple', 'banana', 'orange']
for fruit in fruits:
    print(fruit)"""
        # Long examples
        else:
            user_msg = f"Question {i}: Write a comprehensive guide on Python decorators with multiple examples."
            assistant_msg = """Python decorators are a powerful feature that allows you to modify function behavior.

Basic Decorator:
def my_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

Decorator with Arguments:
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

Class-based Decorators:
class Counter:
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call {self.count}")
        return self.func(*args, **kwargs)

Decorators are widely used in frameworks like Flask and Django."""
        
        examples.append(TrainingExample(
            messages=[
                {"role": "system", "content": "You are a helpful Python programming tutor."},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        ))
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CHAT,
        metadata={"purpose": "python_tutoring"}
    )


def main():
    """Run cost estimation and validation example."""
    
    print("="*80)
    print("COST ESTIMATION AND VALIDATION EXAMPLE")
    print("="*80)
    
    # Step 1: Create sample dataset
    print("\nStep 1: Creating sample dataset")
    dataset = create_sample_training_dataset()
    print(f"Dataset size: {len(dataset)} examples")
    
    # Step 2: Estimate training tokens
    print("\n" + "="*80)
    print("TOKEN ESTIMATION")
    print("="*80)
    
    total_tokens = estimate_training_tokens(dataset)
    print(f"\nEstimated total tokens: {total_tokens:,}")
    print(f"Average tokens per example: {total_tokens // len(dataset):,}")
    
    # For different epoch counts
    for n_epochs in [1, 3, 5]:
        training_tokens = total_tokens * n_epochs
        print(f"Training tokens ({n_epochs} epochs): {training_tokens:,}")
    
    # Step 3: Estimate costs for different models
    print("\n" + "="*80)
    print("COST ESTIMATION")
    print("="*80)
    
    models = [
        ("gpt-3.5-turbo", [1, 3, 5]),
        ("gpt-4", [1, 3]),
    ]
    
    for model, epoch_options in models:
        print(f"\n{model.upper()}:")
        print("-"*40)
        
        for n_epochs in epoch_options:
            cost_estimate = estimate_cost(dataset, model=model, n_epochs=n_epochs)
            print(f"  {n_epochs} epoch{'s' if n_epochs > 1 else ''}:")
            print(f"    Training tokens: {cost_estimate['total_training_tokens']:,}")
            print(f"    Estimated cost: ${cost_estimate['estimated_training_cost_usd']:.2f}")
            print(f"    Cost per epoch: ${cost_estimate['cost_per_epoch_usd']:.2f}")
    
    # Step 4: Dataset validation
    print("\n" + "="*80)
    print("DATASET VALIDATION")
    print("="*80)
    
    # Validate with moderate level only (to save time)
    print(f"\nValidation level: MODERATE")
    print("-"*40)
    
    validation_result = validate_dataset(dataset, level=ValidationLevel.MODERATE)
    
    print(f"Is valid: {validation_result.is_valid}")
    print(f"Total examples: {validation_result.total_examples}")
    print(f"Valid examples: {validation_result.valid_examples}")
    print(f"Invalid examples: {validation_result.invalid_examples}")
    
    if validation_result.errors:
        print(f"Errors: {len(validation_result.errors)}")
        for error in validation_result.errors[:3]:
            print(f"  - {error}")
    
    if validation_result.warnings:
        print(f"Warnings: {len(validation_result.warnings)}")
        for warning in validation_result.warnings[:3]:
            print(f"  - {warning}")
    
    # Step 5: Check token limits
    print("\n" + "="*80)
    print("TOKEN LIMIT CHECKS")
    print("="*80)
    
    # Check only one limit to save time
    max_tokens = 4096
    
    print(f"\nChecking limit: {max_tokens:,} tokens")
    print("-"*40)
    
    token_check = check_token_limits(dataset, max_tokens=max_tokens)
    
    print(f"Total examples: {token_check['total_examples']}")
    print(f"Examples exceeding limit: {token_check['exceeding_limit']}")
    print(f"Average tokens: {token_check['avg_tokens']:.0f}")
    print(f"Max tokens found: {token_check['max_tokens_found']:.0f}")
    print(f"Min tokens found: {token_check['min_tokens_found']:.0f}")
    
    if token_check['exceeding_limit'] > 0:
        print(f"\nFirst few exceeding examples:")
        for ex in token_check['exceeding_examples'][:3]:
            print(f"  Example {ex['index']}: {ex['tokens']} tokens")
    
    # Step 6: Validate messages structure
    print("\n" + "="*80)
    print("MESSAGE STRUCTURE VALIDATION")
    print("="*80)
    
    # Valid messages
    valid_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    result = validate_messages(valid_messages)
    print("\nValid messages:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    
    # Invalid messages
    invalid_messages = [
        {"role": "user"},  # Missing content
        {"content": "Hello!"},  # Missing role
        {"role": "unknown", "content": "Hi"}  # Invalid role
    ]
    
    result = validate_messages(invalid_messages)
    print("\nInvalid messages:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {len(result.errors)}")
    if result.errors:
        for error in result.errors:
            print(f"    - {error}")
    if result.warnings:
        for warning in result.warnings:
            print(f"    - {warning}")
    
    # Step 7: Estimate training time
    print("\n" + "="*80)
    print("TRAINING TIME ESTIMATION")
    print("="*80)
    
    batch_sizes = [4, 8, 16]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-"*40)
        
        time_est = estimate_training_time(dataset, n_epochs=3, batch_size=batch_size)
        
        print(f"Steps per epoch: {time_est['steps_per_epoch']}")
        print(f"Total steps: {time_est['total_steps']}")
        print(f"Estimated time: {time_est['estimated_hours']:.2f} hours")
        print(f"              ({time_est['estimated_minutes']:.1f} minutes)")
    
    # Step 8: Calculate optimal batch size
    print("\n" + "="*80)
    print("OPTIMAL BATCH SIZE")
    print("="*80)
    
    gpu_configs = [
        (8, "Consumer GPU"),
        (16, "Professional GPU"),
        (24, "High-end GPU"),
        (40, "Data center GPU"),
    ]
    
    print(f"\nFor dataset size: {len(dataset)} examples")
    print("-"*40)
    
    for gpu_memory, description in gpu_configs:
        batch_size = calculate_optimal_batch_size(len(dataset), gpu_memory)
        print(f"{description} ({gpu_memory}GB): batch_size={batch_size}")
    
    # Step 9: Budget planning summary
    print("\n" + "="*80)
    print("BUDGET PLANNING SUMMARY")
    print("="*80)
    
    print(f"\nDataset: {len(dataset)} examples")
    print(f"Total tokens: {total_tokens:,}")
    print(f"\nRecommended configuration:")
    print(f"  Model: gpt-3.5-turbo")
    print(f"  Epochs: 3")
    print(f"  Batch size: 8")
    
    recommended_cost = estimate_cost(dataset, model="gpt-3.5-turbo", n_epochs=3)
    recommended_time = estimate_training_time(dataset, n_epochs=3, batch_size=8)
    
    print(f"\nEstimated cost: ${recommended_cost['estimated_training_cost_usd']:.2f}")
    print(f"Estimated time: {recommended_time['estimated_hours']:.2f} hours")
    print(f"\nValidation: {'PASSED' if validation_result.is_valid else 'FAILED'}")
    
    if validation_result.is_valid:
        print("\nDataset is ready for fine-tuning!")
    else:
        print("\nPlease fix validation errors before proceeding.")


if __name__ == "__main__":
    main()
