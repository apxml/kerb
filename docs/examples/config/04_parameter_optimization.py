"""Model Parameter Optimization Example

This example demonstrates managing and optimizing model parameters for different tasks.

Main concepts:
- Task-specific model configurations
- Parameter tuning strategies
- Temperature and token management
- Model comparison for different use cases
- A/B testing configurations
"""

from kerb.config import ConfigManager, create_model_config
from kerb.config.enums import ProviderType


def main():
    """Run model parameter optimization example."""
    
    print("="*80)
    print("MODEL PARAMETER OPTIMIZATION EXAMPLE")
    print("="*80)
    
    config = ConfigManager(app_name="parameter_optimization")
    
    # Step 1: Create task-specific model configurations
    print("\nStep 1: Task-Specific Model Configurations")
    print("-"*80)
    
    # Creative writing - higher temperature
    creative_model = create_model_config(
        name="gpt-4-creative",
        provider=ProviderType.OPENAI,
        temperature=0.9,  # High for creativity
        top_p=0.95,
        max_tokens=2048,
        presence_penalty=0.6,  # Encourage diverse topics
        frequency_penalty=0.3,  # Reduce repetition
        metadata={"task": "creative_writing", "quality": "high_variance"}
    )
    config.add_model(creative_model)
    print(f"Creative Writing Model: {creative_model.name}")
    print(f"  Temperature: {creative_model.temperature}")
    print(f"  Presence penalty: {creative_model.presence_penalty}")
    
    # Code generation - moderate temperature
    code_model = create_model_config(
        name="gpt-4-code",
        provider=ProviderType.OPENAI,
        temperature=0.3,  # Low-moderate for accuracy
        top_p=1.0,
        max_tokens=4096,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        metadata={"task": "code_generation", "quality": "accuracy"}
    )
    config.add_model(code_model)
    print(f"\nCode Generation Model: {code_model.name}")
    print(f"  Temperature: {code_model.temperature}")
    print(f"  Max tokens: {code_model.max_tokens}")
    
    # Data analysis - very low temperature
    analysis_model = create_model_config(
        name="gpt-4-analysis",
        provider=ProviderType.OPENAI,
        temperature=0.1,  # Very low for consistency
        top_p=1.0,
        max_tokens=4096,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        metadata={"task": "data_analysis", "quality": "deterministic"}
    )
    config.add_model(analysis_model)
    print(f"\nData Analysis Model: {analysis_model.name}")
    print(f"  Temperature: {analysis_model.temperature}")
    print(f"  Task: {analysis_model.metadata['task']}")
    
    # Conversational - balanced
    chat_model = create_model_config(
        name="gpt-4-chat",
        provider=ProviderType.OPENAI,
        temperature=0.7,  # Balanced
        top_p=0.9,
        max_tokens=2048,
        presence_penalty=0.2,
        frequency_penalty=0.2,
        metadata={"task": "conversation", "quality": "balanced"}
    )
    config.add_model(chat_model)
    print(f"\nConversational Model: {chat_model.name}")
    print(f"  Temperature: {chat_model.temperature}")
    print(f"  Top-p: {chat_model.top_p}")
    
    # Step 2: Token management strategies
    print("\nStep 2: Token Management Strategies")
    print("-"*80)
    
    token_configs = {
        "summarization": {
            "name": "gpt-4-summarize",
            "max_tokens": 500,  # Short summaries
            "temperature": 0.3,
            "purpose": "Concise summaries"
        },
        "long_form": {
            "name": "gpt-4-longform",
            "max_tokens": 4096,  # Long content
            "temperature": 0.7,
            "purpose": "Detailed articles"
        },
        "quick_response": {
            "name": "gpt-3.5-quick",
            "max_tokens": 150,  # Very short
            "temperature": 0.5,
            "purpose": "Quick answers"
        }
    }
    
    for task, params in token_configs.items():
        model = create_model_config(
            name=params["name"],
            provider=ProviderType.OPENAI,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            metadata={"token_strategy": task, "purpose": params["purpose"]}
        )
        config.add_model(model)
        print(f"{task.replace('_', ' ').title():20} - {params['max_tokens']:4} tokens - {params['purpose']}")
    
    # Step 3: A/B testing configurations
    print("\nStep 3: A/B Testing Model Variants")
    print("-"*80)
    
    # Variant A: Conservative
    variant_a = create_model_config(
        name="gpt-4-variant-a",
        provider=ProviderType.OPENAI,
        temperature=0.3,
        max_tokens=2048,
        top_p=0.9,
        metadata={
            "variant": "A",
            "strategy": "conservative",
            "test_group": "control"
        }
    )
    config.add_model(variant_a)
    
    # Variant B: Experimental
    variant_b = create_model_config(
        name="gpt-4-variant-b",
        provider=ProviderType.OPENAI,
        temperature=0.8,
        max_tokens=2048,
        top_p=0.95,
        presence_penalty=0.5,
        metadata={
            "variant": "B",
            "strategy": "experimental",
            "test_group": "treatment"
        }
    )
    config.add_model(variant_b)
    
    print("A/B Test Configuration:")
    print(f"\nVariant A (Control):")
    print(f"  Temperature: {variant_a.temperature}")
    print(f"  Top-p: {variant_a.top_p}")
    print(f"  Strategy: {variant_a.metadata['strategy']}")
    
    print(f"\nVariant B (Treatment):")
    print(f"  Temperature: {variant_b.temperature}")
    print(f"  Top-p: {variant_b.top_p}")
    print(f"  Presence penalty: {variant_b.presence_penalty}")
    print(f"  Strategy: {variant_b.metadata['strategy']}")
    
    # Step 4: Compare models for use case selection
    print("\nStep 4: Use Case Model Comparison")
    print("-"*80)
    
    use_cases = [
        ("creative_writing", "gpt-4-creative"),
        ("code_generation", "gpt-4-code"),
        ("data_analysis", "gpt-4-analysis"),
        ("conversation", "gpt-4-chat")
    ]
    
    print("\nRecommended models by use case:")
    for use_case, model_name in use_cases:
        model = config.get_model(model_name)
        if model:
            print(f"\n{use_case.replace('_', ' ').title()}:")
            print(f"  Model: {model.name}")
            print(f"  Temperature: {model.temperature}")
            print(f"  Max tokens: {model.max_tokens}")
            print(f"  Penalties: presence={model.presence_penalty}, frequency={model.frequency_penalty}")
    
    # Step 5: Dynamic parameter updates
    print("\nStep 5: Dynamic Parameter Updates")
    print("-"*80)
    
    print("\nScenario: Adjusting model for production deployment")
    
    dev_model = config.get_model("gpt-4-code")
    print(f"Development settings:")
    print(f"  Temperature: {dev_model.temperature}")
    print(f"  Timeout: {dev_model.timeout}s")
    
    # Update for production
    prod_model_config = config.get_model("gpt-4-code")
    prod_model_config.temperature = 0.2  # More deterministic
    prod_model_config.timeout = 90.0      # Longer timeout
    prod_model_config.max_retries = 5     # More retries
    config.add_model(prod_model_config)
    
    prod_model = config.get_model("gpt-4-code")
    print(f"\nProduction settings:")
    print(f"  Temperature: {prod_model.temperature}")
    print(f"  Timeout: {prod_model.timeout}s")
    print(f"  Max retries: {prod_model.max_retries}")
    
    # Step 6: Cost optimization through parameter tuning
    print("\nStep 6: Cost Optimization Strategy")
    print("-"*80)
    
    cost_tiers = [
        {
            "name": "gpt-4-premium",
            "max_tokens": 4096,
            "temperature": 0.7,
            "tier": "premium",
            "use": "Complex tasks requiring high quality"
        },
        {
            "name": "gpt-4-standard",
            "max_tokens": 2048,
            "temperature": 0.5,
            "tier": "standard",
            "use": "Regular tasks with good quality"
        },
        {
            "name": "gpt-3.5-economy",
            "max_tokens": 1024,
            "temperature": 0.5,
            "tier": "economy",
            "use": "Simple tasks where cost matters"
        }
    ]
    
    print("Cost-optimized model tiers:")
    for tier in cost_tiers:
        model = create_model_config(
            name=tier["name"],
            provider=ProviderType.OPENAI,
            max_tokens=tier["max_tokens"],
            temperature=tier["temperature"],
            metadata={"cost_tier": tier["tier"], "use_case": tier["use"]}
        )
        config.add_model(model)
        print(f"\n{tier['tier'].upper()} Tier ({tier['name']}):")
        print(f"  Max tokens: {tier['max_tokens']}")
        print(f"  Use case: {tier['use']}")
    
    # Step 7: Model configuration summary
    print("\nStep 7: Configuration Summary")
    print("-"*80)
    
    all_models = config.list_models()
    print(f"\nTotal models configured: {len(all_models)}")
    
    # Group by temperature ranges
    temp_groups = {
        "deterministic": [],
        "balanced": [],
        "creative": []
    }
    
    for model_name in all_models:
        model = config.get_model(model_name)
        if model.temperature <= 0.3:
            temp_groups["deterministic"].append(model_name)
        elif model.temperature <= 0.7:
            temp_groups["balanced"].append(model_name)
        else:
            temp_groups["creative"].append(model_name)
    
    print("\nModels by temperature profile:")
    for group, models in temp_groups.items():
        print(f"\n{group.capitalize()} (temp range):")
        for model_name in models:
            model = config.get_model(model_name)
            print(f"  - {model_name}: temp={model.temperature}")
    
    print("\n" + "="*80)
    print("Model parameter optimization completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
