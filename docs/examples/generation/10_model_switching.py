"""
Model Switching and Universal Wrapper Example
=============================================

This example showcases the power of the universal wrapper pattern,
demonstrating how easily you can switch between different models and providers.

Main concepts:
- Creating a Generator instance as a universal wrapper
- Seamlessly switching between models
- Temporary model overrides with context managers
- Comparing multiple models with a single call
- Managing API keys for multiple providers
- Tracking model usage history
"""

from kerb.generation import Generator, ModelName
from kerb.core import Message
from kerb.core.types import MessageRole


def example_basic_switching():
    """Basic model switching demonstration."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Model Switching")
    print("="*80)
    
    # Create a universal generator
    gen = Generator(model=ModelName.GPT_4O_MINI)
    
    prompt = "What is polymorphism in programming?"
    
    # Use the default model
    print(f"\n1. Using default model: {gen.get_current_model()}")
    response1 = gen.generate(prompt)
    print(f"   Response: {response1.content[:80]}...")
    print(f"   Cost: ${response1.cost:.6f}")
    
    # Switch to a different model
    gen.set_model(ModelName.CLAUDE_35_HAIKU)
    print(f"\n2. Switched to: {gen.get_current_model()}")
    try:
        response2 = gen.generate(prompt)
        print(f"   Response: {response2.content[:80]}...")
        print(f"   Cost: ${response2.cost:.6f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Switch again
    gen.set_model(ModelName.GEMINI_15_FLASH)
    print(f"\n3. Switched to: {gen.get_current_model()}")
    try:
        response3 = gen.generate(prompt)
        print(f"   Response: {response3.content[:80]}...")
        print(f"   Cost: ${response3.cost:.6f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Check history
    history = gen.get_model_history()
    print(f"\n4. Model switch history: {[str(m) for m in history]}")


def example_context_manager():
    """Demonstrate temporary model switching with context manager."""

# %%
# Setup and Imports
# -----------------
    print("\n" + "="*80)
    print("EXAMPLE 2: Temporary Model Switch with Context Manager")
    print("="*80)
    
    gen = Generator(model=ModelName.GPT_4O_MINI)
    
    prompt = "Explain REST APIs in one sentence."
    
    print(f"\nDefault model: {gen.get_current_model()}")
    response1 = gen.generate(prompt)
    print(f"Response: {response1.content}")
    
    # Temporary switch for a specific task
    print(f"\nTemporarily switching to Claude...")
    try:
        with gen.using_model(ModelName.CLAUDE_35_HAIKU):
            print(f"Current model in context: {gen.get_current_model()}")
            response2 = gen.generate(prompt)
            print(f"Response: {response2.content}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Automatically reverted back
    print(f"\nBack to default model: {gen.get_current_model()}")
    response3 = gen.generate(prompt)
    print(f"Response: {response3.content}")



# %%
# Example Model Override
# ----------------------

def example_model_override():
    """Demonstrate per-call model override."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Per-Call Model Override")
    print("="*80)
    
    # Set up generator with a default model
    gen = Generator(model=ModelName.GPT_4O_MINI, temperature=0.7)
    
    prompt = "Write a Python function to reverse a string."
    
    # Use default model
    print(f"\n1. Using default model ({gen.get_current_model()}):")
    response1 = gen.generate(prompt)
    print(f"   {response1.content[:100]}...")
    
    # Override for this specific call
    print(f"\n2. Override with Claude (default stays {gen.get_current_model()}):")
    try:
        response2 = gen.generate(prompt, model=ModelName.CLAUDE_35_HAIKU)
        print(f"   {response2.content[:100]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Confirm default model unchanged
    print(f"\n3. Confirm default still: {gen.get_current_model()}")
    response3 = gen.generate("Hello!")
    print(f"   {response3.content[:50]}...")


def example_compare_models():
    """Compare responses across multiple models effortlessly."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Easy Multi-Model Comparison")
    print("="*80)
    
    gen = Generator()  # No default model needed for comparison
    
    prompt = "What are the benefits of static typing?"
    
    print(f"\nPrompt: {prompt}\n")
    
    # Compare across multiple models with ONE call!
    results = gen.compare_models(
        prompt,
        models=[
            ModelName.GPT_4O_MINI,
            ModelName.GPT_35_TURBO,
            ModelName.CLAUDE_35_HAIKU,
            ModelName.GEMINI_15_FLASH,
        ],
        temperature=0.7
    )
    
    print("Results:")
    for model, response in results.items():
        if isinstance(response, dict) and not response.get("success", True):
            print(f"\n{model}: ❌ {response.get('error', 'Unknown error')}")
        else:
            print(f"\n{model}: ✓")
            print(f"  Cost: ${response.cost:.6f}")
            print(f"  Tokens: {response.usage.total_tokens}")
            print(f"  Response: {response.content[:80]}...")



# %%
# Example Model Comparator
# ------------------------

def example_model_comparator():
    """Use Generator's compare_models for advanced comparison features."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Advanced Comparison with Generator")
    print("="*80)
    
    gen = Generator()
    
    prompt = "Explain the observer pattern in software design."
    
    results = gen.compare_models(
        prompt,
        models=[
            ModelName.GPT_4O_MINI,
            ModelName.CLAUDE_35_HAIKU,
        ],
        temperature=0.5,
        max_tokens=200
    )
    
    # Print comparison results
    print(f"\nComparing {len(results)} models:\n")
    for model, response in results.items():
        if isinstance(response, dict) and not response.get("success", True):
            print(f"{model}: ❌ {response.get('error', 'Unknown error')}")
        else:
            print(f"{model}:")
            print(f"  Cost: ${response.cost:.6f}")
            print(f"  Tokens: {response.usage.total_tokens}")
            print(f"  Response (first 150 chars): {response.content[:150]}...")
            print()


def example_multiple_api_keys():
    """Manage multiple provider API keys."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Managing Multiple Provider API Keys")
    print("="*80)
    
    # Set up generator with API keys for multiple providers
    gen = Generator(
        api_keys={
            "openai": "sk-...",  # Your OpenAI key
            "anthropic": "sk-ant-...",  # Your Anthropic key
            "google": "...",  # Your Google key
        }
    )
    
    print("\nGenerator configured with API keys for:")
    print("  - OpenAI")
    print("  - Anthropic")
    print("  - Google")
    
    print("\nYou can now seamlessly switch between these providers!")
    print("The Generator will automatically use the correct API key.")



# %%
# Example Configuration Management
# --------------------------------

def example_configuration_management():
    """Manage default configurations across model switches."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Configuration Management")
    print("="*80)
    
    # Create generator with default configuration
    gen = Generator(
        model=ModelName.GPT_4O_MINI,
        temperature=0.8,
        max_tokens=150,
        top_p=0.9
    )
    
    prompt = "Generate a creative name for a code linter."
    
    print("\nUsing default configuration:")
    print("  Temperature: 0.8")
    print("  Max Tokens: 150")
    
    response1 = gen.generate(prompt)
    print(f"\nGenerated: {response1.content}")
    
    # Override configuration for specific call
    print("\nOverriding temperature for this call:")
    response2 = gen.generate(prompt, temperature=0.3)
    print(f"\nGenerated (more conservative): {response2.content}")
    
    # Default config persists
    print("\nDefault configuration still applies to subsequent calls")
    response3 = gen.generate(prompt)
    print(f"\nGenerated: {response3.content}")


def example_practical_workflow():
    """Demonstrate a practical workflow with model switching."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Practical Workflow - Cheap then Expensive")
    print("="*80)
    
    gen = Generator()
    
    # Start with a cheaper model for initial draft
    print("\n1. Use GPT-4o-mini for initial draft (cheaper):")
    gen.set_model(ModelName.GPT_4O_MINI)
    
    prompt = "Write a docstring for a function that merges sorted lists."
    draft = gen.generate(prompt)
    print(f"   Draft: {draft.content[:100]}...")
    print(f"   Cost: ${draft.cost:.6f}")
    
    # Switch to more powerful model for refinement
    print("\n2. Switch to Claude for refinement (better quality):")
    gen.set_model(ModelName.CLAUDE_35_HAIKU)
    
    refine_prompt = f"Improve this docstring:\n{draft.content}"
    try:
        refined = gen.generate(refine_prompt, max_tokens=200)
        print(f"   Refined: {refined.content[:100]}...")
        print(f"   Cost: ${refined.cost:.6f}")
        
        total_cost = draft.cost + refined.cost
        print(f"\n3. Total cost: ${total_cost:.6f}")
        print("   By using cheaper model first, we optimized costs!")
    except Exception as e:
        print(f"   Error: {e}")



# %%
# Example Cost Tracking
# ---------------------

def example_cost_tracking():
    """Track costs across model switches."""
    print("\n" + "="*80)
    print("EXAMPLE 9: Cost Tracking Across Models")
    print("="*80)
    
    gen = Generator(model=ModelName.GPT_4O_MINI)
    
    prompts = [
        "What is async/await?",
        "Explain closures in JavaScript.",
        "What are decorators in Python?",
    ]
    
    # Use different models for different prompts
    models = [
        ModelName.GPT_4O_MINI,
        ModelName.CLAUDE_35_HAIKU,
        ModelName.GEMINI_15_FLASH,
    ]
    
    for prompt, model in zip(prompts, models):
        gen.set_model(model)
        print(f"\nUsing {model.value}:")
        try:
            response = gen.generate(prompt)
            print(f"  Prompt: {prompt}")
            print(f"  Cost: ${response.cost:.6f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Get cost summary
    summary = gen.get_cost_summary()
    print(f"\n{'='*40}")
    print("Cost Summary:")
    print(f"  Total Requests: {summary['total_requests']}")
    print(f"  Total Cost: ${summary['total_cost']:.6f}")


def main():
    """Run all model switching examples."""
    print("\n" + "#"*80)
    print("# MODEL SWITCHING & UNIVERSAL WRAPPER EXAMPLES")
    print("# Demonstrating Seamless Model Management")
    print("#"*80)
    
    examples = [
        ("Basic Switching", example_basic_switching),
        ("Context Manager", example_context_manager),
        ("Model Override", example_model_override),
        ("Compare Models", example_compare_models),
        ("Advanced Comparison", example_model_comparator),
        ("Multiple API Keys", example_multiple_api_keys),
        ("Configuration Management", example_configuration_management),
        ("Practical Workflow", example_practical_workflow),
        ("Cost Tracking", example_cost_tracking),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n{name} Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "#"*80)
    print("# Examples completed")
    print("# The universal wrapper makes model switching effortless!")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
