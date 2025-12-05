"""
Multi-Provider Comparison Example
=================================

This example demonstrates comparing responses across different LLM providers.

Main concepts:
- Testing same prompt across multiple providers
- Comparing response quality and characteristics
- Measuring latency and cost differences
- Using ModelName enum for type-safe model names
- Using LLMProvider enum for routing
- Provider-specific capabilities
"""

import time
from typing import List, Dict, Any
from kerb.generation import generate, ModelName, LLMProvider
from kerb.generation.config import GenerationResponse


def compare_providers_simple(prompt: str, models: List[ModelName]) -> Dict[str, Any]:
    """Compare response from multiple providers for a single prompt.
    
    Args:
        prompt: The prompt to test
        models: List of models to compare
        
    Returns:
        Dictionary with comparison results
    """

# %%
# Setup and Imports
# -----------------
    results = {}
    
    for model in models:
        try:
            start = time.time()
            response = generate(prompt, model=model, temperature=0.7)
            elapsed = time.time() - start
            
            results[model.value] = {
                "success": True,
                "response": response,
                "elapsed": elapsed
            }
        except Exception as e:
            results[model.value] = {
                "success": False,
                "error": str(e),
                "elapsed": 0
            }
    
    return results



# %%
# Example Basic Comparison
# ------------------------

def example_basic_comparison():
    """Compare a simple prompt across providers."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Provider Comparison")
    print("="*80)
    
    prompt = "Explain dependency injection in software engineering."
    
    models = [
        ModelName.GPT_4O_MINI,
        ModelName.CLAUDE_35_HAIKU,
        ModelName.GEMINI_15_FLASH,
    ]
    
    print(f"\nPrompt: {prompt}")
    print("\nComparing responses from multiple providers...\n")
    
    results = compare_providers_simple(prompt, models)
    
    for model_name, result in results.items():
        print(f"{model_name}:")
        if result["success"]:
            response = result["response"]
            print(f"  Provider: {response.provider.value}")
            print(f"  Latency: {result['elapsed']:.3f}s")
            print(f"  Tokens: {response.usage.total_tokens}")
            print(f"  Cost: ${response.cost:.6f}")
            print(f"  Response: {response.content[:100]}...\n")
        else:
            print(f"  Error: {result['error']}\n")


def example_cost_comparison():
    """Compare costs across providers for the same task."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Cost Comparison")
    print("="*80)
    
    prompt = "List the top 5 Python web frameworks with one-line descriptions."
    
    models = [
        ModelName.GPT_4O_MINI,
        ModelName.GPT_35_TURBO,
        ModelName.CLAUDE_35_HAIKU,
        ModelName.GEMINI_15_FLASH,
    ]
    
    print(f"\nTask: {prompt}\n")
    
    results = compare_providers_simple(prompt, models)
    
    # Sort by cost
    successful_results = [
        (name, result) for name, result in results.items() 
        if result["success"]
    ]
    successful_results.sort(key=lambda x: x[1]["response"].cost)
    
    print("Cost Ranking (cheapest to most expensive):\n")
    
    for i, (model_name, result) in enumerate(successful_results, 1):
        response = result["response"]
        print(f"{i}. {model_name}")
        print(f"   Cost: ${response.cost:.6f}")
        print(f"   Tokens: {response.usage.total_tokens}")
        print(f"   Cost per 1K tokens: ${(response.cost / response.usage.total_tokens * 1000):.6f}\n")



# %%
# Example Latency Comparison
# --------------------------

def example_latency_comparison():
    """Compare response latency across providers."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Latency Comparison")
    print("="*80)
    
    prompt = "What is REST?"
    
    models = [
        ModelName.GPT_4O_MINI,
        ModelName.CLAUDE_35_HAIKU,
        ModelName.GEMINI_15_FLASH,
    ]
    
    print(f"\nPrompt: {prompt}")
    print("\nMeasuring response times...\n")
    
    # Run multiple times for more accurate measurement
    num_runs = 3
    all_results = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...", end="\r")
        results = compare_providers_simple(prompt, models)
        all_results.append(results)
    
    print(" " * 40)  # Clear line
    
    # Calculate average latencies
    avg_latencies = {}
    for model in models:
        latencies = [
            results[model.value]["elapsed"] 
            for results in all_results 
            if results[model.value]["success"]
        ]
        if latencies:
            avg_latencies[model.value] = sum(latencies) / len(latencies)
    
    # Sort by latency
    sorted_by_latency = sorted(avg_latencies.items(), key=lambda x: x[1])
    
    print("Average Latency Ranking (fastest to slowest):\n")
    
    for i, (model_name, avg_latency) in enumerate(sorted_by_latency, 1):
        print(f"{i}. {model_name}: {avg_latency:.3f}s")


def example_quality_comparison():
    """Compare response quality for a technical question."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Response Quality Comparison")
    print("="*80)
    
    prompt = "Explain the difference between concurrency and parallelism with examples."
    
    models = [
        ModelName.GPT_4O_MINI,
        ModelName.CLAUDE_35_HAIKU,
    ]
    
    print(f"\nPrompt: {prompt}\n")
    
    results = compare_providers_simple(prompt, models)
    
    for model_name, result in results.items():
        if result["success"]:
            response = result["response"]
            print(f"\n{'-'*80}")
            print(f"Model: {model_name}")
            print(f"Provider: {response.provider.value}")
            print(f"{'-'*80}")
            print(response.content)
            print(f"\nTokens: {response.usage.total_tokens}, Cost: ${response.cost:.6f}")



# %%
# Example Temperature Comparison
# ------------------------------

def example_temperature_comparison():
    """Compare how different providers respond to temperature settings."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Temperature Sensitivity Comparison")
    print("="*80)
    
    prompt = "Generate a creative name for a Python package manager."
    
    models = [
        ModelName.GPT_4O_MINI,
        ModelName.CLAUDE_35_HAIKU,
    ]
    
    temperatures = [0.3, 0.7, 1.0]
    
    print(f"\nPrompt: {prompt}\n")
    
    for model in models:
        print(f"\n{model.value}:")
        for temp in temperatures:
            try:
                response = generate(prompt, model=model, temperature=temp)
                print(f"  Temp {temp}: {response.content}")
            except Exception as e:
                print(f"  Temp {temp}: Error - {e}")


def example_model_capabilities():
    """Compare provider-specific capabilities."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Model Capabilities Overview")
    print("="*80)
    
    model_info = [
        {
            "model": ModelName.GPT_4O_MINI,
            "provider": "OpenAI",
            "strengths": ["General purpose", "Function calling", "JSON mode"],
            "max_tokens": "128K context"
        },
        {
            "model": ModelName.CLAUDE_35_HAIKU,
            "provider": "Anthropic",
            "strengths": ["Fast responses", "Good reasoning", "Long context"],
            "max_tokens": "200K context"
        },
        {
            "model": ModelName.GEMINI_15_FLASH,
            "provider": "Google",
            "strengths": ["Cost effective", "Fast", "Multimodal"],
            "max_tokens": "1M context"
        },
    ]
    
    print("\nModel Capabilities Comparison:\n")
    
    for info in model_info:
        print(f"{info['model'].value} ({info['provider']}):")
        print(f"  Context: {info['max_tokens']}")
        print(f"  Strengths: {', '.join(info['strengths'])}")
        
        # Test with a simple prompt
        try:
            response = generate(
                "Hello, how are you?",
                model=info["model"],
                max_tokens=20
            )
            print(f"  Test response: {response.content}")
            print(f"  Cost: ${response.cost:.6f}\n")
        except Exception as e:
            print(f"  Error: {e}\n")



# %%
# Main
# ----

def main():
    """Run all multi-provider comparison examples."""
    print("\n" + "#"*80)
    print("# MULTI-PROVIDER COMPARISON EXAMPLES")
    print("#"*80)
    
    try:
        example_basic_comparison()
    except Exception as e:
        print(f"\nExample 1 Error: {e}")
    
    try:
        example_cost_comparison()
    except Exception as e:
        print(f"\nExample 2 Error: {e}")
    
    try:
        example_latency_comparison()
    except Exception as e:
        print(f"\nExample 3 Error: {e}")
    
    try:
        example_quality_comparison()
    except Exception as e:
        print(f"\nExample 4 Error: {e}")
    
    try:
        example_temperature_comparison()
    except Exception as e:
        print(f"\nExample 5 Error: {e}")
    
    try:
        example_model_capabilities()
    except Exception as e:
        print(f"\nExample 6 Error: {e}")
    
    print("\n" + "#"*80)
    print("# Examples completed")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
