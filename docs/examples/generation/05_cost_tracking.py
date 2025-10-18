"""Cost Tracking Example

This example demonstrates tracking and analyzing LLM API costs.

Main concepts:
- Using CostTracker to monitor spending
- Analyzing costs by model and provider
- Comparing cost efficiency
- Setting budget alerts
- Global vs per-session cost tracking
"""

from kerb.generation import generate, generate_batch, ModelName
from kerb.generation.utils import CostTracker


def example_basic_cost_tracking():
    """Track costs for a series of LLM calls."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Cost Tracking")
    print("="*80)
    
    # Create a cost tracker
    tracker = CostTracker()
    
    prompts = [
        "What is Python?",
        "Explain list comprehensions.",
        "What are decorators?",
    ]
    
    print("\nGenerating responses and tracking costs...\n")
    
    for i, prompt in enumerate(prompts, 1):
        response = generate(
            prompt,
            model=ModelName.GPT_4O_MINI,
            cost_tracker=tracker
        )
        print(f"{i}. {prompt}")
        print(f"   Tokens: {response.usage.total_tokens}, Cost: ${response.cost:.6f}")
    
    # Get cost summary
    summary = tracker.get_summary()
    
    print("\n" + "-"*80)
    print("COST SUMMARY")
    print("-"*80)
    print(f"Total cost: ${summary['total_cost']:.6f}")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Total requests: {summary['total_requests']}")


def example_multi_model_tracking():
    """Track costs across multiple models."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multi-Model Cost Tracking")
    print("="*80)
    
    tracker = CostTracker()
    
    prompt = "Explain async/await in Python."
    
    models = [
        ModelName.GPT_4O_MINI,
        ModelName.GPT_35_TURBO,
        ModelName.CLAUDE_35_HAIKU,
    ]
    
    print(f"\nPrompt: {prompt}")
    print("\nTesting across models...\n")
    
    for model in models:
        try:
            response = generate(
                prompt,
                model=model,
                cost_tracker=tracker,
                max_tokens=100
            )
            print(f"{model.value}:")
            print(f"  Tokens: {response.usage.total_tokens}")
            print(f"  Cost: ${response.cost:.6f}\n")
        except Exception as e:
            print(f"{model.value}: Error - {e}\n")
    
    # Detailed summary
    summary = tracker.get_summary()
    
    print("-"*80)
    print("BREAKDOWN BY MODEL")
    print("-"*80)
    
    for model, cost in summary['cost_by_model'].items():
        requests = summary['requests_by_model'][model]
        tokens = summary['tokens_by_model'][model]
        avg_cost = cost / requests if requests > 0 else 0
        
        print(f"{model}:")
        print(f"  Requests: {requests}")
        print(f"  Total cost: ${cost:.6f}")
        print(f"  Avg cost/request: ${avg_cost:.6f}")
        print(f"  Total tokens: {tokens:,}\n")
    
    print(f"Grand Total: ${summary['total_cost']:.6f}")


def example_batch_cost_tracking():
    """Track costs for batch operations."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Operation Cost Tracking")
    print("="*80)
    
    tracker = CostTracker()
    
    # Generate multiple documentation strings
    functions = [
        "calculate_fibonacci",
        "merge_sort_algorithm",
        "binary_search_tree",
        "validate_email",
        "parse_json_data",
    ]
    
    prompts = [
        f"Write a one-line docstring for a function named {func}"
        for func in functions
    ]
    
    print(f"\nGenerating {len(prompts)} docstrings in batch...\n")
    
    responses = generate_batch(
        prompts,
        model=ModelName.GPT_4O_MINI,
        cost_tracker=tracker,
        max_concurrent=3,
        max_tokens=50
    )
    
    print(f"Generated {len(responses)} responses")
    
    # Analyze cost efficiency
    summary = tracker.get_summary()
    avg_cost_per_request = summary['total_cost'] / len(responses)
    avg_tokens_per_request = summary['total_tokens'] / len(responses)
    
    print("\n" + "-"*80)
    print("BATCH COST ANALYSIS")
    print("-"*80)
    print(f"Total cost: ${summary['total_cost']:.6f}")
    print(f"Cost per docstring: ${avg_cost_per_request:.6f}")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Avg tokens per docstring: {avg_tokens_per_request:.1f}")


def example_cost_comparison():
    """Compare costs between different approaches."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Cost Comparison Between Models")
    print("="*80)
    
    task = "List 3 Python testing best practices."
    
    # Test with expensive model
    tracker_expensive = CostTracker()
    try:
        response_expensive = generate(
            task,
            model=ModelName.GPT_4O,
            cost_tracker=tracker_expensive,
            max_tokens=150
        )
        expensive_cost = tracker_expensive.total_cost
        expensive_success = True
    except:
        expensive_cost = 0
        expensive_success = False
    
    # Test with cheap model
    tracker_cheap = CostTracker()
    try:
        response_cheap = generate(
            task,
            model=ModelName.GPT_4O_MINI,
            cost_tracker=tracker_cheap,
            max_tokens=150
        )
        cheap_cost = tracker_cheap.total_cost
        cheap_success = True
    except:
        cheap_cost = 0
        cheap_success = False
    
    print(f"\nTask: {task}\n")
    
    if expensive_success:
        print(f"GPT-4o (Premium Model):")
        print(f"  Cost: ${expensive_cost:.6f}")
        print(f"  Tokens: {tracker_expensive.total_tokens}")
    
    if cheap_success:
        print(f"\nGPT-4o-mini (Economy Model):")
        print(f"  Cost: ${cheap_cost:.6f}")
        print(f"  Tokens: {tracker_cheap.total_tokens}")
    
    if expensive_success and cheap_success:
        savings = expensive_cost - cheap_cost
        savings_pct = (savings / expensive_cost * 100) if expensive_cost > 0 else 0
        print(f"\nCost Savings:")
        print(f"  Difference: ${savings:.6f}")
        print(f"  Savings: {savings_pct:.1f}%")


def example_budget_monitoring():
    """Monitor costs against a budget."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Budget Monitoring")
    print("="*80)
    
    tracker = CostTracker()
    budget = 0.01  # $0.01 budget
    
    print(f"\nBudget: ${budget:.4f}")
    print("Generating responses...\n")
    
    prompts = [
        "What is a variable?",
        "What is a function?",
        "What is a class?",
        "What is a module?",
        "What is a package?",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        # Check budget before request
        if tracker.total_cost >= budget:
            print(f"\nBudget exceeded after {i-1} requests!")
            print(f"Total spent: ${tracker.total_cost:.6f}")
            break
        
        response = generate(
            prompt,
            model=ModelName.GPT_4O_MINI,
            cost_tracker=tracker,
            max_tokens=30
        )
        
        remaining = budget - tracker.total_cost
        print(f"{i}. {prompt}")
        print(f"   Cost: ${response.cost:.6f}, Remaining: ${remaining:.6f}")
    
    else:
        print(f"\nAll requests completed within budget!")
        print(f"Total spent: ${tracker.total_cost:.6f}")
        print(f"Remaining: ${budget - tracker.total_cost:.6f}")


def example_cost_optimization():
    """Demonstrate cost optimization strategies."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Cost Optimization Strategies")
    print("="*80)
    
    prompt = "Explain Python decorators."
    
    strategies = [
        {
            "name": "High Quality",
            "model": ModelName.GPT_4O,
            "max_tokens": 200,
            "temperature": 0.7
        },
        {
            "name": "Balanced",
            "model": ModelName.GPT_4O_MINI,
            "max_tokens": 150,
            "temperature": 0.7
        },
        {
            "name": "Economy",
            "model": ModelName.GPT_35_TURBO,
            "max_tokens": 100,
            "temperature": 0.5
        },
    ]
    
    print(f"\nPrompt: {prompt}\n")
    print("Testing cost optimization strategies...\n")
    
    results = []
    
    for strategy in strategies:
        tracker = CostTracker()
        
        try:
            response = generate(
                prompt,
                model=strategy["model"],
                cost_tracker=tracker,
                max_tokens=strategy["max_tokens"],
                temperature=strategy["temperature"]
            )
            
            results.append({
                "name": strategy["name"],
                "cost": tracker.total_cost,
                "tokens": response.usage.total_tokens,
                "response_length": len(response.content)
            })
        except Exception as e:
            print(f"{strategy['name']}: Error - {e}")
    
    # Display results
    print("-"*80)
    print(f"{'Strategy':<15} {'Cost':<12} {'Tokens':<10} {'Response Length'}")
    print("-"*80)
    
    for result in results:
        print(f"{result['name']:<15} ${result['cost']:<11.6f} {result['tokens']:<10} {result['response_length']} chars")
    
    if results:
        cheapest = min(results, key=lambda x: x['cost'])
        print(f"\nMost cost-effective: {cheapest['name']} at ${cheapest['cost']:.6f}")


def main():
    """Run all cost tracking examples."""
    print("\n" + "#"*80)
    print("# COST TRACKING EXAMPLES")
    print("#"*80)
    
    try:
        example_basic_cost_tracking()
    except Exception as e:
        print(f"\nExample 1 Error: {e}")
    
    try:
        example_multi_model_tracking()
    except Exception as e:
        print(f"\nExample 2 Error: {e}")
    
    try:
        example_batch_cost_tracking()
    except Exception as e:
        print(f"\nExample 3 Error: {e}")
    
    try:
        example_cost_comparison()
    except Exception as e:
        print(f"\nExample 4 Error: {e}")
    
    try:
        example_budget_monitoring()
    except Exception as e:
        print(f"\nExample 5 Error: {e}")
    
    try:
        example_cost_optimization()
    except Exception as e:
        print(f"\nExample 6 Error: {e}")
    
    print("\n" + "#"*80)
    print("# Examples completed")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
