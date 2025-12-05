"""
Model Comparison and Selection.
===============================

This example demonstrates how to systematically compare multiple LLM models
to select the best one for your use case. Common use cases:
- Comparing different model providers (OpenAI, Anthropic, etc.)
- Selecting between model versions (GPT-3.5 vs GPT-4)
- Evaluating open-source vs proprietary models
- Cost-performance tradeoffs
"""

from kerb.evaluation import (
    compare_outputs,
    run_benchmark,
    TestCase,
    calculate_f1_score,
    calculate_bleu,
    assess_coherence,
)


class ModelSimulator:
    """Simulate different LLM models for comparison."""
    
    @staticmethod
    def gpt4_style(prompt: str) -> str:
        """Simulate GPT-4 style responses (detailed, accurate)."""

# %%
# Setup and Imports
# -----------------
        responses = {
            "what is python": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability with significant indentation and supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "explain ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines programmed to think and learn. It encompasses various subfields including machine learning, natural language processing, computer vision, and robotics.",
            "summarize ml": "Machine Learning is a subset of AI that enables systems to automatically learn and improve from experience without explicit programming. It uses algorithms to identify patterns in data and make predictions or decisions.",
        }
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        return "I can provide detailed information on various topics."
    
    @staticmethod

# %%
# Gpt35 Style
# -----------

    def gpt35_style(prompt: str) -> str:
        """Simulate GPT-3.5 style responses (good but less detailed)."""
        responses = {
            "what is python": "Python is a programming language known for its simplicity and readability. It's widely used in web development, data science, and automation.",
            "explain ai": "AI is the simulation of human intelligence by machines. It includes learning, reasoning, and problem-solving capabilities.",
            "summarize ml": "Machine learning is a branch of AI where systems learn from data to make predictions and decisions.",
        }
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        return "I can help with various questions."
    
    @staticmethod
    def claude_style(prompt: str) -> str:
        """Simulate Claude style responses (conversational, nuanced)."""
        responses = {
            "what is python": "Python is a versatile programming language that's particularly valued for its clean, readable syntax. Created in 1991, it has become one of the most popular languages for beginners and experts alike, especially in fields like data science and web development.",
            "explain ai": "Artificial Intelligence represents our effort to create machines that can perform tasks requiring human-like intelligence. This broad field spans from simple rule-based systems to complex neural networks that can learn and adapt.",
            "summarize ml": "Machine learning is fundamentally about enabling computers to learn from experience. Rather than following explicit instructions, ML systems identify patterns in data and use them to make informed decisions about new situations.",
        }
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        return "I'd be happy to explore this topic with you."



# %%
# Compare Model Accuracy
# ----------------------

def compare_model_accuracy():
    """Compare accuracy of different models on factual questions."""
    print("=" * 80)
    print("MODEL ACCURACY COMPARISON")
    print("=" * 80)
    
    test_cases = [
        TestCase(
            id="fact_1",
            input="What is Python?",
            expected_output="Python is a programming language"
        ),
        TestCase(
            id="fact_2",
            input="Explain AI",
            expected_output="Artificial Intelligence simulates human intelligence"
        ),
        TestCase(
            id="fact_3",
            input="Summarize ML",
            expected_output="Machine learning enables systems to learn from data"
        ),
    ]
    
    models = {
        "GPT-4": ModelSimulator.gpt4_style,
        "GPT-3.5": ModelSimulator.gpt35_style,
        "Claude": ModelSimulator.claude_style,
    }
    
    print(f"\nComparing {len(models)} models on {len(test_cases)} test cases...")
    print("Metric: F1 Score (token overlap with expected answer)\n")
    
    results = {}
    
    for model_name, model_fn in models.items():
        print(f"Evaluating {model_name}...")
        
        def evaluator(output: str, expected: str) -> float:
            return calculate_f1_score(output, expected)
        
        result = run_benchmark(
            test_cases,
            model_fn,
            evaluator,
            threshold=0.3,
            name=model_name
        )
        
        results[model_name] = result
    
    print("\n" + "-" * 80)
    print("Accuracy Comparison:")
    print("-" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].average_score, reverse=True)
    
    for rank, (model_name, result) in enumerate(sorted_results, 1):
        print(f"{rank}. {model_name}")
        print(f"   Average F1 Score: {result.average_score:.3f}")
        print(f"   Pass Rate: {result.pass_rate:.1f}%")
        print(f"   Passed: {result.passed_tests}/{result.total_tests}")
        print()


def compare_response_quality():
    """Compare response quality across multiple dimensions."""
    print("=" * 80)
    print("MULTI-DIMENSIONAL QUALITY COMPARISON")
    print("=" * 80)
    
    prompt = "Explain how neural networks work"
    
    responses = {
        "GPT-4": "Neural networks are computational models inspired by biological neural networks in the human brain. They consist of interconnected layers of nodes (neurons), each processing and transforming input data. Through training on datasets, the network adjusts connection weights to minimize prediction errors, enabling it to learn complex patterns and make accurate predictions on new data.",
        
        "GPT-3.5": "Neural networks are models inspired by the brain. They have layers of nodes that process data. During training, the network learns by adjusting weights to improve predictions.",
        
        "Claude": "Neural networks represent an interesting approach to machine learning that takes inspiration from how our brains work. Think of them as layers of interconnected processing units, where each layer transforms the input data in increasingly abstract ways. Through training, these networks learn to recognize patterns by adjusting the strength of connections between units.",
    }
    
    print(f"\nPrompt: {prompt}\n")
    print("-" * 80)
    
    evaluation_criteria = {
        "Completeness": lambda text: len(text.split()) / 50.0,  # Longer = more complete
        "Clarity": lambda text: 1.0 if 10 <= len(text.split()) <= 100 else 0.7,
        "Detail": lambda text: min(len(text.split()) / 60.0, 1.0),
    }
    
    scores_by_model = {model: {} for model in responses.keys()}
    
    for criterion_name, criterion_fn in evaluation_criteria.items():
        print(f"\n{criterion_name}:")
        for model, response in responses.items():
            score = criterion_fn(response)
            scores_by_model[model][criterion_name] = score
            print(f"  {model}: {score:.3f}")
    
    print("\n" + "=" * 80)
    print("Overall Scores:")
    print("=" * 80)
    
    for model, scores in scores_by_model.items():
        avg_score = sum(scores.values()) / len(scores)
        print(f"{model}: {avg_score:.3f}")
    
    best_model = max(scores_by_model.items(), key=lambda x: sum(x[1].values()) / len(x[1]))
    print(f"\nBest Overall: {best_model[0]}")



# %%
# Compare Cost Performance
# ------------------------

def compare_cost_performance():
    """Compare models considering both quality and cost."""
    print("\n" + "=" * 80)
    print("COST-PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Simulated model characteristics
    models_info = {
        "GPT-4": {
            "quality_score": 0.92,
            "cost_per_1k_tokens": 0.03,
            "speed_seconds": 2.5,
        },
        "GPT-3.5": {
            "quality_score": 0.78,
            "cost_per_1k_tokens": 0.002,
            "speed_seconds": 1.2,
        },
        "Claude": {
            "quality_score": 0.88,
            "cost_per_1k_tokens": 0.008,
            "speed_seconds": 1.8,
        },
        "Open Source": {
            "quality_score": 0.65,
            "cost_per_1k_tokens": 0.0,
            "speed_seconds": 3.0,
        },
    }
    
    print("\nModel Characteristics:")
    print("-" * 80)
    print(f"{'Model':<15} {'Quality':<10} {'Cost/1K':<12} {'Speed(s)':<10}")
    print("-" * 80)
    
    for model, info in models_info.items():
        print(f"{model:<15} {info['quality_score']:<10.3f} "
              f"${info['cost_per_1k_tokens']:<11.3f} {info['speed_seconds']:<10.1f}")
    
    # Calculate value scores
    print("\n" + "=" * 80)
    print("Value Analysis (Quality per Dollar):")
    print("=" * 80)
    
    for model, info in models_info.items():
        if info['cost_per_1k_tokens'] > 0:
            value = info['quality_score'] / info['cost_per_1k_tokens']
            print(f"{model}: {value:.2f} (quality points per $)")
        else:
            print(f"{model}: Infinite (free tier)")
    
    print("\n" + "=" * 80)
    print("Recommendations by Use Case:")
    print("=" * 80)
    print("- High-stakes/Critical tasks: GPT-4 (highest quality)")
    print("- High-volume/Cost-sensitive: GPT-3.5 (best value)")
    print("- Balanced needs: Claude (good quality, reasonable cost)")
    print("- Experimental/Learning: Open Source (free)")


def side_by_side_comparison():
    """Side-by-side comparison of model outputs."""
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE OUTPUT COMPARISON")
    print("=" * 80)
    
    test_prompts = [
        "Explain recursion in programming",
        "What is quantum computing?",
        "How does photosynthesis work?",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'=' * 80}")
        print(f"Prompt {i}: {prompt}")
        print("=" * 80)
        
        print(f"\nGPT-4:")
        print(f"  {ModelSimulator.gpt4_style(prompt)}")
        
        print(f"\nGPT-3.5:")
        print(f"  {ModelSimulator.gpt35_style(prompt)}")
        
        print(f"\nClaude:")
        print(f"  {ModelSimulator.claude_style(prompt)}")



# %%
# Compare Consistency
# -------------------

def compare_consistency():
    """Compare model consistency across similar prompts."""
    print("\n" + "=" * 80)
    print("CONSISTENCY ANALYSIS")
    print("=" * 80)
    
    # Similar prompts that should get similar responses
    similar_prompts = [
        "What is machine learning?",
        "Explain machine learning",
        "Define machine learning",
    ]
    
    print("\nTesting consistency across similar prompts...")
    print("(Similar prompts should yield consistent responses)\n")
    
    models = {
        "GPT-4": ModelSimulator.gpt4_style,
        "GPT-3.5": ModelSimulator.gpt35_style,
        "Claude": ModelSimulator.claude_style,
    }
    
    for model_name, model_fn in models.items():
        responses = [model_fn(prompt) for prompt in similar_prompts]
        
        # Calculate consistency (using BLEU between responses)
        consistency_scores = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                score = calculate_bleu(responses[i], responses[j])
                consistency_scores.append(score)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        
        print(f"{model_name}:")
        print(f"  Consistency Score: {avg_consistency:.3f}")
        print(f"  Assessment: {'HIGH' if avg_consistency > 0.5 else 'MODERATE' if avg_consistency > 0.3 else 'LOW'}")
        print()


def automated_model_selection():
    """Automate model selection based on requirements."""
    print("=" * 80)
    print("AUTOMATED MODEL SELECTION")
    print("=" * 80)
    
    # Define requirements
    requirements = {
        "min_quality_score": 0.80,
        "max_cost_per_1k": 0.01,
        "max_latency_seconds": 2.0,
    }
    
    models = {
        "GPT-4": {"quality": 0.92, "cost": 0.03, "latency": 2.5},
        "GPT-3.5": {"quality": 0.78, "cost": 0.002, "latency": 1.2},
        "Claude": {"quality": 0.88, "cost": 0.008, "latency": 1.8},
    }
    
    print("\nRequirements:")
    print(f"- Minimum Quality: {requirements['min_quality_score']}")
    print(f"- Maximum Cost: ${requirements['max_cost_per_1k']}/1K tokens")
    print(f"- Maximum Latency: {requirements['max_latency_seconds']}s")
    
    print("\n" + "-" * 80)
    print("Evaluating Models:")
    print("-" * 80)
    
    candidates = []
    
    for model_name, specs in models.items():
        meets_quality = specs['quality'] >= requirements['min_quality_score']
        meets_cost = specs['cost'] <= requirements['max_cost_per_1k']
        meets_latency = specs['latency'] <= requirements['max_latency_seconds']
        
        print(f"\n{model_name}:")
        print(f"  Quality: {specs['quality']:.3f} {'✓' if meets_quality else '✗'}")
        print(f"  Cost: ${specs['cost']:.3f} {'✓' if meets_cost else '✗'}")
        print(f"  Latency: {specs['latency']:.1f}s {'✓' if meets_latency else '✗'}")
        
        if meets_quality and meets_cost and meets_latency:
            candidates.append((model_name, specs))
            print(f"  Status: CANDIDATE")
        else:
            print(f"  Status: REJECTED")
    
    print("\n" + "=" * 80)
    if candidates:
        # Select best candidate (highest quality among candidates)
        best = max(candidates, key=lambda x: x[1]['quality'])
        print(f"Selected Model: {best[0]}")
        print(f"Quality: {best[1]['quality']:.3f}")
        print(f"Cost: ${best[1]['cost']:.3f}/1K tokens")
        print(f"Latency: {best[1]['latency']:.1f}s")
    else:
        print("No models meet all requirements!")
        print("Consider relaxing constraints or using a different model.")



# %%
# Main
# ----

def main():
    """Run all model comparison examples."""
    compare_model_accuracy()
    compare_response_quality()
    compare_cost_performance()
    side_by_side_comparison()
    compare_consistency()
    automated_model_selection()
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Compare models across multiple dimensions (accuracy, quality, cost)")
    print("2. Consider cost-performance tradeoffs for your use case")
    print("3. Test consistency to ensure reliable behavior")
    print("4. Use systematic evaluation to make data-driven decisions")
    print("5. Different models excel at different tasks")
    print("\nModel Selection Criteria:")
    print("- Accuracy: How correct are the outputs?")
    print("- Quality: How well-written and detailed?")
    print("- Cost: What's the price per token?")
    print("- Speed: How fast is the response?")
    print("- Consistency: How stable across similar inputs?")
    print("\nBest Practices:")
    print("- Define clear evaluation criteria before testing")
    print("- Use representative test cases from your domain")
    print("- Consider both technical metrics and business constraints")
    print("- Re-evaluate periodically as models improve")


if __name__ == "__main__":
    main()
