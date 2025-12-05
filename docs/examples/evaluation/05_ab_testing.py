"""
A/B Testing for LLM Systems.
============================

This example demonstrates how to perform statistical A/B testing to compare
different model variants, prompts, or configurations. Common use cases:
- Testing prompt variations to find the best performer
- Comparing different model versions
- Evaluating configuration changes
- Making data-driven decisions about model improvements
"""

from kerb.evaluation import (
    ab_test,
    compare_outputs,
    calculate_statistics,
)


def test_prompt_variations():
    """A/B test two different prompt variations."""
    print("=" * 80)
    print("A/B TESTING PROMPT VARIATIONS")
    print("=" * 80)
    
    # Simulate outputs from two prompt variations
    # Prompt A: "Answer briefly: {question}"
    # Prompt B: "Provide a concise answer: {question}"
    
    outputs_a = [
        "Python is a programming language.",
        "Machine learning is a subset of AI.",
        "NLP processes human language.",
        "APIs enable software communication.",
        "Cloud computing provides on-demand resources.",
    ]
    
    outputs_b = [
        "Python is a high-level programming language known for simplicity.",
        "Machine learning enables systems to learn from data automatically.",
        "Natural Language Processing helps computers understand human language.",
        "APIs allow different software applications to communicate.",
        "Cloud computing delivers computing services over the internet.",
    ]
    
    print("\nPrompt A: 'Answer briefly: {question}'")
    print("Prompt B: 'Provide a concise answer: {question}'")
    print(f"\nTesting with {len(outputs_a)} samples from each variant...")
    
    # Evaluation function: longer, more detailed answers score higher
    def evaluate_detail_level(output: str) -> float:
        """Score based on answer length and detail."""

# %%
# Setup and Imports
# -----------------
        word_count = len(output.split())
        # Ideal answer: 8-15 words
        if word_count < 8:
            return word_count / 8.0
        elif word_count <= 15:
            return 1.0
        else:
            # Penalize very long answers
            return max(0.5, 1.0 - (word_count - 15) * 0.05)
    
    results = ab_test(
        outputs_a,
        outputs_b,
        evaluation_fn=evaluate_detail_level,
        labels=("Prompt A", "Prompt B")
    )
    
    print("\n" + "-" * 80)
    print("A/B Test Results:")
    print("-" * 80)
    print(f"Winner: {results['winner']}")
    print(f"\n{results['variant_a']['label']} Statistics:")
    print(f"  Mean Score: {results['variant_a']['mean']:.3f}")
    print(f"  Std Dev: {results['variant_a']['stdev']:.3f}")
    print(f"\n{results['variant_b']['label']} Statistics:")
    print(f"  Mean Score: {results['variant_b']['mean']:.3f}")
    print(f"  Std Dev: {results['variant_b']['stdev']:.3f}")
    print(f"\nDifference: {results['difference']:.3f}")
    print(f"Statistical Significance: {results.get('significant', 'N/A')}")



# %%
# Compare Model Versions
# ----------------------

def compare_model_versions():
    """A/B test two different model versions."""
    print("\n" + "=" * 80)
    print("A/B TESTING MODEL VERSIONS")
    print("=" * 80)
    
    # Simulate outputs from two model versions
    # Model v1: Baseline
    # Model v2: Fine-tuned version
    
    outputs_v1 = [
        "The answer is positive.",
        "This is correct.",
        "Yes, that's right.",
        "Affirmative.",
        "Correct response.",
        "Indeed.",
        "That is accurate.",
        "True statement.",
    ]
    
    outputs_v2 = [
        "Based on the context, the answer is positive.",
        "This statement is correct according to the information provided.",
        "Yes, that's right. Here's why: it aligns with the facts.",
        "Affirmative. This conclusion is supported by the evidence.",
        "Correct response, considering all the relevant factors.",
        "Indeed, this is accurate based on the given data.",
        "That is accurate as per the documentation.",
        "True statement, which can be verified from multiple sources.",
    ]
    
    print("\nComparing Model v1 (Baseline) vs Model v2 (Fine-tuned)")
    print(f"Sample size: {len(outputs_v1)} outputs each\n")
    
    # Evaluation: prefer answers with reasoning
    def evaluate_with_reasoning(output: str) -> float:
        """Higher score for answers that include reasoning."""
        reasoning_keywords = ['because', 'based on', 'according to', 'considering', 
                             'supported by', 'as per', 'verified', 'why']
        
        base_score = 0.5
        output_lower = output.lower()
        
        for keyword in reasoning_keywords:
            if keyword in output_lower:
                base_score += 0.1
        
        # Bonus for longer, more detailed responses
        word_count = len(output.split())
        if word_count > 8:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    results = ab_test(
        outputs_v1,
        outputs_v2,
        evaluation_fn=evaluate_with_reasoning,
        labels=("Model v1", "Model v2")
    )
    
    print("-" * 80)
    print("Results:")
    print("-" * 80)
    print(f"Winner: {results['winner']}")
    print(f"\nModel v1 (Baseline):")
    print(f"  Mean Score: {results['variant_a']['mean']:.3f}")
    print(f"  Min/Max: {min(results['variant_a']['scores']):.3f} / {max(results['variant_a']['scores']):.3f}")
    print(f"\nModel v2 (Fine-tuned):")
    print(f"  Mean Score: {results['variant_b']['mean']:.3f}")
    print(f"  Min/Max: {min(results['variant_b']['scores']):.3f} / {max(results['variant_b']['scores']):.3f}")
    print(f"\nImprovement: {(results['variant_b']['mean'] - results['variant_a']['mean']) / results['variant_a']['mean'] * 100:.1f}%")
    
    # Decision
    if results['winner'] == results['variant_b']['label']:
        print("\nRecommendation: Deploy Model v2 (shows clear improvement)")
    else:
        print("\nRecommendation: Keep Model v1 (v2 doesn't show improvement)")



# %%
# Test Response Length
# --------------------

def test_response_length():
    """A/B test different response length configurations."""
    print("\n" + "=" * 80)
    print("A/B TESTING RESPONSE LENGTH")
    print("=" * 80)
    
    # Short responses (max_tokens=50)
    short_responses = [
        "AI is artificial intelligence.",
        "ML is machine learning.",
        "NLP handles language.",
        "DL is deep learning.",
        "NN are neural networks.",
    ]
    
    # Long responses (max_tokens=200)
    long_responses = [
        "Artificial Intelligence (AI) is a broad field of computer science focused on creating intelligent machines that can perform tasks requiring human-like intelligence.",
        "Machine Learning (ML) is a subset of AI that enables systems to automatically learn and improve from experience without being explicitly programmed.",
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language in a valuable way.",
        "Deep Learning (DL) is a subset of machine learning based on artificial neural networks with multiple layers that can learn complex patterns.",
        "Neural Networks (NN) are computing systems inspired by biological neural networks that constitute animal brains and form the foundation of deep learning.",
    ]
    
    print("\nConfiguration A: Short responses (max_tokens=50)")
    print("Configuration B: Long responses (max_tokens=200)")
    print(f"\nEvaluating {len(short_responses)} responses from each configuration...\n")
    
    # Evaluation: balance between informativeness and conciseness
    def evaluate_quality(output: str) -> float:
        """Score balancing detail and conciseness."""
        word_count = len(output.split())
        
        # Optimal range: 12-20 words
        if word_count < 8:
            return 0.3  # Too brief
        elif 8 <= word_count < 12:
            return 0.6
        elif 12 <= word_count <= 20:
            return 1.0  # Optimal
        elif 20 < word_count <= 30:
            return 0.8
        else:
            return 0.5  # Too verbose
    
    results = ab_test(
        short_responses,
        long_responses,
        evaluation_fn=evaluate_quality,
        labels=("Short (50 tokens)", "Long (200 tokens)")
    )
    
    print("-" * 80)
    print("Results:")
    print("-" * 80)
    print(f"Winner: {results['winner']}")
    print(f"\nShort responses: Mean = {results['variant_a']['mean']:.3f}")
    print(f"Long responses: Mean = {results['variant_b']['mean']:.3f}")
    print(f"\nConclusion: {results['winner']} configuration provides better balance")



# %%
# Multi Variant Comparison
# ------------------------

def multi_variant_comparison():
    """Compare multiple variants simultaneously."""
    print("\n" + "=" * 80)
    print("MULTI-VARIANT COMPARISON")
    print("=" * 80)
    
    # Simulate three different prompt strategies
    variants = {
        "Direct": [
            "Paris is the capital of France.",
            "Tokyo is the capital of Japan.",
            "London is the capital of UK.",
            "Berlin is the capital of Germany.",
        ],
        "Contextual": [
            "The capital of France is Paris, a major European city.",
            "Tokyo, the capital of Japan, is one of the world's largest cities.",
            "London serves as the capital of the United Kingdom.",
            "Berlin is the capital and largest city of Germany.",
        ],
        "Educational": [
            "Paris, located in northern France, has been the country's capital since 987 CE.",
            "Tokyo became Japan's capital in 1868, replacing Kyoto.",
            "London has been England's capital since the 12th century.",
            "Berlin became the capital of unified Germany in 1990.",
        ],
    }
    
    print("\nComparing three prompt strategies:")
    print("1. Direct: Simple, factual answers")
    print("2. Contextual: Adds relevant context")
    print("3. Educational: Includes historical information")
    print()
    
    # Evaluate based on informativeness
    def evaluate_informativeness(output: str) -> float:
        """Score based on information density."""
        word_count = len(output.split())
        # More words generally means more information (up to a point)
        return min(word_count / 15.0, 1.0)
    
    # Calculate scores for each variant
    variant_scores = {}
    for variant_name, outputs in variants.items():
        scores = [evaluate_informativeness(output) for output in outputs]
        variant_scores[variant_name] = {
            'mean': sum(scores) / len(scores),
            'std': (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5,
            'min': min(scores),
            'max': max(scores),
        }
    
    print("-" * 80)
    print("Comparison Results:")
    print("-" * 80)
    
    # Sort by mean score
    sorted_variants = sorted(variant_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    for rank, (variant, stats) in enumerate(sorted_variants, 1):
        print(f"\n{rank}. {variant}")
        print(f"   Mean Score: {stats['mean']:.3f}")
        print(f"   Std Dev: {stats['std']:.3f}")
        print(f"   Min/Max: {stats['min']:.3f} / {stats['max']:.3f}")
    
    print(f"\nBest Variant: {sorted_variants[0][0]}")
    print(f"Rankings: {[v for v, _ in sorted_variants]}")



# %%
# Statistical Significance Analysis
# ---------------------------------

def statistical_significance_analysis():
    """Demonstrate statistical analysis of A/B test results."""
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    
    # Scenario: Two nearly identical configurations
    outputs_config1 = [0.75, 0.82, 0.68, 0.79, 0.85, 0.73, 0.80, 0.77]
    outputs_config2 = [0.78, 0.84, 0.70, 0.81, 0.87, 0.75, 0.82, 0.79]
    
    print("\nComparing two similar configurations...")
    print("Testing whether small differences are statistically meaningful.\n")
    
    # Calculate statistics for both
    stats1 = calculate_statistics([0.75, 0.82, 0.68, 0.79, 0.85, 0.73, 0.80, 0.77])
    stats2 = calculate_statistics([0.78, 0.84, 0.70, 0.81, 0.87, 0.75, 0.82, 0.79])
    
    print("Configuration 1:")
    print(f"  Mean: {stats1['mean']:.3f}")
    print(f"  Median: {stats1['median']:.3f}")
    print(f"  Std Dev: {stats1['stdev']:.3f}")
    
    print("\nConfiguration 2:")
    print(f"  Mean: {stats2['mean']:.3f}")
    print(f"  Median: {stats2['median']:.3f}")
    print(f"  Std Dev: {stats2['stdev']:.3f}")
    
    difference = stats2['mean'] - stats1['mean']
    print(f"\nMean Difference: {difference:.3f} ({difference/stats1['mean']*100:+.1f}%)")
    
    # Simple significance check
    if abs(difference) < 0.05:
        print("Significance: Difference is too small to be meaningful")
        print("Recommendation: Keep current configuration")
    else:
        print("Significance: Difference is meaningful")
        print(f"Recommendation: Use Configuration {'2' if difference > 0 else '1'}")


def main():
    """Run all A/B testing examples."""
    test_prompt_variations()
    compare_model_versions()
    test_response_length()
    multi_variant_comparison()
    statistical_significance_analysis()
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. A/B testing enables data-driven decision making")
    print("2. Compare variants with statistical rigor")
    print("3. Consider sample size and statistical significance")
    print("4. Use multiple evaluation metrics for comprehensive comparison")
    print("5. Document and track experiment results over time")
    print("\nBest Practices:")
    print("- Define clear success metrics before testing")
    print("- Use sufficient sample sizes (typically 30+ per variant)")
    print("- Run tests long enough to account for variability")
    print("- Consider both statistical and practical significance")
    print("- Monitor for degradation in other metrics")


if __name__ == "__main__":
    main()
