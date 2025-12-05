"""
Benchmarking LLM Models and Prompts.
====================================

This example demonstrates how to benchmark LLM systems systematically
using test cases and metrics. Common use cases:
- Evaluating model performance on standard datasets
- Comparing different prompt templates
- Tracking model improvements over time
- Regression testing for LLM applications
"""

from kerb.evaluation import (
    run_benchmark,
    benchmark_prompts,
    TestCase,
    calculate_f1_score,
    calculate_bleu,
)


def simple_qa_generator(question: str) -> str:
    """
    Simple Q&A generator for demonstration.
    In production, replace with actual LLM calls.
    """

# %%
# Setup and Imports
# -----------------
    qa_map = {
        "what is python": "Python is a high-level programming language known for its simplicity and readability.",
        "what is machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
        "what is nlp": "NLP (Natural Language Processing) is the branch of AI focused on language understanding.",
        "who created python": "Python was created by Guido van Rossum.",
        "what is a neural network": "A neural network is a computational model inspired by biological neurons.",
    }
    
    question_lower = question.lower()
    for key, answer in qa_map.items():
        if key in question_lower:
            return answer
    return "I don't have information about that topic."



# %%
# Benchmark Qa System
# -------------------

def benchmark_qa_system():
    """Benchmark a question-answering system on test cases."""
    print("=" * 80)
    print("BENCHMARKING Q&A SYSTEM")
    print("=" * 80)
    
    # Create test cases
    test_cases = [
        TestCase(
            id="qa_1",
            input="What is Python?",
            expected_output="Python is a programming language"
        ),
        TestCase(
            id="qa_2",
            input="What is machine learning?",
            expected_output="Machine learning is AI that learns from data"
        ),
        TestCase(
            id="qa_3",
            input="What is NLP?",
            expected_output="Natural Language Processing"
        ),
        TestCase(
            id="qa_4",
            input="Who created Python?",
            expected_output="Guido van Rossum"
        ),
        TestCase(
            id="qa_5",
            input="What is a neural network?",
            expected_output="Neural network is like a brain"
        ),
    ]
    
    print(f"\nRunning benchmark with {len(test_cases)} test cases...")
    
    # Define evaluation function
    def evaluate_answer(output: str, expected: str) -> float:
        """Evaluate using F1 score."""
        return calculate_f1_score(output, expected)
    
    # Run benchmark
    result = run_benchmark(
        test_cases,
        simple_qa_generator,
        evaluate_answer,
        threshold=0.3,  # 30% F1 score to pass
        name="Q&A System Benchmark"
    )
    
    print("\n" + "-" * 80)
    print("Benchmark Results:")
    print("-" * 80)
    print(f"Total Tests: {result.total_tests}")
    print(f"Passed: {result.passed_tests}")
    print(f"Failed: {result.failed_tests}")
    print(f"Pass Rate: {result.pass_rate:.1f}%")
    print(f"Average Score: {result.average_score:.3f}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    
    # Show individual scores
    print("\nIndividual Test Scores:")
    for i, (test_case, score) in enumerate(zip(test_cases, result.scores), 1):
        status = "PASS" if score >= 0.3 else "FAIL"
        print(f"  {test_case.id}: {score:.3f} [{status}]")



# %%
# Compare Prompt Templates
# ------------------------

def compare_prompt_templates():
    """Compare different prompt templates to find the best one."""
    print("\n" + "=" * 80)
    print("PROMPT TEMPLATE COMPARISON")
    print("=" * 80)
    
    # Different prompt templates
    prompts = [
        ("simple", "Q: {input}\nA:"),
        ("instructive", "Answer the following question concisely:\n{input}"),
        ("detailed", "Please provide a detailed answer to: {input}"),
    ]
    
    # Test inputs
    test_inputs = [
        "What is Python?",
        "What is machine learning?",
        "What is NLP?",
    ]
    
    print(f"\nComparing {len(prompts)} prompt templates on {len(test_inputs)} inputs...")
    
    def generate_with_template(template: str, input_text: str) -> str:
        """Generate answer using prompt template."""
        prompted_input = template.format(input=input_text)
        # Extract just the question for our simple generator
        return simple_qa_generator(input_text)
    

# %%
# Evaluate By Length
# ------------------

    def evaluate_by_length(output: str) -> float:
        """Simple evaluation: longer answers score higher."""
        word_count = len(output.split())
        # Normalize to 0-1 range (assume 20 words is a good answer)
        return min(word_count / 20.0, 1.0)
    
    # Benchmark all prompts
    results = benchmark_prompts(
        prompts,
        test_inputs,
        generate_with_template,
        evaluate_by_length
    )
    
    print("\n" + "-" * 80)
    print("Prompt Comparison Results:")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"\nTemplate '{name}':")
        print(f"  Average Score: {result.average_score:.3f}")
        print(f"  Pass Rate: {result.pass_rate:.1f}%")
        print(f"  Total Tests: {result.total_tests}")
    
    # Find best prompt
    best_prompt = max(results.items(), key=lambda x: x[1].average_score)
    print(f"\nBest Prompt: '{best_prompt[0]}' (score: {best_prompt[1].average_score:.3f})")


def benchmark_summarization():
    """Benchmark a summarization system."""
    print("\n" + "=" * 80)
    print("SUMMARIZATION BENCHMARK")
    print("=" * 80)
    

# %%
# Simple Summarizer
# -----------------

    def simple_summarizer(text: str) -> str:
        """Simple extractive summarizer - takes first sentence."""
        sentences = text.split('.')
        return sentences[0].strip() + '.' if sentences else text
    
    test_cases = [
        TestCase(
            id="sum_1",
            input="Python is a versatile programming language. It is used for web development, data science, and automation. Many companies rely on Python for their applications.",
            expected_output="Python is a versatile programming language used widely."
        ),
        TestCase(
            id="sum_2",
            input="Machine learning enables computers to learn from data. It powers recommendation systems, image recognition, and natural language processing. The field is rapidly evolving.",
            expected_output="Machine learning enables data-driven learning for various applications."
        ),
        TestCase(
            id="sum_3",
            input="Climate change is affecting ecosystems globally. Rising temperatures impact wildlife habitats. Scientists urge immediate action.",
            expected_output="Climate change impacts global ecosystems significantly."
        ),
    ]
    
    print(f"\nBenchmarking summarization on {len(test_cases)} test cases...")
    
    def evaluate_summary(output: str, expected: str) -> float:
        """Evaluate summary using BLEU score."""
        return calculate_bleu(output, expected)
    
    result = run_benchmark(
        test_cases,
        simple_summarizer,
        evaluate_summary,
        threshold=0.2,
        name="Summarization Benchmark"
    )
    
    print("\n" + "-" * 80)
    print("Results:")
    print("-" * 80)
    print(f"Pass Rate: {result.pass_rate:.1f}%")
    print(f"Average BLEU: {result.average_score:.3f}")
    print(f"Passed: {result.passed_tests}/{result.total_tests}")



# %%
# Regression Testing
# ------------------

def regression_testing():
    """Use benchmarking for regression testing."""
    print("\n" + "=" * 80)
    print("REGRESSION TESTING")
    print("=" * 80)
    
    # Standard test suite for regression testing
    test_suite = [
        TestCase(id="reg_1", input="What is AI?", expected_output="artificial intelligence"),
        TestCase(id="reg_2", input="What is ML?", expected_output="machine learning"),
        TestCase(id="reg_3", input="Define NLP", expected_output="natural language processing"),
    ]
    
    print("\nRunning regression test suite...")
    print("This helps ensure model updates don't degrade performance.\n")
    
    def evaluate_keywords(output: str, expected: str) -> float:
        """Simple keyword-based evaluation."""
        output_lower = output.lower()
        expected_lower = expected.lower()
        
        # Check if expected keywords appear in output
        expected_words = set(expected_lower.split())
        output_words = set(output_lower.split())
        
        if not expected_words:
            return 0.0
        
        matches = len(expected_words & output_words)
        return matches / len(expected_words)
    
    # Simulate "before" and "after" model versions
    print("=" * 80)
    print("Version 1.0 (Baseline)")
    print("=" * 80)
    
    result_v1 = run_benchmark(
        test_suite,
        simple_qa_generator,
        evaluate_keywords,
        threshold=0.5,
        name="Version 1.0"
    )
    
    print(f"Pass Rate: {result_v1.pass_rate:.1f}%")
    print(f"Average Score: {result_v1.average_score:.3f}")
    
    # Simulate improved model

# %%
# Improved Generator
# ------------------

    def improved_generator(question: str) -> str:
        """Improved version with better answers."""
        base_answer = simple_qa_generator(question)
        # Add more detail to simulate improvement
        return base_answer
    
    print("\n" + "=" * 80)
    print("Version 2.0 (Updated Model)")
    print("=" * 80)
    
    result_v2 = run_benchmark(
        test_suite,
        improved_generator,
        evaluate_keywords,
        threshold=0.5,
        name="Version 2.0"
    )
    
    print(f"Pass Rate: {result_v2.pass_rate:.1f}%")
    print(f"Average Score: {result_v2.average_score:.3f}")
    
    # Compare versions
    print("\n" + "=" * 80)
    print("Regression Analysis")
    print("=" * 80)
    
    score_diff = result_v2.average_score - result_v1.average_score
    print(f"Score Change: {score_diff:+.3f}")
    
    if score_diff > 0:
        print(f"Status: IMPROVED ({abs(score_diff):.1%} better)")
    elif score_diff < 0:
        print(f"Status: REGRESSION ({abs(score_diff):.1%} worse)")
    else:
        print("Status: NO CHANGE")


def track_performance_over_time():
    """Demonstrate tracking performance metrics over iterations."""
    print("\n" + "=" * 80)
    print("PERFORMANCE TRACKING")
    print("=" * 80)
    
    test_cases = [
        TestCase(id="t1", input="What is Python?", expected_output="programming language"),
        TestCase(id="t2", input="What is ML?", expected_output="machine learning"),
    ]
    

# %%
# Keyword Eval
# ------------

    def keyword_eval(output: str, expected: str) -> float:
        return 1.0 if expected.lower() in output.lower() else 0.0
    
    # Simulate multiple evaluation runs
    print("\nTracking performance across 5 evaluation runs:\n")
    
    scores_history = []
    
    for run in range(1, 6):
        result = run_benchmark(
            test_cases,
            simple_qa_generator,
            keyword_eval,
            name=f"Run {run}"
        )
        scores_history.append(result.average_score)
        print(f"Run {run}: Average Score = {result.average_score:.3f}")
    
    # Analysis
    print("\n" + "-" * 80)
    print("Performance Summary:")
    print("-" * 80)
    print(f"Best Score: {max(scores_history):.3f}")
    print(f"Worst Score: {min(scores_history):.3f}")
    print(f"Mean Score: {sum(scores_history) / len(scores_history):.3f}")
    print(f"Score Range: {max(scores_history) - min(scores_history):.3f}")



# %%
# Main
# ----

def main():
    """Run all benchmarking examples."""
    benchmark_qa_system()
    compare_prompt_templates()
    benchmark_summarization()
    regression_testing()
    track_performance_over_time()
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Benchmarks provide systematic performance measurement")
    print("2. Compare prompts to optimize your LLM application")
    print("3. Use regression testing to catch performance degradation")
    print("4. Track metrics over time to measure improvements")
    print("5. Define clear test cases and evaluation criteria")
    print("\nBest Practices:")
    print("- Maintain a standard test suite for your domain")
    print("- Set appropriate score thresholds for pass/fail")
    print("- Run benchmarks regularly (CI/CD integration)")
    print("- Track both aggregate and per-test metrics")


if __name__ == "__main__":
    main()
