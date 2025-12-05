"""
Performance Testing Example
===========================

This example demonstrates how to measure and benchmark LLM application performance.

Main concepts:
- Measuring response latency
- Calculating throughput (requests/second)
- Benchmarking different prompts
- Percentile latency analysis (P50, P95, P99)
- Performance regression detection
- Generating performance reports

Use cases for LLM developers:
- Benchmarking prompt efficiency
- Comparing model performance
- Detecting performance regressions
- Optimizing inference speed
- Capacity planning
- SLA compliance testing
"""

import time
from kerb.testing.performance import (
    measure_latency,
    measure_throughput,
    benchmark_prompts,
    PerformanceReport
)
from kerb.testing import MockLLM, MockBehavior


def simulate_llm_call(prompt: str, delay: float = 0.1) -> str:
    """Simulate an LLM call with configurable delay."""
    time.sleep(delay)
    return f"Response to: {prompt[:30]}..."


def main():
    """Run performance testing examples."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("PERFORMANCE TESTING EXAMPLE")
    print("="*80)
    
    # Example 1: Basic latency measurement
    print("\n1. BASIC LATENCY MEASUREMENT")
    print("-"*80)
    
    # Fast mock LLM
    fast_llm = MockLLM(
        responses="This is a quick response",
        behavior=MockBehavior.FIXED,
        latency=0.05  # 50ms
    )
    

# %%
# Call Llm
# --------

    def call_llm(prompt):
        return fast_llm.generate(prompt).content
    
    metrics = measure_latency(
        fn=call_llm,
        args=["What is machine learning?"],
        num_runs=10,
        warmup=2
    )
    
    print(f"Performance metrics:")
    print(f"  Total requests: {metrics.total_requests}")
    print(f"  Average latency: {metrics.avg_latency:.3f}s")
    print(f"  Min latency: {metrics.min_latency:.3f}s")
    print(f"  Max latency: {metrics.max_latency:.3f}s")
    print(f"  P50 latency: {metrics.p50_latency:.3f}s")
    print(f"  P95 latency: {metrics.p95_latency:.3f}s")
    print(f"  P99 latency: {metrics.p99_latency:.3f}s")
    print(f"  Throughput: {metrics.throughput:.2f} req/s")
    
    # Example 2: Comparing different prompt lengths
    print("\n2. COMPARING PROMPT LENGTHS")
    print("-"*80)
    
    # Simulate different latencies for different prompt lengths

# %%
# Variable Latency Llm
# --------------------

    def variable_latency_llm(prompt: str) -> str:
        # Longer prompts take slightly longer
        delay = 0.05 + (len(prompt) * 0.0001)
        time.sleep(delay)
        return f"Response (processed {len(prompt)} chars)"
    
    short_prompt = "Hi"
    medium_prompt = "Explain machine learning in simple terms."
    long_prompt = "Provide a comprehensive explanation of machine learning, including supervised learning, unsupervised learning, reinforcement learning, and their applications in modern AI systems."
    
    prompts_to_test = [
        ("short", short_prompt),
        ("medium", medium_prompt),
        ("long", long_prompt)
    ]
    
    print("Benchmarking different prompt lengths:")
    for name, prompt in prompts_to_test:
        metrics = measure_latency(
            fn=variable_latency_llm,
            args=[prompt],
            num_runs=5,
            warmup=1
        )
        print(f"\n  {name} ({len(prompt)} chars):")
        print(f"    Avg latency: {metrics.avg_latency:.3f}s")
        print(f"    P95 latency: {metrics.p95_latency:.3f}s")
    
    # Example 3: Throughput measurement
    print("\n3. THROUGHPUT MEASUREMENT")
    print("-"*80)
    
    fast_mock = MockLLM(
        responses="Fast response",
        behavior=MockBehavior.FIXED,
        latency=0.01
    )
    

# %%
# Batch Call
# ----------

    def batch_call(prompt):
        return fast_mock.generate(prompt).content
    
    test_prompts = [
        ["Query 1"],
        ["Query 2"],
        ["Query 3"],
    ]
    
    print("Measuring throughput over 2 seconds...")
    throughput = measure_throughput(
        fn=batch_call,
        args_list=test_prompts,
        duration=2.0
    )
    
    print(f"Throughput: {throughput:.2f} requests/second")
    
    # Example 4: Benchmarking multiple prompts
    print("\n4. BENCHMARKING MULTIPLE PROMPTS")
    print("-"*80)
    
    test_llm = MockLLM(
        responses="Benchmark response",
        behavior=MockBehavior.FIXED,
        latency=0.03
    )
    
    benchmark_test_prompts = [
        "What is Python?",
        "Explain machine learning",
        "How does a neural network work?",
        "What is natural language processing?",
    ]
    

# %%
# Llm Function
# ------------

    def llm_function(prompt: str) -> str:
        return test_llm.generate(prompt).content
    
    print("Benchmarking prompt set...")
    results = benchmark_prompts(
        prompts=benchmark_test_prompts,
        llm_fn=llm_function,
        num_runs=5
    )
    
    for prompt_id, metrics in results.items():
        print(f"\n  {prompt_id}:")
        print(f"    Avg: {metrics.avg_latency:.3f}s")
        print(f"    P95: {metrics.p95_latency:.3f}s")
        print(f"    Throughput: {metrics.throughput:.2f} req/s")
    
    # Example 5: Performance report generation
    print("\n5. PERFORMANCE REPORT GENERATION")
    print("-"*80)
    
    report = PerformanceReport()
    
    # Add metrics from different test scenarios
    scenarios = {
        "simple_qa": 0.02,
        "code_generation": 0.15,
        "long_form_writing": 0.25,
        "translation": 0.08,
    }
    
    for scenario_name, latency in scenarios.items():
        mock = MockLLM(responses="test", latency=latency)
        
        metrics = measure_latency(
            fn=lambda p: mock.generate(p).content,
            args=["test prompt"],
            num_runs=5,
            warmup=1
        )
        
        report.add_metrics(scenario_name, metrics)
    
    print(report.generate_report())
    
    # Example 6: Detecting performance regressions
    print("\n6. PERFORMANCE REGRESSION DETECTION")
    print("-"*80)
    
    # Baseline performance
    baseline_llm = MockLLM(responses="baseline", latency=0.05)
    baseline_metrics = measure_latency(
        fn=lambda p: baseline_llm.generate(p).content,
        args=["test"],
        num_runs=10
    )
    
    # New version with regression
    regression_llm = MockLLM(responses="new version", latency=0.08)
    new_metrics = measure_latency(
        fn=lambda p: regression_llm.generate(p).content,
        args=["test"],
        num_runs=10
    )
    
    print(f"Baseline P95 latency: {baseline_metrics.p95_latency:.3f}s")
    print(f"New P95 latency: {new_metrics.p95_latency:.3f}s")
    
    # Check for regression (>10% increase)
    threshold = 0.10  # 10% regression tolerance
    regression_ratio = (new_metrics.p95_latency - baseline_metrics.p95_latency) / baseline_metrics.p95_latency
    
    if regression_ratio > threshold:
        print(f"\nWARNING: Performance regression detected!")
        print(f"  Regression: {regression_ratio*100:.1f}% slower")
    else:
        print(f"\nNo significant regression (within {threshold*100}% tolerance)")
    
    # Example 7: Percentile analysis
    print("\n7. PERCENTILE LATENCY ANALYSIS")
    print("-"*80)
    
    # Simulate variable latency
    import random
    random.seed(42)
    
    variable_llm = MockLLM(
        responses=["Response A", "Response B", "Response C"],
        behavior=MockBehavior.RANDOM,
        latency=0.05  # Base latency
    )
    
    # Add some variability

# %%
# Variable Call
# -------------

    def variable_call(prompt):
        # Add random jitter
        extra_delay = random.random() * 0.05
        time.sleep(extra_delay)
        return variable_llm.generate(prompt).content
    
    metrics = measure_latency(
        fn=variable_call,
        args=["test"],
        num_runs=20,
        warmup=2
    )
    
    print("Latency distribution:")
    print(f"  P50 (median): {metrics.p50_latency:.3f}s")
    print(f"  P95: {metrics.p95_latency:.3f}s")
    print(f"  P99: {metrics.p99_latency:.3f}s")
    print(f"  Max: {metrics.max_latency:.3f}s")
    print(f"\nInterpretation:")
    print(f"  50% of requests complete in < {metrics.p50_latency:.3f}s")
    print(f"  95% of requests complete in < {metrics.p95_latency:.3f}s")
    print(f"  99% of requests complete in < {metrics.p99_latency:.3f}s")
    
    # Example 8: Load testing simulation
    print("\n8. LOAD TESTING SIMULATION")
    print("-"*80)
    
    load_test_llm = MockLLM(
        responses="Load test response",
        behavior=MockBehavior.FIXED,
        latency=0.02
    )
    
    concurrent_requests = [
        ["Request 1"],
        ["Request 2"],
        ["Request 3"],
        ["Request 4"],
        ["Request 5"],
    ]
    
    print("Simulating concurrent load...")
    load_throughput = measure_throughput(
        fn=lambda p: load_test_llm.generate(p).content,
        args_list=concurrent_requests,
        duration=3.0
    )
    
    print(f"Sustained throughput: {load_throughput:.2f} req/s")
    print(f"Estimated capacity:")
    print(f"  Per minute: {load_throughput * 60:.0f} requests")
    print(f"  Per hour: {load_throughput * 3600:.0f} requests")
    print(f"  Per day: {load_throughput * 86400:.0f} requests")
    
    # Example 9: SLA compliance check
    print("\n9. SLA COMPLIANCE CHECK")
    print("-"*80)
    
    sla_requirements = {
        "p50_max": 0.100,  # 100ms
        "p95_max": 0.200,  # 200ms
        "p99_max": 0.500,  # 500ms
    }
    
    sla_llm = MockLLM(responses="SLA test", latency=0.05)
    sla_metrics = measure_latency(
        fn=lambda p: sla_llm.generate(p).content,
        args=["test"],
        num_runs=20
    )
    
    print("SLA Compliance Check:")
    
    checks = [
        ("P50", sla_metrics.p50_latency, sla_requirements["p50_max"]),
        ("P95", sla_metrics.p95_latency, sla_requirements["p95_max"]),
        ("P99", sla_metrics.p99_latency, sla_requirements["p99_max"]),
    ]
    
    all_pass = True
    for name, actual, requirement in checks:
        passed = actual <= requirement
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {actual:.3f}s <= {requirement:.3f}s [{status}]")
    
    print(f"\nOverall SLA compliance: {'PASS' if all_pass else 'FAIL'}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("- Measure latency to track performance")
    print("- Use percentiles (P95, P99) for SLA compliance")
    print("- Monitor throughput for capacity planning")
    print("- Detect regressions by comparing baselines")
    print("- Generate reports for stakeholders")
    print("- Test under load to ensure scalability")


if __name__ == "__main__":
    main()
