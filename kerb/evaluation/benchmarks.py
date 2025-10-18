"""Benchmarking utilities for evaluation.

This module provides functions for running benchmarks on test cases
and comparing different prompts or models.
"""

import statistics
import time
from typing import Callable, Dict, List, Tuple

from .types import BenchmarkResult, TestCase

# ============================================================================
# Benchmarking Functions
# ============================================================================


def run_benchmark(
    test_cases: List[TestCase],
    generation_fn: Callable[[str], str],
    evaluation_fn: Callable[[str, str], float],
    threshold: float = 0.7,
    name: str = "benchmark",
) -> BenchmarkResult:
    """Run a benchmark on a set of test cases.

    Args:
        test_cases: List of test cases
        generation_fn: Function to generate output from input
        evaluation_fn: Function to evaluate output (returns score 0-1)
        threshold: Pass threshold (default: 0.7)
        name: Benchmark name

    Returns:
        BenchmarkResult: Benchmark results

    Example:
        >>> cases = [TestCase(id="1", input="What is AI?", expected_output="Artificial Intelligence")]
        >>> result = run_benchmark(cases, lambda x: "AI means " + x, lambda o, e: 0.8)
        >>> result.pass_rate
        100.0
    """
    start_time = time.time()

    scores = []
    passed = 0
    failed = 0
    details = []

    for test_case in test_cases:
        try:
            # Generate output
            output = generation_fn(test_case.input)

            # Evaluate
            if test_case.expected_output:
                score = evaluation_fn(output, test_case.expected_output)
            else:
                # No expected output, just score the output
                score = evaluation_fn(output, "")

            scores.append(score)

            if score >= threshold:
                passed += 1
            else:
                failed += 1

            details.append(
                {
                    "test_id": test_case.id,
                    "score": score,
                    "passed": score >= threshold,
                    "output": output[:100],
                }
            )

        except Exception as e:
            failed += 1
            scores.append(0.0)
            details.append(
                {
                    "test_id": test_case.id,
                    "score": 0.0,
                    "passed": False,
                    "error": str(e),
                }
            )

    execution_time = time.time() - start_time

    return BenchmarkResult(
        name=name,
        total_tests=len(test_cases),
        passed_tests=passed,
        failed_tests=failed,
        average_score=statistics.mean(scores) if scores else 0.0,
        scores=scores,
        execution_time=execution_time,
        details={"test_results": details},
    )


def benchmark_prompts(
    prompts: List[Tuple[str, str]],
    test_inputs: List[str],
    generation_fn: Callable[[str, str], str],
    evaluation_fn: Callable[[str], float],
) -> Dict[str, BenchmarkResult]:
    """Benchmark multiple prompts against test inputs.

    Args:
        prompts: List of (prompt_id, prompt_template) tuples
        test_inputs: List of test inputs
        generation_fn: Function(prompt, input) -> output
        evaluation_fn: Function(output) -> score

    Returns:
        dict: Benchmark results for each prompt

    Example:
        >>> results = benchmark_prompts(
        ...     [("v1", "Answer: {input}"), ("v2", "Detailed answer: {input}")],
        ...     ["What is AI?", "What is ML?"],
        ...     lambda p, i: p.format(input=i),
        ...     lambda o: len(o.split()) / 10
        ... )
        >>> len(results)
        2
    """
    results = {}

    for prompt_id, prompt_template in prompts:
        scores = []

        for test_input in test_inputs:
            try:
                output = generation_fn(prompt_template, test_input)
                score = evaluation_fn(output)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        results[prompt_id] = BenchmarkResult(
            name=prompt_id,
            total_tests=len(test_inputs),
            passed_tests=sum(1 for s in scores if s >= 0.7),
            failed_tests=sum(1 for s in scores if s < 0.7),
            average_score=statistics.mean(scores) if scores else 0.0,
            scores=scores,
        )

    return results
