"""Prompt testing and regression utilities."""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .types import PromptTestCase, TestResult


class PromptTestSuite:
    """Collection of prompt tests for regression testing."""

    def __init__(self, name: str):
        """Initialize test suite.

        Args:
            name: Suite name
        """
        self.name = name
        self.test_cases: List[PromptTestCase] = []
        self.results: List[TestResult] = []

    def add_test(self, test_case: PromptTestCase) -> None:
        """Add a test case to the suite."""
        self.test_cases.append(test_case)

    def run(
        self, llm_fn: Callable[[str], str], verbose: bool = False
    ) -> List[TestResult]:
        """Run all tests in the suite.

        Args:
            llm_fn: Function that takes a prompt and returns a response
            verbose: Print progress

        Returns:
            List of test results
        """
        self.results = []

        for test_case in self.test_cases:
            if verbose:
                print(f"Running test: {test_case.name}")

            for i, test_input in enumerate(test_case.test_inputs):
                # Format prompt with inputs
                prompt = test_case.prompt_template.format(**test_input)

                # Generate response
                start_time = time.time()
                try:
                    response = llm_fn(prompt)
                    latency = time.time() - start_time

                    # Validate response
                    passed = True
                    error = None

                    # Check expected output if provided
                    if test_case.expected_outputs and i < len(
                        test_case.expected_outputs
                    ):
                        expected = test_case.expected_outputs[i]
                        if response != expected:
                            passed = False
                            error = f"Output mismatch: expected '{expected}', got '{response}'"

                    # Run custom validators
                    for validator in test_case.validators:
                        try:
                            validator(response)
                        except AssertionError as e:
                            passed = False
                            error = str(e)
                            break

                    result = TestResult(
                        test_id=f"{test_case.name}_{i}",
                        passed=passed,
                        actual_output=response,
                        expected_output=(
                            test_case.expected_outputs[i]
                            if test_case.expected_outputs
                            and i < len(test_case.expected_outputs)
                            else None
                        ),
                        error=error,
                        latency=latency,
                    )

                except Exception as e:
                    result = TestResult(
                        test_id=f"{test_case.name}_{i}",
                        passed=False,
                        actual_output="",
                        error=str(e),
                        latency=time.time() - start_time,
                    )

                self.results.append(result)

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get test summary statistics."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "avg_latency": (
                sum(r.latency for r in self.results) / total if total > 0 else 0.0
            ),
        }


def run_prompt_regression(
    test_suite: PromptTestSuite,
    llm_fn: Callable[[str], str],
    baseline_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run prompt regression tests.

    Args:
        test_suite: Test suite to run
        llm_fn: LLM function to test
        baseline_file: Optional baseline results to compare against

    Returns:
        Regression test results
    """
    # Run tests
    results = test_suite.run(llm_fn)

    # Load baseline if provided
    regression_detected = False
    differences = []

    if baseline_file and baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)

        # Compare results
        for i, result in enumerate(results):
            if i < len(baseline):
                baseline_result = baseline[i]
                if result.actual_output != baseline_result.get("actual_output"):
                    regression_detected = True
                    differences.append(
                        {
                            "test_id": result.test_id,
                            "baseline": baseline_result.get("actual_output"),
                            "current": result.actual_output,
                        }
                    )

    return {
        "summary": test_suite.get_summary(),
        "results": [asdict(r) for r in results],
        "regression_detected": regression_detected,
        "differences": differences,
    }


def compare_prompt_versions(
    prompt_v1: str, prompt_v2: str, test_inputs: List[str], llm_fn: Callable[[str], str]
) -> Dict[str, Any]:
    """Compare two prompt versions.

    Args:
        prompt_v1: First prompt template
        prompt_v2: Second prompt template
        test_inputs: Test inputs to compare
        llm_fn: LLM function

    Returns:
        Comparison results
    """
    v1_results = []
    v2_results = []

    for input_text in test_inputs:
        # Test version 1
        p1 = prompt_v1.format(input=input_text)
        r1 = llm_fn(p1)
        v1_results.append(r1)

        # Test version 2
        p2 = prompt_v2.format(input=input_text)
        r2 = llm_fn(p2)
        v2_results.append(r2)

    # Calculate differences
    differences = sum(1 for r1, r2 in zip(v1_results, v2_results) if r1 != r2)

    return {
        "v1_results": v1_results,
        "v2_results": v2_results,
        "differences": differences,
        "difference_rate": differences / len(test_inputs) if test_inputs else 0.0,
    }
