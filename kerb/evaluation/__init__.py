"""Evaluation utilities for LLM applications.

This module provides comprehensive evaluation tools for assessing LLM outputs:

Ground Truth Metrics:
    calculate_bleu() - BLEU score for n-gram overlap
    calculate_rouge() - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    calculate_meteor() - METEOR score with precision, recall, and word order
    calculate_exact_match() - Exact match evaluation
    calculate_f1_score() - Token-level F1 score
    calculate_semantic_similarity() - Semantic similarity between texts

Quality Assessment:
    assess_coherence() - Coherence and logical flow
    assess_fluency() - Fluency and naturalness
    assess_faithfulness() - Faithfulness to source material
    assess_answer_relevance() - Answer relevance to question
    detect_hallucination() - Detect unfounded claims

LLM-as-Judge:
    llm_as_judge() - Use LLM to judge output quality
    pairwise_comparison() - Compare two outputs using LLM

A/B Testing:
    ab_test() - Statistical A/B testing of outputs
    compare_outputs() - Multi-output comparison with rankings

Benchmarking:
    run_benchmark() - Run benchmark on test cases
    benchmark_prompts() - Benchmark multiple prompts

Statistical Analysis:
    calculate_statistics() - Statistical measures (mean, median, stdev, etc.)
    confidence_interval() - Confidence intervals for scores

Data Classes:
    EvaluationResult - Result with score and details
    ComparisonResult - Result of comparing outputs
    BenchmarkResult - Benchmark run results
    TestCase - Test case definition

Enums:
    EvaluationMetric - Standard evaluation metrics
    JudgmentCriterion - Criteria for LLM-as-judge

Examples:
    >>> # Common usage - core classes and metrics
    >>> from kerb.evaluation import EvaluationResult, calculate_bleu, calculate_rouge
    >>>
    >>> # Specialized imports - metrics module
    >>> from kerb.evaluation.metrics import (
    ...     calculate_meteor,
    ...     calculate_semantic_similarity
    ... )
    >>>
    >>> # Quality assessment
    >>> from kerb.evaluation.quality import (
    ...     assess_coherence,
    ...     detect_hallucination
    ... )
    >>>
    >>> # LLM-as-judge
    >>> from kerb.evaluation.judges import llm_as_judge, pairwise_comparison
    >>>
    >>> # Benchmarking
    >>> from kerb.evaluation.benchmarks import run_benchmark
"""

# Submodule imports for specialized functionality
from . import benchmarks, comparison, judges, metrics, quality, statistics
# Import benchmarking functions to top-level
from .benchmarks import benchmark_prompts, run_benchmark
# Import comparison functions to top-level
from .comparison import ab_test, compare_outputs
# Import LLM-as-judge functions to top-level
from .judges import llm_as_judge, pairwise_comparison
from .metrics import (calculate_bleu, calculate_exact_match,
                      calculate_f1_score, calculate_meteor, calculate_rouge,
                      calculate_semantic_similarity)
# Import commonly used quality functions to top-level
from .quality import (assess_answer_relevance, assess_coherence,
                      assess_faithfulness, assess_fluency,
                      detect_hallucination)
# Import statistics functions to top-level
from .statistics import calculate_statistics, confidence_interval
# Top-level imports: Core classes and most common functions
from .types import (BenchmarkResult, ComparisonResult, EvaluationMetric,
                    EvaluationResult, JudgmentCriterion, TestCase)

__all__ = [
    # Core data classes and enums
    "EvaluationResult",
    "ComparisonResult",
    "BenchmarkResult",
    "TestCase",
    "EvaluationMetric",
    "JudgmentCriterion",
    # Submodules
    "metrics",
    "quality",
    "judges",
    "comparison",
    "benchmarks",
    "statistics",
    # Ground truth comparison metrics (from metrics)
    "calculate_bleu",
    "calculate_rouge",
    "calculate_meteor",
    "calculate_exact_match",
    "calculate_f1_score",
    "calculate_semantic_similarity",
    # Quality assessment (from quality)
    "assess_coherence",
    "assess_fluency",
    "assess_faithfulness",
    "assess_answer_relevance",
    "detect_hallucination",
    # LLM-as-judge (from judges)
    "llm_as_judge",
    "pairwise_comparison",
    # A/B testing and comparison (from comparison)
    "ab_test",
    "compare_outputs",
    # Benchmarking (from benchmarks)
    "run_benchmark",
    "benchmark_prompts",
    # Statistical analysis (from statistics)
    "calculate_statistics",
    "confidence_interval",
]
