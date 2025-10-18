"""Performance testing utilities."""

import time
from typing import Any, Callable, Dict, List

from .types import PerformanceMetrics


def measure_latency(
    fn: Callable, args: List[Any], num_runs: int = 10, warmup: int = 2
) -> PerformanceMetrics:
    """Measure function latency.

    Args:
        fn: Function to measure
        args: Function arguments
        num_runs: Number of runs
        warmup: Number of warmup runs

    Returns:
        PerformanceMetrics object
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)

    # Measure
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        fn(*args)
        latencies.append(time.time() - start)

    latencies.sort()
    total_latency = sum(latencies)

    return PerformanceMetrics(
        total_requests=num_runs,
        total_latency=total_latency,
        avg_latency=total_latency / num_runs,
        min_latency=min(latencies),
        max_latency=max(latencies),
        p50_latency=latencies[len(latencies) // 2],
        p95_latency=latencies[int(len(latencies) * 0.95)],
        p99_latency=latencies[int(len(latencies) * 0.99)],
        throughput=num_runs / total_latency,
        tokens_per_second=0.0,  # Would need token counting
    )


def measure_throughput(
    fn: Callable, args_list: List[List[Any]], duration: float = 10.0
) -> float:
    """Measure throughput (requests per second).

    Args:
        fn: Function to measure
        args_list: List of argument lists
        duration: Test duration in seconds

    Returns:
        Throughput (requests/second)
    """
    start = time.time()
    count = 0

    while time.time() - start < duration:
        args = args_list[count % len(args_list)]
        fn(*args)
        count += 1

    elapsed = time.time() - start
    return count / elapsed


def benchmark_prompts(
    prompts: List[str], llm_fn: Callable[[str], str], num_runs: int = 5
) -> Dict[str, PerformanceMetrics]:
    """Benchmark multiple prompts.

    Args:
        prompts: List of prompts to benchmark
        llm_fn: LLM function
        num_runs: Number of runs per prompt

    Returns:
        Dict mapping prompt to performance metrics
    """
    results = {}

    for i, prompt in enumerate(prompts):
        metrics = measure_latency(llm_fn, [prompt], num_runs)
        results[f"prompt_{i}"] = metrics

    return results


class PerformanceReport:
    """Performance test report."""

    def __init__(self):
        """Initialize report."""
        self.metrics: Dict[str, PerformanceMetrics] = {}

    def add_metrics(self, name: str, metrics: PerformanceMetrics) -> None:
        """Add metrics to report."""
        self.metrics[name] = metrics

    def generate_report(self) -> str:
        """Generate formatted report."""
        lines = ["Performance Report", "=" * 50, ""]

        for name, metrics in self.metrics.items():
            lines.append(f"{name}:")
            lines.append(f"  Requests: {metrics.total_requests}")
            lines.append(f"  Avg Latency: {metrics.avg_latency:.3f}s")
            lines.append(f"  P95 Latency: {metrics.p95_latency:.3f}s")
            lines.append(f"  Throughput: {metrics.throughput:.2f} req/s")
            lines.append("")

        return "\n".join(lines)
