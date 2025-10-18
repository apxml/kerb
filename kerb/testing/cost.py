"""Cost tracking for testing."""

from typing import Dict, List

from .types import CostReport


class CostTracker:
    """Track testing costs."""
    
    def __init__(self):
        """Initialize cost tracker."""
        self.total_cost = 0.0
        self.total_tokens = 0
        self.requests = 0
        self.costs_by_model: Dict[str, float] = {}
        self.tokens_by_model: Dict[str, int] = {}
    
    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float
    ) -> None:
        """Record a request.
        
        Args:
            model: Model name
            prompt_tokens: Prompt tokens
            completion_tokens: Completion tokens
            cost: Request cost
        """
        self.total_cost += cost
        self.total_tokens += prompt_tokens + completion_tokens
        self.requests += 1
        
        if model not in self.costs_by_model:
            self.costs_by_model[model] = 0.0
            self.tokens_by_model[model] = 0
        
        self.costs_by_model[model] += cost
        self.tokens_by_model[model] += prompt_tokens + completion_tokens
    
    def get_report(self) -> CostReport:
        """Get cost report."""
        return CostReport(
            total_cost=self.total_cost,
            total_tokens=self.total_tokens,
            total_requests=self.requests,
            cost_by_model=self.costs_by_model.copy(),
            tokens_by_model=self.tokens_by_model.copy()
        )
    
    def reset(self) -> None:
        """Reset tracking."""
        self.total_cost = 0.0
        self.total_tokens = 0
        self.requests = 0
        self.costs_by_model.clear()
        self.tokens_by_model.clear()


def estimate_test_cost(
    test_cases: List[str],
    model: str = "gpt-4o-mini",
    avg_completion_tokens: int = 100
) -> float:
    """Estimate cost before running tests.
    
    Args:
        test_cases: List of test prompts
        model: Model to use
        avg_completion_tokens: Average completion tokens
        
    Returns:
        Estimated cost
    """
    from ..generation import MODEL_PRICING
    
    if model not in MODEL_PRICING:
        return 0.0
    
    input_price, output_price = MODEL_PRICING[model]
    
    total_cost = 0.0
    for prompt in test_cases:
        prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
        cost = (prompt_tokens * input_price + avg_completion_tokens * output_price) / 1_000_000
        total_cost += cost
    
    return total_cost


def get_cost_report(tracker: CostTracker) -> str:
    """Get formatted cost report.
    
    Args:
        tracker: CostTracker instance
        
    Returns:
        Formatted report string
    """
    report = tracker.get_report()
    
    lines = [
        "Cost Report",
        "=" * 50,
        f"Total Cost: ${report.total_cost:.4f}",
        f"Total Tokens: {report.total_tokens:,}",
        f"Total Requests: {report.total_requests}",
        "",
        "By Model:",
    ]
    
    for model, cost in report.cost_by_model.items():
        tokens = report.tokens_by_model[model]
        lines.append(f"  {model}: ${cost:.4f} ({tokens:,} tokens)")
    
    return "\n".join(lines)
