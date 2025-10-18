"""Data models and enums for evaluation.

This module contains all data classes and enumerations used throughout
the evaluation subpackage.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


# ============================================================================
# Enums
# ============================================================================

class EvaluationMetric(Enum):
    """Standard evaluation metrics."""
    BLEU = "bleu"
    ROUGE_1 = "rouge-1"
    ROUGE_2 = "rouge-2"
    ROUGE_L = "rouge-l"
    METEOR = "meteor"
    BERTSCORE = "bertscore"
    EXACT_MATCH = "exact_match"
    F1 = "f1"
    SEMANTIC_SIMILARITY = "semantic_similarity"


class JudgmentCriterion(Enum):
    """Criteria for LLM-as-judge evaluation."""
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    FAITHFULNESS = "faithfulness"
    CONSISTENCY = "consistency"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EvaluationResult:
    """Result of an evaluation with score and details."""
    metric: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    passed: Optional[bool] = None
    
    def __repr__(self) -> str:
        passed_str = f", passed={self.passed}" if self.passed is not None else ""
        return f"EvaluationResult(metric='{self.metric}', score={self.score:.4f}{passed_str})"


@dataclass
class ComparisonResult:
    """Result of comparing two outputs."""
    output_a_id: str
    output_b_id: str
    winner: Optional[str]  # 'a', 'b', or None for tie
    scores: Dict[str, float]
    confidence: float = 0.0
    reasoning: str = ""
    
    def __repr__(self) -> str:
        return f"ComparisonResult(winner='{self.winner}', confidence={self.confidence:.2f})"


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_score: float
    scores: List[float]
    execution_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        return (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0.0
    
    def __repr__(self) -> str:
        return f"BenchmarkResult(name='{self.name}', pass_rate={self.pass_rate:.1f}%, avg_score={self.average_score:.4f})"


@dataclass
class TestCase:
    """A single test case for evaluation."""
    id: str
    input: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reference_outputs: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return f"TestCase(id='{self.id}', input='{self.input[:30]}...')"
