"""Data types and enums for testing utilities."""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class MockBehavior(Enum):
    """Behavior modes for mock LLM."""
    FIXED = "fixed"  # Return fixed responses
    SEQUENTIAL = "sequential"  # Return responses in sequence
    RANDOM = "random"  # Return random responses
    PATTERN = "pattern"  # Match patterns and return responses
    CALLABLE = "callable"  # Use callable to generate responses


class FixtureFormat(Enum):
    """Supported fixture file formats."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    YAML = "yaml"


@dataclass
class MockResponse:
    """Mock LLM response."""
    content: str
    model: str = "mock-model"
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_generation_response(self):
        """Convert to GenerationResponse format."""
        from ..generation import GenerationResponse, Usage, LLMProvider
        
        return GenerationResponse(
            content=self.content,
            model=self.model,
            provider=LLMProvider.LOCAL,
            usage=Usage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens
            ),
            finish_reason=self.finish_reason,
            latency=self.latency,
            cached=False,
            metadata=self.metadata
        )


@dataclass
class TestCase:
    """Test case definition."""
    id: str
    prompt: str
    expected_output: Optional[str] = None
    expected_patterns: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_fn: Optional[Callable] = None


@dataclass
class TestResult:
    """Test execution result."""
    test_id: str
    passed: bool
    actual_output: str
    expected_output: Optional[str] = None
    error: Optional[str] = None
    latency: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FixtureData:
    """Container for fixture data."""
    prompt: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PromptFixture:
    """Fixture for prompt-response pairs."""
    id: str
    prompt: str
    expected_response: str
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseFixture:
    """Fixture for deterministic responses."""
    pattern: str
    response: str
    response_type: str = "exact"  # exact, template, function
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTestCase:
    """Prompt test case for regression testing."""
    name: str
    prompt_template: str
    test_inputs: List[Dict[str, Any]]
    expected_outputs: Optional[List[str]] = None
    validators: List[Callable] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SnapshotData:
    """Snapshot data for snapshot testing."""
    name: str
    content: str
    hash: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for testing."""
    total_requests: int
    total_latency: float
    avg_latency: float
    min_latency: float
    max_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float  # requests per second
    tokens_per_second: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostReport:
    """Cost tracking report."""
    total_cost: float
    total_tokens: int
    total_requests: int
    cost_by_model: Dict[str, float]
    tokens_by_model: Dict[str, int]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
