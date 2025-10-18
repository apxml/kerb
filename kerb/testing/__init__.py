"""Testing utilities for LLM applications.

This module provides comprehensive testing tools for LLM development:

Mock LLM Responses:
    MockLLM - Mock LLM provider with configurable responses
    MockStreamingLLM - Mock streaming LLM responses
    create_mock_llm() - Helper to create mock LLM instances

Response Fixtures:
    PromptFixture - Fixture for prompt-response pairs
    ResponseFixture - Fixture for deterministic responses
    FixtureManager - Manage and organize test fixtures
    load_fixtures() - Load fixtures from file
    save_fixtures() - Save fixtures to file

Deterministic Testing:
    DeterministicResponseGenerator - Generate consistent test responses
    SeededResponseGenerator - Seeded random responses
    PatternResponseGenerator - Pattern-based responses

Response Recording:
    ResponseRecorder - Record actual LLM responses
    RecordingSession - Context manager for recording sessions
    replay_responses() - Replay recorded responses

Dataset Management:
    TestDataset - Dataset for evaluation testing
    create_dataset() - Create test datasets
    load_dataset() - Load datasets from various formats
    split_dataset() - Split into train/val/test
    augment_dataset() - Augment datasets with variations

Prompt Testing:
    PromptTestCase - Single prompt test case
    PromptTestSuite - Collection of prompt tests
    run_prompt_regression() - Run regression tests on prompts
    compare_prompt_versions() - Compare prompt versions

Assertion Helpers:
    assert_response_contains() - Check if response contains text
    assert_response_matches() - Check regex match
    assert_response_json() - Validate JSON response
    assert_response_length() - Check response length
    assert_response_quality() - Quality assertions
    assert_no_hallucination() - Check for hallucinations
    assert_safety_compliance() - Check safety guidelines

Output Validators:
    validate_json_schema() - Validate JSON against schema
    validate_code_syntax() - Validate code syntax
    validate_format() - Validate output format
    validate_consistency() - Check consistency across generations

Diff and Comparison:
    diff_responses() - Diff two responses
    compare_responses() - Compare multiple responses
    highlight_differences() - Highlight key differences

Snapshot Testing:
    SnapshotManager - Manage response snapshots
    create_snapshot() - Create snapshot from response
    compare_snapshot() - Compare against snapshot
    update_snapshot() - Update existing snapshot

Test Doubles:
    StubEmbedding - Stub embedding model
    StubRetriever - Stub retrieval system
    StubVectorStore - Stub vector store
    create_test_double() - Factory for test doubles

Performance Testing:
    measure_latency() - Measure response latency
    measure_throughput() - Measure throughput
    benchmark_prompts() - Benchmark prompt performance
    PerformanceReport - Performance test report

Cost Tracking:
    CostTracker - Track testing costs
    estimate_test_cost() - Estimate cost before running
    get_cost_report() - Get cost breakdown

Utilities:
    seed_randomness() - Set random seed for reproducibility
    capture_warnings() - Capture warning messages
    isolate_test() - Isolation context manager
    cleanup_resources() - Clean up test resources

Data Classes:
    MockResponse - Mock LLM response
    TestCase - Test case definition
    TestResult - Test execution result
    FixtureData - Fixture data container

Enums:
    MockBehavior - Mock behavior modes
    FixtureFormat - Fixture file formats
"""

# Submodule imports for specialized functionality
from . import (assertions, comparison, cost, datasets, doubles, fixtures,
               performance, prompts, recording, snapshots, utils, validators)
from .assertions import (assert_response_contains, assert_response_json,
                         assert_response_quality)
from .datasets import TestDataset, create_dataset, load_dataset
from .fixtures import FixtureManager, load_fixtures, save_fixtures
from .mocking import MockLLM, MockStreamingLLM, create_mock_llm
# Top-level imports: Core types and most common functionality
from .types import (CostReport, FixtureData,  # Enums; Data classes
                    FixtureFormat, MockBehavior, MockResponse,
                    PerformanceMetrics, PromptFixture, PromptTestCase,
                    ResponseFixture, SnapshotData, TestCase, TestResult)

__all__ = [
    # Enums
    "MockBehavior",
    "FixtureFormat",
    # Data classes
    "MockResponse",
    "TestCase",
    "TestResult",
    "FixtureData",
    "PromptFixture",
    "ResponseFixture",
    "PromptTestCase",
    "SnapshotData",
    "PerformanceMetrics",
    "CostReport",
    # Mock LLM
    "MockLLM",
    "MockStreamingLLM",
    "create_mock_llm",
    # Fixtures
    "FixtureManager",
    "load_fixtures",
    "save_fixtures",
    # Datasets
    "TestDataset",
    "create_dataset",
    "load_dataset",
    # Assertions
    "assert_response_contains",
    "assert_response_json",
    "assert_response_quality",
    # Submodules
    "fixtures",
    "assertions",
    "validators",
    "comparison",
    "snapshots",
    "doubles",
    "performance",
    "cost",
    "utils",
    "recording",
    "prompts",
    "datasets",
]
