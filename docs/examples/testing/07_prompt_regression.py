"""Prompt Regression Testing Example

This example demonstrates how to test prompt changes and detect regressions
when updating prompt templates.

Main concepts:
- Testing prompt template versions
- Comparing outputs across versions
- Detecting unintended changes
- A/B testing prompts
- Prompt evolution tracking
- Regression detection

Use cases for LLM developers:
- Validating prompt improvements
- Ensuring prompt changes don't break functionality
- Comparing prompt variations
- Testing system prompt updates
- Tracking prompt engineering iterations
- Maintaining prompt quality
"""

from typing import Dict, List, Tuple
from kerb.testing import (
    MockLLM,
    MockBehavior,
    TestDataset,
    assert_response_contains
)
from kerb.testing.snapshots import SnapshotManager
from pathlib import Path


class PromptTemplate:
    """Simple prompt template class."""
    
    def __init__(self, template: str, version: str):
        self.template = template
        self.version = version
    
    def format(self, **kwargs) -> str:
        """Format template with variables."""
        return self.template.format(**kwargs)


def mock_llm_with_prompt(prompt: str) -> str:
    """Mock LLM that responds based on prompt structure."""
    # Simulate different responses based on prompt characteristics
    if "step by step" in prompt.lower():
        return "Step 1: Analyze the problem\nStep 2: Design solution\nStep 3: Implement"
    elif "concise" in prompt.lower() or "brief" in prompt.lower():
        return "Brief answer here."
    elif "detailed" in prompt.lower():
        return "This is a detailed explanation with multiple paragraphs covering all aspects."
    else:
        return "Standard response to the query."


def main():
    """Run prompt regression testing examples."""
    
    print("="*80)
    print("PROMPT REGRESSION TESTING EXAMPLE")
    print("="*80)
    
    # Example 1: Basic prompt version comparison
    print("\n1. BASIC PROMPT VERSION COMPARISON")
    print("-"*80)
    
    # Version 1: Original prompt
    prompt_v1 = PromptTemplate(
        template="Answer the following question: {question}",
        version="1.0"
    )
    
    # Version 2: Improved prompt
    prompt_v2 = PromptTemplate(
        template="Provide a concise answer to this question: {question}",
        version="2.0"
    )
    
    # Test question
    test_input = {"question": "What is machine learning?"}
    
    # Generate with both versions
    response_v1 = mock_llm_with_prompt(prompt_v1.format(**test_input))
    response_v2 = mock_llm_with_prompt(prompt_v2.format(**test_input))
    
    print(f"V1 prompt: {prompt_v1.template}")
    print(f"V1 response: {response_v1}")
    print(f"\nV2 prompt: {prompt_v2.template}")
    print(f"V2 response: {response_v2}")
    print(f"\nResponse changed: {response_v1 != response_v2}")
    
    # Example 2: Testing with multiple test cases
    print("\n2. TESTING WITH MULTIPLE TEST CASES")
    print("-"*80)
    
    # Create test dataset
    test_cases = [
        {"question": "What is Python?"},
        {"question": "Explain neural networks"},
        {"question": "How does ML work?"},
    ]
    
    print("Testing prompt changes across multiple examples:")
    
    for i, test_case in enumerate(test_cases, 1):
        resp_v1 = mock_llm_with_prompt(prompt_v1.format(**test_case))
        resp_v2 = mock_llm_with_prompt(prompt_v2.format(**test_case))
        
        changed = resp_v1 != resp_v2
        print(f"\n  Test {i}: {test_case['question']}")
        print(f"    Changed: {changed}")
        if changed:
            print(f"    V1 length: {len(resp_v1)} chars")
            print(f"    V2 length: {len(resp_v2)} chars")
    
    # Example 3: Snapshot-based regression testing
    print("\n3. SNAPSHOT-BASED REGRESSION TESTING")
    print("-"*80)
    
    snapshot_dir = Path("temp_prompt_snapshots")
    snapshot_manager = SnapshotManager(snapshot_dir)
    
    # Create baseline snapshots with V1
    print("Creating baseline snapshots with V1 prompts...")
    for i, test_case in enumerate(test_cases):
        prompt = prompt_v1.format(**test_case)
        response = mock_llm_with_prompt(prompt)
        snapshot_manager.create_snapshot(
            name=f"test_case_{i}",
            content=response,
            metadata={"prompt_version": "1.0", "test_case": test_case}
        )
    
    print("Baseline snapshots created")
    
    # Test V2 against baselines
    print("\nTesting V2 prompts against baseline...")
    regressions = []
    
    for i, test_case in enumerate(test_cases):
        prompt = prompt_v2.format(**test_case)
        response = mock_llm_with_prompt(prompt)
        matches, diff = snapshot_manager.compare_snapshot(f"test_case_{i}", response)
        
        if not matches:
            regressions.append(i)
            print(f"  Test {i}: CHANGED")
        else:
            print(f"  Test {i}: UNCHANGED")
    
    if regressions:
        print(f"\nWarning: {len(regressions)} test(s) showed changes")
    else:
        print("\nNo regressions detected")
    
    # Example 4: Quality assertion testing
    print("\n4. QUALITY ASSERTION TESTING")
    print("-"*80)
    
    def test_prompt_quality(prompt_template: PromptTemplate, test_case: dict) -> dict:
        """Test prompt outputs meet quality criteria."""
        prompt = prompt_template.format(**test_case)
        response = mock_llm_with_prompt(prompt)
        
        results = {
            "version": prompt_template.version,
            "passed": True,
            "checks": {}
        }
        
        # Check 1: Response is not empty
        try:
            assert len(response) > 0
            results["checks"]["not_empty"] = "PASS"
        except AssertionError:
            results["checks"]["not_empty"] = "FAIL"
            results["passed"] = False
        
        # Check 2: Response has minimum length
        try:
            assert len(response) >= 10
            results["checks"]["min_length"] = "PASS"
        except AssertionError:
            results["checks"]["min_length"] = "FAIL"
            results["passed"] = False
        
        # Check 3: Response contains relevant content
        question_words = test_case["question"].lower().split()
        relevant_words = ["python", "neural", "network", "ml", "machine", "learning"]
        has_relevant = any(word in response.lower() for word in relevant_words)
        
        results["checks"]["relevance"] = "PASS" if has_relevant else "SKIP"
        
        return results
    
    # Test both versions
    test_case = {"question": "What is machine learning?"}
    
    v1_quality = test_prompt_quality(prompt_v1, test_case)
    v2_quality = test_prompt_quality(prompt_v2, test_case)
    
    print(f"V1 quality checks: {v1_quality['checks']}")
    print(f"V1 passed: {v1_quality['passed']}")
    print(f"\nV2 quality checks: {v2_quality['checks']}")
    print(f"V2 passed: {v2_quality['passed']}")
    
    # Example 5: A/B testing prompts
    print("\n5. A/B TESTING PROMPTS")
    print("-"*80)
    
    # Create three prompt variations
    variations = {
        "A": PromptTemplate("Answer: {question}", "A"),
        "B": PromptTemplate("Provide a detailed answer: {question}", "B"),
        "C": PromptTemplate("Answer step by step: {question}", "C"),
    }
    
    test_question = {"question": "How to learn Python?"}
    
    print("Testing prompt variations:")
    results = {}
    
    for name, template in variations.items():
        prompt = template.format(**test_question)
        response = mock_llm_with_prompt(prompt)
        
        results[name] = {
            "response": response,
            "length": len(response),
            "word_count": len(response.split())
        }
        
        print(f"\n  Variation {name}:")
        print(f"    Response: {response[:60]}...")
        print(f"    Length: {results[name]['length']} chars")
        print(f"    Words: {results[name]['word_count']}")
    
    # Select best variation based on criteria
    best = max(results.items(), key=lambda x: x[1]['word_count'])
    print(f"\nBest variation by word count: {best[0]}")
    
    # Example 6: System prompt regression testing
    print("\n6. SYSTEM PROMPT REGRESSION TESTING")
    print("-"*80)
    
    system_prompts = {
        "v1": "You are a helpful assistant.",
        "v2": "You are a helpful assistant. Be concise and accurate.",
        "v3": "You are a helpful assistant. Provide detailed explanations."
    }
    
    def generate_with_system_prompt(system: str, user_query: str) -> str:
        """Simulate generation with system prompt."""
        combined = f"{system}\n\nUser: {user_query}"
        return mock_llm_with_prompt(combined)
    
    user_query = "Explain Python"
    
    print("Testing system prompt variations:")
    for version, system_prompt in system_prompts.items():
        response = generate_with_system_prompt(system_prompt, user_query)
        print(f"\n  {version}: {system_prompt}")
        print(f"    Response: {response[:50]}...")
    
    # Example 7: Tracking prompt metrics over versions
    print("\n7. TRACKING PROMPT METRICS")
    print("-"*80)
    
    def evaluate_prompt_version(
        template: PromptTemplate,
        test_dataset: List[dict]
    ) -> dict:
        """Evaluate a prompt version on a dataset."""
        metrics = {
            "version": template.version,
            "total_tests": len(test_dataset),
            "avg_response_length": 0,
            "avg_word_count": 0,
            "passed_quality": 0
        }
        
        total_length = 0
        total_words = 0
        
        for test_case in test_dataset:
            prompt = template.format(**test_case)
            response = mock_llm_with_prompt(prompt)
            
            total_length += len(response)
            total_words += len(response.split())
            
            # Simple quality check
            if len(response) >= 10:
                metrics["passed_quality"] += 1
        
        metrics["avg_response_length"] = total_length / len(test_dataset)
        metrics["avg_word_count"] = total_words / len(test_dataset)
        metrics["quality_rate"] = metrics["passed_quality"] / metrics["total_tests"]
        
        return metrics
    
    # Evaluate versions
    v1_metrics = evaluate_prompt_version(prompt_v1, test_cases)
    v2_metrics = evaluate_prompt_version(prompt_v2, test_cases)
    
    print("Prompt version metrics:")
    print(f"\n  V1:")
    print(f"    Avg response length: {v1_metrics['avg_response_length']:.1f}")
    print(f"    Avg word count: {v1_metrics['avg_word_count']:.1f}")
    print(f"    Quality rate: {v1_metrics['quality_rate']:.0%}")
    
    print(f"\n  V2:")
    print(f"    Avg response length: {v2_metrics['avg_response_length']:.1f}")
    print(f"    Avg word count: {v2_metrics['avg_word_count']:.1f}")
    print(f"    Quality rate: {v2_metrics['quality_rate']:.0%}")
    
    # Example 8: Automated regression detection
    print("\n8. AUTOMATED REGRESSION DETECTION")
    print("-"*80)
    
    def detect_regressions(
        old_version: PromptTemplate,
        new_version: PromptTemplate,
        test_cases: List[dict],
        threshold: float = 0.3
    ) -> dict:
        """Detect if new version causes regressions."""
        
        regressions = []
        improvements = []
        unchanged = []
        
        for i, test_case in enumerate(test_cases):
            old_response = mock_llm_with_prompt(old_version.format(**test_case))
            new_response = mock_llm_with_prompt(new_version.format(**test_case))
            
            # Simple similarity check (character-based)
            if old_response == new_response:
                unchanged.append(i)
            else:
                # Check if quality changed significantly
                old_quality = len(old_response.split())
                new_quality = len(new_response.split())
                
                change_ratio = abs(new_quality - old_quality) / max(old_quality, 1)
                
                if change_ratio > threshold:
                    if new_quality < old_quality:
                        regressions.append((i, "shorter response"))
                    else:
                        improvements.append((i, "longer response"))
        
        return {
            "regressions": regressions,
            "improvements": improvements,
            "unchanged": unchanged,
            "regression_count": len(regressions)
        }
    
    results = detect_regressions(prompt_v1, prompt_v2, test_cases)
    
    print(f"Regression analysis:")
    print(f"  Regressions: {results['regression_count']}")
    print(f"  Improvements: {len(results['improvements'])}")
    print(f"  Unchanged: {len(results['unchanged'])}")
    
    if results['regressions']:
        print(f"\n  Detected regressions:")
        for idx, reason in results['regressions']:
            print(f"    Test {idx}: {reason}")
    
    # Cleanup
    print("\n9. CLEANUP")
    print("-"*80)
    import shutil
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
        print("Cleaned up snapshot directory")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("- Test prompt changes before deploying")
    print("- Use snapshots to detect regressions")
    print("- Compare metrics across versions")
    print("- A/B test prompt variations")
    print("- Track quality over iterations")
    print("- Automate regression detection")


if __name__ == "__main__":
    main()
