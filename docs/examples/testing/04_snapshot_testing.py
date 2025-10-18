"""Snapshot Testing Example

This example demonstrates snapshot-based testing for detecting unintended changes
in LLM prompt outputs.

Main concepts:
- Creating snapshots of expected outputs
- Comparing current outputs against snapshots
- Updating snapshots when changes are intentional
- Detecting regressions in prompt behavior
- Version controlling prompt outputs

Use cases for LLM developers:
- Regression testing for prompt changes
- Detecting unintended output variations
- Tracking prompt evolution over time
- Ensuring consistent responses
- Validating prompt template updates
- Team collaboration on prompt engineering
"""

from pathlib import Path
from kerb.testing.snapshots import (
    SnapshotManager,
    create_snapshot,
    compare_snapshot,
    update_snapshot
)


def mock_llm_generate(prompt: str, version: int = 1) -> str:
    """Mock LLM function with version-specific responses."""
    responses_v1 = {
        "summarize_ml": "ML is a subset of AI that learns from data.",
        "explain_python": "Python is a high-level programming language.",
        "code_fibonacci": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
    }
    
    responses_v2 = {
        "summarize_ml": "Machine learning is an AI subset enabling systems to learn from data without explicit programming.",
        "explain_python": "Python is a versatile, high-level programming language known for readability.",
        "code_fibonacci": "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    }
    
    # Extract key from prompt
    if "machine learning" in prompt.lower():
        key = "summarize_ml"
    elif "python" in prompt.lower():
        key = "explain_python"
    elif "fibonacci" in prompt.lower():
        key = "code_fibonacci"
    else:
        return "Default response"
    
    return responses_v2[key] if version == 2 else responses_v1[key]


def main():
    """Run snapshot testing examples."""
    
    print("="*80)
    print("SNAPSHOT TESTING EXAMPLE")
    print("="*80)
    
    # Setup test directory
    snapshot_dir = Path("temp_test_snapshots")
    
    # Example 1: Creating initial snapshots
    print("\n1. CREATING INITIAL SNAPSHOTS")
    print("-"*80)
    
    manager = SnapshotManager(snapshot_dir)
    
    # Create snapshots for baseline responses
    prompts = {
        "ml_summary": "Summarize what machine learning is",
        "python_explain": "Explain what Python is",
        "fibonacci_code": "Write a Fibonacci function"
    }
    
    print("Creating baseline snapshots...")
    for name, prompt in prompts.items():
        response = mock_llm_generate(prompt, version=1)
        snapshot = manager.create_snapshot(
            name=name,
            content=response,
            metadata={"prompt": prompt, "version": 1}
        )
        print(f"  Created snapshot: {name}")
        print(f"    Content hash: {snapshot.hash[:16]}...")
    
    # Example 2: Comparing against snapshots (matching)
    print("\n2. COMPARING AGAINST SNAPSHOTS - MATCHING")
    print("-"*80)
    
    print("Testing with same responses...")
    for name, prompt in prompts.items():
        response = mock_llm_generate(prompt, version=1)
        matches, diff = manager.compare_snapshot(name, response)
        
        if matches:
            print(f"  {name}: PASS (no changes)")
        else:
            print(f"  {name}: FAIL (changes detected)")
            if diff:
                print(f"    Diff: {diff[:100]}...")
    
    # Example 3: Detecting changes (regression)
    print("\n3. DETECTING CHANGES - REGRESSION TEST")
    print("-"*80)
    
    print("Testing with modified responses (v2)...")
    for name, prompt in prompts.items():
        # Using v2 which has different responses
        response = mock_llm_generate(prompt, version=2)
        matches, diff = manager.compare_snapshot(name, response)
        
        if matches:
            print(f"  {name}: PASS (no changes)")
        else:
            print(f"  {name}: FAIL (changes detected)")
            print(f"    This indicates a regression or intentional change")
    
    # Example 4: Inspecting snapshot details
    print("\n4. INSPECTING SNAPSHOT DETAILS")
    print("-"*80)
    
    snapshot = manager.load_snapshot("ml_summary")
    if snapshot:
        print(f"Snapshot: ml_summary")
        print(f"  Content: {snapshot.content[:80]}...")
        print(f"  Hash: {snapshot.hash}")
        print(f"  Created: {snapshot.created_at}")
        print(f"  Metadata: {snapshot.metadata}")
    
    # Example 5: Updating snapshots
    print("\n5. UPDATING SNAPSHOTS")
    print("-"*80)
    
    print("Scenario: Intentional prompt improvement")
    print("Updating snapshots to new expected outputs...")
    
    # Update snapshots with v2 responses
    for name, prompt in prompts.items():
        response = mock_llm_generate(prompt, version=2)
        updated = manager.update_snapshot(
            name=name,
            content=response,
            metadata={"prompt": prompt, "version": 2, "updated": True}
        )
        print(f"  Updated snapshot: {name}")
    
    # Verify updates
    print("\nVerifying updates...")
    for name, prompt in prompts.items():
        response = mock_llm_generate(prompt, version=2)
        matches, diff = manager.compare_snapshot(name, response)
        print(f"  {name}: {'PASS' if matches else 'FAIL'}")
    
    # Example 6: Convenience functions
    print("\n6. USING CONVENIENCE FUNCTIONS")
    print("-"*80)
    
    # Create snapshot using convenience function
    content = "This is a test response for convenience function demo"
    snapshot = create_snapshot(
        name="convenience_test",
        content=content,
        snapshot_dir=snapshot_dir
    )
    print(f"Created snapshot with convenience function: {snapshot.name}")
    
    # Compare using convenience function
    matches, diff = compare_snapshot(
        name="convenience_test",
        content=content,
        snapshot_dir=snapshot_dir
    )
    print(f"Comparison result: {'MATCH' if matches else 'DIFF'}")
    
    # Example 7: Multi-version testing
    print("\n7. MULTI-VERSION SNAPSHOT TESTING")
    print("-"*80)
    
    # Create versioned snapshots
    versions_dir = snapshot_dir / "versions"
    
    for version in [1, 2]:
        version_manager = SnapshotManager(versions_dir / f"v{version}")
        
        prompt = "Explain what Python is"
        response = mock_llm_generate(prompt, version=version)
        
        version_manager.create_snapshot(
            name="python_explanation",
            content=response,
            metadata={"version": version}
        )
        print(f"Created v{version} snapshot")
    
    # Compare versions
    v1_manager = SnapshotManager(versions_dir / "v1")
    v2_manager = SnapshotManager(versions_dir / "v2")
    
    v1_snapshot = v1_manager.load_snapshot("python_explanation")
    v2_snapshot = v2_manager.load_snapshot("python_explanation")
    
    if v1_snapshot and v2_snapshot:
        print("\nVersion comparison:")
        print(f"  V1: {v1_snapshot.content[:60]}...")
        print(f"  V2: {v2_snapshot.content[:60]}...")
        print(f"  Changed: {v1_snapshot.hash != v2_snapshot.hash}")
    
    # Example 8: Snapshot-based regression suite
    print("\n8. REGRESSION TEST SUITE")
    print("-"*80)
    
    def run_regression_suite(
        test_cases: dict,
        llm_func,
        snapshot_manager: SnapshotManager
    ) -> dict:
        """Run a suite of regression tests against snapshots."""
        results = {}
        
        for name, prompt in test_cases.items():
            response = llm_func(prompt)
            matches, diff = snapshot_manager.compare_snapshot(name, response)
            results[name] = {
                "passed": matches,
                "diff": diff if not matches else None
            }
        
        return results
    
    # Run regression suite
    test_cases = {
        "ml_summary": "Summarize what machine learning is",
        "python_explain": "Explain what Python is",
    }
    
    print("Running regression test suite...")
    results = run_regression_suite(
        test_cases,
        lambda p: mock_llm_generate(p, version=2),
        manager
    )
    
    passed = sum(1 for r in results.values() if r["passed"])
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    for name, result in results.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {name}: {status}")
    
    # Example 9: Workflow summary
    print("\n9. SNAPSHOT TESTING WORKFLOW")
    print("-"*80)
    
    workflow_steps = [
        "1. Create snapshots from known-good outputs",
        "2. Store snapshots in version control",
        "3. Run tests comparing current outputs to snapshots",
        "4. When test fails, review if change is:",
        "   - Bug (regression) -> fix the code",
        "   - Intentional improvement -> update snapshot",
        "5. Update snapshots when prompted improvements are verified",
        "6. Commit updated snapshots to version control"
    ]
    
    print("Typical snapshot testing workflow:")
    for step in workflow_steps:
        print(f"  {step}")
    
    # Cleanup
    print("\n10. CLEANUP")
    print("-"*80)
    import shutil
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
        print("Cleaned up test snapshots directory")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("- Snapshots detect unintended prompt changes")
    print("- Version control snapshots like code")
    print("- Update snapshots for intentional improvements")
    print("- Great for regression testing prompts")
    print("- Enables confident prompt refactoring")
    print("- Supports team collaboration on prompts")


if __name__ == "__main__":
    main()
