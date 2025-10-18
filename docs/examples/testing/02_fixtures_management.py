"""Fixtures Management Example

This example demonstrates how to create, save, load, and use fixtures for
deterministic LLM testing.

Main concepts:
- Creating reusable prompt-response fixtures
- Saving and loading fixtures from disk
- Using FixtureManager for organization
- Deterministic response generation
- Managing test fixtures across test suites

Use cases for LLM developers:
- Creating golden test sets for regression testing
- Sharing test data across team members
- Version controlling expected outputs
- Building benchmark datasets
- Reproducing issues with specific inputs
"""

from pathlib import Path
from kerb.testing import (
    FixtureManager,
    FixtureData,
    FixtureFormat,
    load_fixtures,
    save_fixtures
)


def main():
    """Run fixtures management examples."""
    
    print("="*80)
    print("FIXTURES MANAGEMENT EXAMPLE")
    print("="*80)
    
    # Example 1: Creating and adding fixtures
    print("\n1. CREATING FIXTURES")
    print("-"*80)
    
    manager = FixtureManager()
    
    # Add fixtures for a code generation task
    manager.add_fixture(
        name="code_gen_fibonacci",
        prompt="Write a Python function to calculate Fibonacci numbers",
        response="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        task_type="code_generation",
        language="python",
        difficulty="easy"
    )
    
    manager.add_fixture(
        name="code_gen_binary_search",
        prompt="Write a Python function for binary search",
        response="def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        task_type="code_generation",
        language="python",
        difficulty="medium"
    )
    
    # Add fixtures for text summarization
    manager.add_fixture(
        name="summarize_article",
        prompt="Summarize: Machine learning is a subset of artificial intelligence...",
        response="Machine learning is an AI subset that enables systems to learn from data without explicit programming.",
        task_type="summarization",
        input_length=100,
        target_length=20
    )
    
    print(f"Created {len(manager.fixtures)} fixtures")
    for name in manager.fixtures.keys():
        print(f"  - {name}")
    
    # Example 2: Saving fixtures to disk
    print("\n2. SAVING FIXTURES")
    print("-"*80)
    
    # Create test directory
    test_dir = Path("temp_test_fixtures")
    test_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    json_path = test_dir / "fixtures.json"
    manager.save(format=FixtureFormat.JSON)
    print(f"Saved fixtures to: {manager.fixtures_dir / 'fixtures.json'}")
    
    # Save using convenience function with custom path
    custom_fixtures = {
        "test1": FixtureData(
            prompt="What is 2+2?",
            response="4",
            metadata={"category": "math"}
        ),
        "test2": FixtureData(
            prompt="What is the capital of France?",
            response="Paris",
            metadata={"category": "geography"}
        )
    }
    
    custom_path = test_dir / "custom_fixtures.json"
    save_fixtures(custom_fixtures, custom_path)
    print(f"Saved custom fixtures to: {custom_path}")
    
    # Example 3: Loading fixtures from disk
    print("\n3. LOADING FIXTURES")
    print("-"*80)
    
    # Load fixtures
    loaded_manager = FixtureManager()
    loaded_manager.load(manager.fixtures_dir / "fixtures.json")
    
    print(f"Loaded {len(loaded_manager.fixtures)} fixtures")
    
    # Access a specific fixture
    fib_fixture = loaded_manager.get_fixture("code_gen_fibonacci")
    if fib_fixture:
        print(f"\nFixture: code_gen_fibonacci")
        print(f"Prompt: {fib_fixture.prompt[:50]}...")
        print(f"Response preview: {fib_fixture.response[:60]}...")
        print(f"Metadata: {fib_fixture.metadata}")
    
    # Example 4: Using fixtures in tests
    print("\n4. USING FIXTURES IN TESTS")
    print("-"*80)
    
    def test_code_generator(llm_func, fixture: FixtureData) -> bool:
        """Test a code generator against a fixture."""
        result = llm_func(fixture.prompt)
        # In real test, you'd check functional equivalence
        # Here we just check it's not empty
        return len(result.strip()) > 0
    
    # Mock LLM function
    def mock_llm(prompt: str) -> str:
        # In real scenario, this would call actual LLM
        for fixture in loaded_manager.fixtures.values():
            if fixture.prompt == prompt:
                return fixture.response
        return "Generated code here..."
    
    # Run test with fixture
    test_fixture = loaded_manager.get_fixture("code_gen_fibonacci")
    if test_fixture:
        passed = test_code_generator(mock_llm, test_fixture)
        print(f"Test with fixture 'code_gen_fibonacci': {'PASS' if passed else 'FAIL'}")
    
    # Example 5: Filtering and organizing fixtures
    print("\n5. FILTERING FIXTURES BY METADATA")
    print("-"*80)
    
    # Add more fixtures with metadata
    manager.add_fixture(
        name="translate_en_to_es",
        prompt="Translate to Spanish: Hello, how are you?",
        response="Hola, como estas?",
        task_type="translation",
        source_lang="en",
        target_lang="es"
    )
    
    manager.add_fixture(
        name="classify_sentiment_pos",
        prompt="Classify sentiment: I love this product!",
        response="positive",
        task_type="classification",
        category="sentiment"
    )
    
    # Filter by task type
    code_fixtures = {
        name: fix for name, fix in manager.fixtures.items()
        if fix.metadata.get("task_type") == "code_generation"
    }
    
    print(f"Code generation fixtures: {len(code_fixtures)}")
    for name in code_fixtures.keys():
        print(f"  - {name}")
    
    # Example 6: Creating test fixture collections
    print("\n6. FIXTURE COLLECTIONS FOR DIFFERENT TASKS")
    print("-"*80)
    
    # Create specialized fixture managers
    qa_manager = FixtureManager(fixtures_dir=Path("temp_test_fixtures/qa"))
    
    qa_pairs = [
        ("What is Python?", "Python is a high-level programming language."),
        ("What is machine learning?", "Machine learning is a branch of AI."),
        ("What is a neural network?", "A neural network is a computational model.")
    ]
    
    for i, (prompt, response) in enumerate(qa_pairs):
        qa_manager.add_fixture(
            name=f"qa_{i+1}",
            prompt=prompt,
            response=response,
            domain="technical"
        )
    
    print(f"Created QA fixture collection: {len(qa_manager.fixtures)} pairs")
    
    # Example 7: Deterministic response generation
    print("\n7. DETERMINISTIC RESPONSE GENERATION")
    print("-"*80)
    
    from kerb.testing.fixtures import DeterministicResponseGenerator
    
    generator = DeterministicResponseGenerator(seed=42)
    
    response_templates = [
        "The answer is A.",
        "The answer is B.",
        "The answer is C.",
        "The answer is D."
    ]
    
    # Same prompt always gets same response
    prompt = "What is the correct answer?"
    
    print("Testing deterministic generation:")
    for i in range(3):
        response = generator.generate(prompt, response_templates)
        print(f"  Attempt {i+1}: {response}")
    
    # Different prompt gets different (but still deterministic) response
    prompt2 = "Select the right option:"
    response2 = generator.generate(prompt2, response_templates)
    print(f"  Different prompt: {response2}")
    
    # Example 8: Versioning fixtures
    print("\n8. FIXTURE VERSIONING")
    print("-"*80)
    
    # Create versioned fixtures
    v1_manager = FixtureManager(fixtures_dir=Path("temp_test_fixtures/v1"))
    v1_manager.add_fixture(
        name="greeting",
        prompt="Say hello",
        response="Hello!",
        version="1.0"
    )
    
    v2_manager = FixtureManager(fixtures_dir=Path("temp_test_fixtures/v2"))
    v2_manager.add_fixture(
        name="greeting",
        prompt="Say hello",
        response="Hello! How can I help you today?",
        version="2.0"
    )
    
    print("Version 1 response:", v1_manager.get_fixture("greeting").response)
    print("Version 2 response:", v2_manager.get_fixture("greeting").response)
    
    # Cleanup
    print("\n9. CLEANUP")
    print("-"*80)
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("Cleaned up test fixtures directory")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("- Fixtures provide deterministic test data")
    print("- FixtureManager organizes fixtures efficiently")
    print("- Fixtures can be saved/loaded for sharing")
    print("- Metadata enables filtering and organization")
    print("- Version control fixtures like code")
    print("- Deterministic generation ensures reproducibility")


if __name__ == "__main__":
    main()
