"""Retry and Error Handling Example

This example demonstrates robust error handling and retry strategies.

Main concepts:
- Using retry_with_exponential_backoff
- Handling API errors gracefully
- Implementing fallback strategies
- Custom retry logic
- Error recovery patterns
"""

import time
from kerb.generation import generate, ModelName
from kerb.generation.utils import retry_with_exponential_backoff


def example_basic_retry():
    """Demonstrate basic retry with exponential backoff."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Retry with Exponential Backoff")
    print("="*80)
    
    attempt_count = {"count": 0}
    
    def unreliable_request():
        """Simulates an unreliable API call."""
        attempt_count["count"] += 1
        print(f"  Attempt {attempt_count['count']}...")
        
        # Simulate failure on first 2 attempts
        if attempt_count["count"] < 3:
            raise Exception(f"Simulated failure (attempt {attempt_count['count']})")
        
        return generate(
            "What is Python?",
            model=ModelName.GPT_4O_MINI,
            max_tokens=30
        )
    
    print("\nTrying unreliable request with retry logic...\n")
    
    try:
        response = retry_with_exponential_backoff(
            unreliable_request,
            max_retries=3,
            initial_delay=0.5
        )
        print(f"\nSuccess after {attempt_count['count']} attempts!")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Failed after all retries: {e}")


def example_error_handling():
    """Handle different types of errors."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Error Type Handling")
    print("="*80)
    
    test_cases = [
        {
            "name": "Valid Request",
            "model": ModelName.GPT_4O_MINI,
            "prompt": "What is an API?"
        },
        {
            "name": "Invalid Model",
            "model": "invalid-model-xyz",
            "prompt": "Test prompt"
        },
        {
            "name": "Empty Prompt",
            "model": ModelName.GPT_4O_MINI,
            "prompt": ""
        },
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        try:
            response = generate(
                test["prompt"],
                model=test["model"],
                max_tokens=30
            )
            print(f"  Success: {response.content[:50]}...")
        except ValueError as e:
            print(f"  ValueError: {e}")
        except KeyError as e:
            print(f"  KeyError: {e}")
        except Exception as e:
            print(f"  Error ({type(e).__name__}): {str(e)[:100]}")


def example_fallback_strategy():
    """Implement fallback to alternative models on failure."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Fallback Strategy")
    print("="*80)
    
    prompt = "Explain the concept of CI/CD in one sentence."
    
    # Try models in order of preference
    fallback_models = [
        ModelName.GPT_4O,
        ModelName.GPT_4O_MINI,
        ModelName.GPT_35_TURBO,
    ]
    
    print(f"\nPrompt: {prompt}")
    print("\nTrying models in order of preference...\n")
    
    for i, model in enumerate(fallback_models, 1):
        print(f"{i}. Trying {model.value}...", end=" ")
        try:
            response = generate(
                prompt,
                model=model,
                max_tokens=50
            )
            print("Success!")
            print(f"\nUsed model: {model.value}")
            print(f"Response: {response.content}")
            print(f"Cost: ${response.cost:.6f}")
            break
        except Exception as e:
            print(f"Failed ({type(e).__name__})")
            if i == len(fallback_models):
                print("\nAll fallback models failed!")


def example_timeout_handling():
    """Handle timeouts and slow responses."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Timeout Handling")
    print("="*80)
    
    import signal
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Request timed out")
    
    prompt = "Explain machine learning."
    timeout_seconds = 10
    
    print(f"\nPrompt: {prompt}")
    print(f"Timeout: {timeout_seconds}s\n")
    
    # Note: signal.alarm only works on Unix systems
    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            response = generate(
                prompt,
                model=ModelName.GPT_4O_MINI,
                max_tokens=100
            )
            signal.alarm(0)  # Cancel alarm
            print(f"Success (completed in time)")
            print(f"Response: {response.content[:80]}...")
        except TimeoutError:
            signal.alarm(0)
            print("Request timed out!")
    except AttributeError:
        # signal.alarm not available on Windows
        print("Timeout handling via signal.alarm not available on this system")
        print("Using try-except instead...\n")
        try:
            response = generate(
                prompt,
                model=ModelName.GPT_4O_MINI,
                max_tokens=100
            )
            print(f"Response: {response.content[:80]}...")
        except Exception as e:
            print(f"Error: {e}")


def example_retry_with_validation():
    """Retry until response meets validation criteria."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Retry with Validation")
    print("="*80)
    
    def generate_and_validate(prompt: str, min_length: int = 50):
        """Generate response and validate it meets criteria."""
        response = generate(
            prompt,
            model=ModelName.GPT_4O_MINI,
            temperature=0.8
        )
        
        if len(response.content) < min_length:
            raise ValueError(f"Response too short: {len(response.content)} < {min_length}")
        
        return response
    
    prompt = "Define polymorphism in OOP."
    min_length = 50
    
    print(f"\nPrompt: {prompt}")
    print(f"Requirement: Response must be at least {min_length} characters\n")
    
    def make_request():
        return generate_and_validate(prompt, min_length)
    
    try:
        response = retry_with_exponential_backoff(
            make_request,
            max_retries=3,
            initial_delay=0.5
        )
        print(f"Success!")
        print(f"Response ({len(response.content)} chars): {response.content}")
    except Exception as e:
        print(f"Failed validation after retries: {e}")


def example_graceful_degradation():
    """Gracefully degrade functionality on errors."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Graceful Degradation")
    print("="*80)
    
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
    ]
    
    print(f"\nProcessing {len(prompts)} prompts with error recovery...\n")
    
    results = []
    errors = []
    
    for i, prompt in enumerate(prompts, 1):
        try:
            response = generate(
                prompt,
                model=ModelName.GPT_4O_MINI,
                max_tokens=50
            )
            results.append({
                "prompt": prompt,
                "response": response.content,
                "status": "success"
            })
            print(f"[{i}/{len(prompts)}] Success: {prompt}")
        except Exception as e:
            errors.append({
                "prompt": prompt,
                "error": str(e),
                "status": "failed"
            })
            print(f"[{i}/{len(prompts)}] Failed: {prompt}")
            # Continue processing other prompts
            continue
    
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    print(f"Successful: {len(results)}/{len(prompts)}")
    print(f"Failed: {len(errors)}/{len(prompts)}")
    
    if results:
        print("\nSuccessful responses available for use")
    
    if errors:
        print(f"\nErrors logged for {len(errors)} prompts:")
        for error in errors:
            print(f"  - {error['prompt']}: {error['error'][:50]}...")


def main():
    """Run all retry and error handling examples."""
    print("\n" + "#"*80)
    print("# RETRY AND ERROR HANDLING EXAMPLES")
    print("#"*80)
    
    try:
        example_basic_retry()
    except Exception as e:
        print(f"\nExample 1 Error: {e}")
    
    try:
        example_error_handling()
    except Exception as e:
        print(f"\nExample 2 Error: {e}")
    
    try:
        example_fallback_strategy()
    except Exception as e:
        print(f"\nExample 3 Error: {e}")
    
    try:
        example_timeout_handling()
    except Exception as e:
        print(f"\nExample 4 Error: {e}")
    
    try:
        example_retry_with_validation()
    except Exception as e:
        print(f"\nExample 5 Error: {e}")
    
    try:
        example_graceful_degradation()
    except Exception as e:
        print(f"\nExample 6 Error: {e}")
    
    print("\n" + "#"*80)
    print("# Examples completed")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
