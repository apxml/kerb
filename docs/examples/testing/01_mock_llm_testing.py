"""
Mock LLM Testing Example
========================

This example demonstrates how to use MockLLM for testing LLM applications
without making actual API calls.

Main concepts:
- Creating MockLLM instances with different behaviors
- Using pattern-based responses
- Sequential and random response modes
- Tracking call history and assertions
- Testing LLM-powered functions without API costs

Use cases for LLM developers:
- Unit testing chatbots and agents
- Testing prompt templates before production
- Validating retry logic and error handling
- Development without API quota concerns
"""

from kerb.testing import MockLLM, MockBehavior


def chatbot_summarize(llm, text: str) -> str:
    """Example chatbot function that summarizes text."""
    prompt = f"Please summarize the following text concisely:\n\n{text}"
    response = llm.generate(prompt)
    return response.content


def chatbot_translate(llm, text: str, target_lang: str) -> str:
    """Example chatbot function that translates text."""

# %%
# Setup and Imports
# -----------------
    prompt = f"Translate to {target_lang}: {text}"
    response = llm.generate(prompt)
    return response.content



# %%
# Chatbot Classify
# ----------------

def chatbot_classify(llm, text: str) -> str:
    """Example chatbot function that classifies text sentiment."""
    prompt = f"Classify the sentiment of this text as positive, negative, or neutral: {text}"
    response = llm.generate(prompt)
    return response.content


def main():
    """Run mock LLM testing examples."""
    
    print("="*80)
    print("MOCK LLM TESTING EXAMPLE")
    print("="*80)
    
    # Example 1: Fixed response behavior
    print("\n1. FIXED RESPONSE BEHAVIOR")
    print("-"*80)
    
    mock_llm = MockLLM(
        responses="This is a concise summary of the text.",
        behavior=MockBehavior.FIXED,
        latency=0.01  # Very fast for testing
    )
    
    result = chatbot_summarize(mock_llm, "Long article text here...")
    print(f"Summary: {result}")
    print(f"Total calls made: {mock_llm.call_count}")
    
    # Test that the function was called
    mock_llm.assert_called()
    print("Assertion passed: Mock was called")
    
    # Example 2: Sequential responses
    print("\n2. SEQUENTIAL RESPONSE BEHAVIOR")
    print("-"*80)
    
    mock_llm = MockLLM(
        responses=[
            "Positive",
            "Negative", 
            "Neutral",
            "Positive"
        ],
        behavior=MockBehavior.SEQUENTIAL,
        latency=0.01
    )
    
    sentiments = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special.",
        "Amazing experience!"
    ]
    
    for text in sentiments:
        result = chatbot_classify(mock_llm, text)
        print(f"Text: '{text[:30]}...' -> {result}")
    
    print(f"Total classifications: {mock_llm.call_count}")
    
    # Example 3: Pattern-based responses
    print("\n3. PATTERN-BASED RESPONSES")
    print("-"*80)
    
    mock_llm = MockLLM(
        responses={
            r"summarize": "This is a brief summary.",
            r"translate.*spanish": "Hola, este es el texto traducido.",
            r"translate.*french": "Bonjour, c'est le texte traduit.",
            r"classify.*sentiment": "Positive",
        },
        behavior=MockBehavior.PATTERN,
        default_response="I don't understand the request.",
        latency=0.01
    )
    
    # Test different patterns
    test_cases = [
        ("summarize this article", "summarize"),
        ("translate to spanish", "spanish translation"),
        ("translate to french", "french translation"),
        ("classify sentiment", "sentiment"),
        ("unknown request", "default"),
    ]
    
    for prompt, description in test_cases:
        response = mock_llm.generate(prompt)
        print(f"{description}: {response.content}")
    
    # Example 4: Call history inspection
    print("\n4. CALL HISTORY INSPECTION")
    print("-"*80)
    
    mock_llm = MockLLM(
        responses="Mock response",
        behavior=MockBehavior.FIXED,
        latency=0.01
    )
    
    # Make several calls
    chatbot_summarize(mock_llm, "Text 1")
    chatbot_translate(mock_llm, "Text 2", "Spanish")
    chatbot_classify(mock_llm, "Text 3")
    
    print(f"Total calls: {mock_llm.call_count}")
    print("\nCall history:")
    for i, call in enumerate(mock_llm.call_history, 1):
        print(f"\nCall {i}:")
        print(f"  Prompt preview: {call['prompt'][:60]}...")
        print(f"  Timestamp: {call['timestamp']}")
    
    # Get last call
    last_call = mock_llm.get_last_call()
    print(f"\nLast call prompt: {last_call['prompt'][:60]}...")
    
    # Example 5: Testing with assertions
    print("\n5. TESTING WITH ASSERTIONS")
    print("-"*80)
    
    mock_llm = MockLLM(
        responses="This is a test response about Python programming.",
        behavior=MockBehavior.FIXED,
        latency=0.01
    )
    
    result = chatbot_summarize(mock_llm, "Python tutorial")
    
    # Assert the mock was called
    mock_llm.assert_called()
    print("Check: Mock was called")
    
    # Assert it was called with specific content
    mock_llm.assert_called_with("Python")
    print("Check: Called with 'Python' in prompt")
    
    # Example 6: Random responses (for testing variability handling)
    print("\n6. RANDOM RESPONSE BEHAVIOR")
    print("-"*80)
    
    mock_llm = MockLLM(
        responses=[
            "Response variant A",
            "Response variant B",
            "Response variant C",
        ],
        behavior=MockBehavior.RANDOM,
        latency=0.01
    )
    
    print("Getting 5 random responses:")
    for i in range(5):
        response = mock_llm.generate("Test prompt")
        print(f"  Response {i+1}: {response.content}")
    
    # Example 7: Token counting and metrics
    print("\n7. TOKEN COUNTING AND METRICS")
    print("-"*80)
    

# %%
# Custom Token Counter
# --------------------

    def custom_token_counter(text: str) -> int:
        """More accurate token counting."""
        # Rough approximation: ~4 chars per token
        return max(1, len(text) // 4)
    
    mock_llm = MockLLM(
        responses="This is a detailed response with multiple sentences to test token counting.",
        behavior=MockBehavior.FIXED,
        latency=0.02,
        token_calculator=custom_token_counter
    )
    
    prompt = "Give me a detailed explanation of machine learning."
    response = mock_llm.generate(prompt)
    
    print(f"Prompt tokens: {response.prompt_tokens}")
    print(f"Completion tokens: {response.completion_tokens}")
    print(f"Total tokens: {response.prompt_tokens + response.completion_tokens}")
    print(f"Latency: {response.latency}s")
    
    # Example 8: Reset functionality for test isolation
    print("\n8. RESET FUNCTIONALITY")
    print("-"*80)
    
    mock_llm = MockLLM(responses="Test", behavior=MockBehavior.FIXED)
    
    # Make some calls
    for i in range(3):
        mock_llm.generate(f"Test {i}")
    
    print(f"Calls before reset: {mock_llm.call_count}")
    
    # Reset for next test
    mock_llm.reset()
    
    print(f"Calls after reset: {mock_llm.call_count}")
    print(f"History length: {len(mock_llm.call_history)}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("- MockLLM enables testing without API calls")
    print("- Different behaviors support various testing scenarios")
    print("- Pattern matching useful for testing routing logic")
    print("- Call history helps verify interactions")
    print("- Token counting aids in cost estimation testing")


if __name__ == "__main__":
    main()
