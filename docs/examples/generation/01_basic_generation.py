"""Basic LLM Generation Example

This example demonstrates fundamental LLM text generation capabilities.

Main concepts:
- Simple text generation using generate()
- Working with different model providers
- Using ModelName enum for type-safe model names
- Converting between message formats
- Inspecting generation responses
"""

from kerb.generation import generate, ModelName, LLMProvider
from kerb.core import Message
from kerb.core.types import MessageRole


def example_simple_string():
    """Generate from a simple string prompt."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple String Generation")
    print("="*80)
    
    # Most basic usage
    response = generate(
        "Write a haiku about AI development",
        model=ModelName.GPT_4O_MINI,
        provider=LLMProvider.OPENAI
    )
    
    print(f"\nModel: {response.model}")
    print(f"Provider: {response.provider.value}")
    print(f"\nGenerated Content:\n{response.content}")
    print(f"\nTokens Used: {response.usage.total_tokens}")
    print(f"Cost: ${response.cost:.6f}")


def example_message_list():
    """Generate from a list of messages."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multi-Message Generation")
    print("="*80)
    
    # Using Message objects for more control
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful Python programming assistant."),
        Message(role=MessageRole.USER, content="Explain list comprehensions in one sentence."),
    ]
    
    response = generate(
        messages,
        model=ModelName.GPT_4O_MINI,
        provider=LLMProvider.OPENAI,
        temperature=0.3  # Lower temperature for more focused responses
    )
    
    print(f"\nConversation:")
    for msg in messages:
        print(f"  {msg.role.value}: {msg.content}")
    
    print(f"\nAssistant: {response.content}")
    print(f"\nLatency: {response.latency:.3f}s")


def example_dict_format():
    """Generate using dictionary format messages."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Dictionary Format Messages")
    print("="*80)
    
    # Can also use simple dictionaries
    messages = [
        {"role": "system", "content": "You are a concise technical writer."},
        {"role": "user", "content": "What is an API in one sentence?"}
    ]
    
    response = generate(
        messages,
        model=ModelName.GPT_4O_MINI,
        provider=LLMProvider.OPENAI,
        max_tokens=50
    )
    
    print(f"\nResponse: {response.content}")
    print(f"Finish Reason: {response.finish_reason}")


def example_different_providers():
    """Compare generation across different providers."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Multi-Provider Support")
    print("="*80)
    
    prompt = "Name three benefits of type hints in Python."
    
    print("\nComparing different providers:\n")
    
    # OpenAI
    try:
        print("\nUsing OpenAI (GPT-4o-mini):")
        response_openai = generate(prompt, model=ModelName.GPT_4O_MINI, provider=LLMProvider.OPENAI, temperature=0.5)
        print(f"  Provider: {response_openai.provider.value}")
        print(f"  {response_openai.content[:100]}...")
        print(f"  Cost: ${response_openai.cost:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Anthropic
    try:
        print("\nUsing Anthropic (Claude-3.5-Haiku):")
        response_anthropic = generate(prompt, model=ModelName.CLAUDE_35_HAIKU, provider=LLMProvider.ANTHROPIC, temperature=0.5)
        print(f"  Provider: {response_anthropic.provider.value}")
        print(f"  {response_anthropic.content[:100]}...")
        print(f"  Cost: ${response_anthropic.cost:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Google
    try:
        print("\nUsing Google (Gemini-1.5-Flash):")
        response_google = generate(prompt, model=ModelName.GEMINI_15_FLASH, provider=LLMProvider.GOOGLE, temperature=0.5)
        print(f"  Provider: {response_google.provider.value}")
        print(f"  {response_google.content[:100]}...")
        print(f"  Cost: ${response_google.cost:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Custom model with LOCAL provider
    try:
        print("\nUsing custom model with LOCAL provider:")
        response_custom = generate(
            prompt,
            model="my-custom-model",
            provider=LLMProvider.LOCAL,
            temperature=0.5
        )
        print(f"  Provider: {response_custom.provider.value}")
        print(f"  {response_custom.content[:100]}...")
    except Exception as e:
        print(f"  Error: {e}")


def example_generation_config():
    """Use detailed generation configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Advanced Configuration")
    print("="*80)
    
    from kerb.generation import GenerationConfig
    
    # Create configuration
    config = GenerationConfig(
        model="gpt-4o-mini",
        temperature=0.8,
        max_tokens=100,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.2
    )
    
    response = generate(
        "Generate a creative name for a Python testing framework.",
        config=config,
        model="gpt-4o-mini",
        provider=LLMProvider.OPENAI
    )
    
    print(f"\nConfiguration:")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max Tokens: {config.max_tokens}")
    print(f"  Top P: {config.top_p}")
    print(f"\nGenerated: {response.content}")


def main():
    """Run all basic generation examples."""
    print("\n" + "#"*80)
    print("# BASIC LLM GENERATION EXAMPLES")
    print("#"*80)
    
    try:
        example_simple_string()
    except Exception as e:
        print(f"\nExample 1 Error: {e}")
    
    try:
        example_message_list()
    except Exception as e:
        print(f"\nExample 2 Error: {e}")
    
    try:
        example_dict_format()
    except Exception as e:
        print(f"\nExample 3 Error: {e}")
    
    try:
        example_different_providers()
    except Exception as e:
        print(f"\nExample 4 Error: {e}")
    
    try:
        example_generation_config()
    except Exception as e:
        print(f"\nExample 5 Error: {e}")
    
    print("\n" + "#"*80)
    print("# Examples completed")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
