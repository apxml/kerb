"""
Universal Generator Example
===========================

This example demonstrates the universal generator with easy model switching.

Main concepts:
- Specifying provider parameter for all generation calls
- Easy model switching using Generator class
- Type-safe model selection with ModelName enum
"""

from kerb.generation import (
    generate,
    Generator,
    ModelName,
    LLMProvider,
    GenerationConfig
)
from kerb.core import Message
from kerb.core.types import MessageRole


def example_provider_specification():
    """Demonstrate provider specification."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Provider Specification")
    print("="*80)
    
    prompt = "What is machine learning?"
    
    print("\nGenerating with different providers:\n")
    
    # OpenAI
    print("1. OpenAI GPT-4o-mini:")
    try:
        response = generate(
            prompt,
            model="gpt-4o-mini",
            provider=LLMProvider.OPENAI
        )
        print(f"   ✓ Routed to: {response.provider.value}")
        print(f"   Response: {response.content[:80]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Anthropic
    print("\n2. Anthropic Claude:")
    try:
        response = generate(
            prompt,
            model="claude-3-5-haiku-20241022",
            provider=LLMProvider.ANTHROPIC
        )
        print(f"   ✓ Routed to: {response.provider.value}")
        print(f"   Response: {response.content[:80]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Google
    print("\n3. Google Gemini:")
    try:
        response = generate(
            prompt,
            model="gemini-1.5-flash",
            provider=LLMProvider.GOOGLE
        )
        print(f"   ✓ Routed to: {response.provider.value}")
        print(f"   Response: {response.content[:80]}...")
    except Exception as e:
        print(f"   Error: {e}")


def example_generator_class():
    """Demonstrate the Generator class."""

# %%
# Setup and Imports
# -----------------
    print("\n" + "="*80)
    print("EXAMPLE 2: Generator Class")
    print("="*80)
    
    # Create a generator
    gen = Generator(
        model="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"\nCreated generator:")
    print(f"  Model: {gen.model}")
    print(f"  Provider: {gen.provider.value}")
    print(f"  Temperature: {gen.default_config['temperature']}")
    
    # Use it multiple times
    prompts = [
        "What is Python?",
        "What is JavaScript?",
    ]
    
    print("\nGenerating responses:")
    for prompt in prompts:
        try:
            response = gen.generate(prompt)
            print(f"\n  Q: {prompt}")
            print(f"  A: {response.content[:60]}...")
        except Exception as e:
            print(f"\n  Q: {prompt}")
            print(f"  Error: {e}")



# %%
# Example Easy Model Switching
# ----------------------------

def example_easy_model_switching():
    """Demonstrate easy model and provider switching."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Model and Provider Switching")
    print("="*80)
    
    prompt = "Explain async programming in one sentence."
    
    # Create generators for different providers
    generators = {
        "GPT-4o-mini (OpenAI)": Generator(
            model="gpt-4o-mini",
            provider=LLMProvider.OPENAI,
            temperature=0.5
        ),
        "Claude-3.5-Haiku (Anthropic)": Generator(
            model="claude-3-5-haiku-20241022",
            provider=LLMProvider.ANTHROPIC,
            temperature=0.5
        ),
        "Gemini-1.5-Flash (Google)": Generator(
            model="gemini-1.5-flash",
            provider=LLMProvider.GOOGLE,
            temperature=0.5
        ),
    }
    
    print(f"\nPrompt: {prompt}\n")
    print("Comparing responses:")
    print("-" * 80)
    
    for name, gen in generators.items():
        try:
            response = gen.generate(prompt)
            print(f"\n{name}:")
            print(f"  {response.content[:100]}...")
            print(f"  (Tokens: {response.usage.total_tokens}, Cost: ${response.cost:.6f})")
        except Exception as e:
            print(f"\n{name}:")
            print(f"  Error: {e}")


def example_custom_models():
    """Demonstrate using custom model names."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Model Names")
    print("="*80)
    
    print("\nUsing custom model names with different providers:")
    
    # Company internal model using OpenAI API
    print("\n1. Internal model using OpenAI API:")
    gen1 = Generator(
        model="acme-corp-gpt-custom-v2",
        provider=LLMProvider.OPENAI,  # Uses OpenAI API
        temperature=0.7
    )
    print(f"   Model: {gen1.model}")
    print(f"   Provider: {gen1.provider.value}")
    print("   ✓ Will call OpenAI API")
    
    # Fine-tuned model using Anthropic API
    print("\n2. Fine-tuned model using Anthropic API:")
    gen2 = Generator(
        model="my-finetuned-claude-medical",
        provider=LLMProvider.ANTHROPIC,  # Uses Anthropic API
        temperature=0.5
    )
    print(f"   Model: {gen2.model}")
    print(f"   Provider: {gen2.provider.value}")
    print("   ✓ Will call Anthropic API")
    
    # Local model
    print("\n3. Local model:")
    gen3 = Generator(
        model="llama-3-70b-local",
        provider=LLMProvider.LOCAL,  # Local/custom endpoint
        temperature=0.8
    )
    print(f"   Model: {gen3.model}")
    print(f"   Provider: {gen3.provider.value}")
    print("   ✓ Will call local/custom provider")



# %%
# Example With Modelname Enum
# ---------------------------

def example_with_modelname_enum():
    """Demonstrate using ModelName enum."""
    print("\n" + "="*80)
    print("EXAMPLE 5: ModelName Enum")
    print("="*80)
    
    print("\nModelName enum provides type-safe model names:\n")
    
    try:
        response = generate(
            "What is REST?",
            model=ModelName.GPT_4O_MINI,
            provider=LLMProvider.OPENAI
        )
        print(f"  Model: {response.model}")
        print(f"  Provider: {response.provider.value}")
        print(f"  Response: {response.content[:60]}...")
    except Exception as e:
        print(f"  Error: {e}")


def example_multi_turn_conversation():
    """Demonstrate multi-turn conversation."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Multi-Turn Conversation")
    print("="*80)
    
    # Create generator
    gen = Generator(
        model="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"\nConversation with {gen.model} via {gen.provider.value}...")
    
    # Build conversation
    conversation = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful Python tutor."),
        Message(role=MessageRole.USER, content="What is a decorator?"),
    ]
    
    try:
        # First exchange
        response1 = gen.generate(conversation)
        print(f"\nUser: What is a decorator?")
        print(f"Assistant: {response1.content[:80]}...")
        
        # Continue conversation
        conversation.append(Message(role=MessageRole.ASSISTANT, content=response1.content))
        conversation.append(Message(role=MessageRole.USER, content="Show me an example"))
        
        response2 = gen.generate(conversation)
        print(f"\nUser: Show me an example")
        print(f"Assistant: {response2.content[:80]}...")
    except Exception as e:
        print(f"\nError: {e}")



# %%
# Main
# ----

def main():
    """Run all examples."""
    print("\n" + "#"*80)
    print("# UNIVERSAL GENERATOR EXAMPLES")
    print("#"*80)
    
    try:
        example_provider_specification()
    except Exception as e:
        print(f"\nExample 1 Error: {e}")
    
    try:
        example_generator_class()
    except Exception as e:
        print(f"\nExample 2 Error: {e}")
    
    try:
        example_easy_model_switching()
    except Exception as e:
        print(f"\nExample 3 Error: {e}")
    
    try:
        example_custom_models()
    except Exception as e:
        print(f"\nExample 4 Error: {e}")
    
    try:
        example_with_modelname_enum()
    except Exception as e:
        print(f"\nExample 5 Error: {e}")
    
    try:
        example_multi_turn_conversation()
    except Exception as e:
        print(f"\nExample 6 Error: {e}")
    
    print("\n" + "#"*80)
    print("# BEST PRACTICES")
    print("#"*80)
    print("\n1. ✓ Specify provider parameter to control which API to call")
    print("2. ✓ Use ModelName enum for type-safe model names")
    print("3. ✓ Use Generator class for stateful generation")
    print("4. ✓ Easy to switch between models and providers")
    print("\n" + "#"*80)
    print("# Examples completed")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
