"""
Streaming Generation Example
============================

This example demonstrates real-time token streaming for LLM responses.

Main concepts:
- Using generate_stream() for token-by-token output
- Processing StreamChunk objects
- Real-time display of generation
- Callback functions for chunk processing
- Measuring streaming performance
"""

import time
from kerb.generation import generate_stream, ModelName
from kerb.core import Message
from kerb.core.types import MessageRole


def example_basic_streaming():
    """Basic streaming example with simple output."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Streaming")
    print("="*80)
    
    prompt = "Explain the concept of async/await in Python in 2 sentences."
    
    print(f"\nPrompt: {prompt}")
    print("\nStreaming Response:")
    print("-" * 80)
    
    start_time = time.time()
    full_content = ""
    chunk_count = 0
    
    for chunk in generate_stream(prompt, model=ModelName.GPT_4O_MINI):
        print(chunk.content, end="", flush=True)
        full_content += chunk.content
        chunk_count += 1
    
    elapsed = time.time() - start_time
    
    print("\n" + "-" * 80)
    print(f"\nStats:")
    print(f"  Chunks received: {chunk_count}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Characters: {len(full_content)}")


def example_streaming_with_callback():
    """Streaming with callback function for processing each chunk."""

# %%
# Setup and Imports
# -----------------
    print("\n" + "="*80)
    print("EXAMPLE 2: Streaming with Callback")
    print("="*80)
    
    chunks_received = []
    

# %%
# Chunk Callback
# --------------

    def chunk_callback(chunk):
        """Process each chunk as it arrives."""
        chunks_received.append({
            "content": chunk.content,
            "timestamp": time.time(),
            "finish_reason": chunk.finish_reason
        })
    
    prompt = "List 5 Python design patterns."
    print(f"\nPrompt: {prompt}")
    print("\nStreaming with callback processing...")
    print("-" * 80)
    
    start_time = time.time()
    
    for chunk in generate_stream(
        prompt, 
        model=ModelName.GPT_4O_MINI,
        callback=chunk_callback
    ):
        print(chunk.content, end="", flush=True)
    
    print("\n" + "-" * 80)
    print(f"\nCallback Analysis:")
    print(f"  Total chunks: {len(chunks_received)}")
    
    if len(chunks_received) > 1:
        # Calculate time between first and last chunk
        time_span = chunks_received[-1]["timestamp"] - chunks_received[0]["timestamp"]
        print(f"  Time span: {time_span:.3f}s")
        print(f"  Avg chunk interval: {time_span/len(chunks_received):.4f}s")


def example_streaming_conversation():
    """Stream a multi-turn conversation."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Streaming Conversation")
    print("="*80)
    
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a concise Python tutor."),
        Message(role=MessageRole.USER, content="What is a decorator?"),
    ]
    
    print("\nConversation:")
    for msg in messages:
        print(f"  {msg.role.value}: {msg.content}")
    
    print("\nAssistant: ", end="", flush=True)
    
    for chunk in generate_stream(messages, model=ModelName.GPT_4O_MINI, temperature=0.7):
        print(chunk.content, end="", flush=True)
    
    print("\n")



# %%
# Example Streaming Comparison
# ----------------------------

def example_streaming_comparison():
    """Compare streaming vs non-streaming generation."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Streaming vs Non-Streaming Comparison")
    print("="*80)
    
    from kerb.generation import generate
    
    prompt = "Explain recursion in Python with a simple example."
    
    # Non-streaming
    print("\n1. Non-Streaming (wait for full response):")
    start = time.time()
    response = generate(prompt, model=ModelName.GPT_4O_MINI)
    elapsed_full = time.time() - start
    
    print(f"   Time to first token: {elapsed_full:.3f}s (entire response)")
    print(f"   Total time: {elapsed_full:.3f}s")
    print(f"   Response length: {len(response.content)} chars")
    
    # Streaming
    print("\n2. Streaming (see tokens as they arrive):")
    start = time.time()
    first_chunk_time = None
    total_chars = 0
    
    for i, chunk in enumerate(generate_stream(prompt, model=ModelName.GPT_4O_MINI)):
        if i == 0:
            first_chunk_time = time.time() - start
        total_chars += len(chunk.content)
    
    elapsed_stream = time.time() - start
    
    print(f"   Time to first token: {first_chunk_time:.3f}s")
    print(f"   Total time: {elapsed_stream:.3f}s")
    print(f"   Response length: {total_chars} chars")
    print(f"\n   Time saved to first token: {elapsed_full - first_chunk_time:.3f}s")


def example_streaming_with_progress():
    """Display progress indicators during streaming."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Streaming with Progress Display")
    print("="*80)
    
    prompt = "Write a docstring for a function that calculates Fibonacci numbers."
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating", end="", flush=True)
    
    content = ""
    dots = 0
    
    for chunk in generate_stream(prompt, model=ModelName.GPT_4O_MINI):
        content += chunk.content
        
        # Show progress dots every 10 characters
        while len(content) > (dots + 1) * 10:
            print(".", end="", flush=True)
            dots += 1
    
    print(" Done!")
    print("-" * 80)
    print(content)
    print("-" * 80)



# %%
# Example Streaming Custom Models
# -------------------------------

def example_streaming_custom_models():
    """Stream with different model configurations."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Streaming Different Models")
    print("="*80)
    
    prompt = "Define 'technical debt' in one sentence."
    
    models_to_test = [
        (ModelName.GPT_4O_MINI, "GPT-4o-mini"),
        (ModelName.GPT_35_TURBO, "GPT-3.5-Turbo"),
    ]
    
    for model, name in models_to_test:
        try:
            print(f"\n{name}:")
            print("  ", end="", flush=True)
            
            start = time.time()
            for chunk in generate_stream(prompt, model=model, temperature=0.5):
                print(chunk.content, end="", flush=True)
            elapsed = time.time() - start
            
            print(f"\n  (Generated in {elapsed:.3f}s)")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Run all streaming generation examples."""
    print("\n" + "#"*80)
    print("# STREAMING GENERATION EXAMPLES")
    print("#"*80)
    
    try:
        example_basic_streaming()
    except Exception as e:
        print(f"\nExample 1 Error: {e}")
    
    try:
        example_streaming_with_callback()
    except Exception as e:
        print(f"\nExample 2 Error: {e}")
    
    try:
        example_streaming_conversation()
    except Exception as e:
        print(f"\nExample 3 Error: {e}")
    
    try:
        example_streaming_comparison()
    except Exception as e:
        print(f"\nExample 4 Error: {e}")
    
    try:
        example_streaming_with_progress()
    except Exception as e:
        print(f"\nExample 5 Error: {e}")
    
    try:
        example_streaming_custom_models()
    except Exception as e:
        print(f"\nExample 6 Error: {e}")
    
    print("\n" + "#"*80)
    print("# Examples completed")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
