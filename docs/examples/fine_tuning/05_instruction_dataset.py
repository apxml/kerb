"""
Instruction Dataset Example
===========================

This example demonstrates how to create instruction-following datasets for fine-tuning.

Main concepts:
- Creating instruction-response pairs
- Generating and managing system prompts
- Standardizing system prompts across dataset
- Extracting system prompts from existing data
- Creating few-shot instruction examples
- Multi-turn instruction conversations

Use case: Preparing datasets for instruction-tuned models that follow
specific task patterns, such as coding assistants, summarizers, or
specialized domain experts.
"""

from kerb.fine_tuning import (
    prepare_dataset,
    TrainingExample,
    TrainingDataset,
    DatasetFormat,
    FineTuningProvider,
)
from kerb.fine_tuning.prompts import (
    generate_system_prompt,
    extract_system_prompts,
    standardize_system_prompts,
)


def create_code_generation_instructions():
    """Create instruction dataset for code generation."""
    examples = []
    
    system_prompt = "You are an expert Python programmer. Generate clean, efficient, and well-documented code based on user instructions."
    
    # Simple function generation
    examples.append(TrainingExample(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write a function to calculate the factorial of a number"},
            {"role": "assistant", "content": """def factorial(n):

# %%
# Setup and Imports
# -----------------
    '''Calculate factorial of n.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Factorial of n
    '''
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""}
        ],
        metadata={"task": "code_generation", "complexity": "simple"}
    ))
    
    # List comprehension
    examples.append(TrainingExample(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Create a function that filters even numbers from a list"},
            {"role": "assistant", "content": """def filter_even(numbers):
    '''Filter even numbers from a list.
    
    Args:
        numbers: List of integers
        
    Returns:
        List containing only even numbers
    '''
    return [n for n in numbers if n % 2 == 0]"""}
        ],
        metadata={"task": "code_generation", "complexity": "simple"}
    ))
    
    # Error handling
    examples.append(TrainingExample(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write a function to safely divide two numbers"},
            {"role": "assistant", "content": """def safe_divide(a, b):
    '''Safely divide two numbers with error handling.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        Result of division or None if error
    '''
    try:
        return a / b
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
        return None
    except TypeError:
        print("Error: Invalid input types")
        return None"""}
        ],
        metadata={"task": "code_generation", "complexity": "intermediate"}
    ))
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CHAT,
        metadata={"task_type": "code_generation"}
    )


def create_summarization_instructions():
    """Create instruction dataset for text summarization."""
    examples = []
    
    system_prompt = "You are an expert at summarizing text. Provide concise, accurate summaries that capture key information."
    
    examples.append(TrainingExample(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": """Summarize this text in one sentence:
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future."""},
            {"role": "assistant", "content": "Machine learning is an AI subset that enables systems to automatically learn and improve from data without explicit programming."}
        ],
        metadata={"task": "summarization", "style": "concise"}
    ))
    
    examples.append(TrainingExample(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": """Provide a bullet-point summary:
Python is a high-level, interpreted programming language known for its clear syntax and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python has a large standard library and an extensive ecosystem of third-party packages."""},
            {"role": "assistant", "content": """- High-level, interpreted programming language
- Clear syntax and highly readable
- Supports procedural, object-oriented, and functional paradigms
- Large standard library and extensive package ecosystem"""}
        ],
        metadata={"task": "summarization", "style": "bullet_points"}
    ))
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CHAT,
        metadata={"task_type": "summarization"}
    )



# %%
# Create Multi Turn Instructions
# ------------------------------

def create_multi_turn_instructions():
    """Create multi-turn instruction conversations."""
    examples = []
    
    system_prompt = "You are a helpful coding tutor. Guide users through programming concepts step by step."
    
    # Multi-turn conversation
    examples.append(TrainingExample(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "I want to learn about Python decorators"},
            {"role": "assistant", "content": "Great! Decorators are a powerful feature in Python. Let's start with the basics. Do you understand functions as first-class objects in Python?"},
            {"role": "user", "content": "Not really, can you explain?"},
            {"role": "assistant", "content": "Of course! In Python, functions are first-class objects, meaning they can be passed as arguments, returned from other functions, and assigned to variables. For example: `def greet(): return 'Hello'` and then `my_func = greet` assigns the function to a variable."},
            {"role": "user", "content": "I see. So how does that relate to decorators?"},
            {"role": "assistant", "content": "Perfect question! A decorator is a function that takes another function and extends its behavior without modifying it. Here's a simple example:\n\n```python\ndef my_decorator(func):\n    def wrapper():\n        print('Before function')\n        func()\n        print('After function')\n    return wrapper\n\n@my_decorator\ndef say_hello():\n    print('Hello!')\n```\n\nThe @my_decorator syntax is shorthand for `say_hello = my_decorator(say_hello)`."}
        ],
        metadata={"task": "tutoring", "topic": "decorators", "turns": 3}
    ))
    
    return TrainingDataset(
        examples=examples,
        format=DatasetFormat.CHAT,
        metadata={"task_type": "multi_turn_instruction"}
    )


def main():
    """Run instruction dataset example."""
    
    print("="*80)
    print("INSTRUCTION DATASET EXAMPLE")
    print("="*80)
    
    # Step 1: Create code generation instructions
    print("\nStep 1: Creating code generation instruction dataset")
    code_dataset = create_code_generation_instructions()
    print(f"Code generation examples: {len(code_dataset)}")
    
    # Display example
    print("\nExample code generation instruction:")
    print("-"*80)
    example = code_dataset.examples[0]
    for msg in example.messages:
        role = msg['role']
        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
        print(f"{role}: {content}")
    
    # Step 2: Create summarization instructions
    print("\n" + "="*80)
    print("SUMMARIZATION INSTRUCTIONS")
    print("="*80)
    
    summary_dataset = create_summarization_instructions()
    print(f"\nSummarization examples: {len(summary_dataset)}")
    
    # Step 3: Create multi-turn instructions
    print("\n" + "="*80)
    print("MULTI-TURN INSTRUCTIONS")
    print("="*80)
    
    multiturn_dataset = create_multi_turn_instructions()
    print(f"\nMulti-turn examples: {len(multiturn_dataset)}")
    
    example = multiturn_dataset.examples[0]
    print(f"Number of messages in conversation: {len(example.messages)}")
    print("\nConversation flow:")
    for i, msg in enumerate(example.messages):
        role = msg['role']
        preview = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
        print(f"  {i+1}. {role}: {preview}")
    
    # Step 4: Generate system prompts
    print("\n" + "="*80)
    print("GENERATING SYSTEM PROMPTS")
    print("="*80)
    
    # Generate for different tasks
    tasks = [
        ("data analysis", ["Analyze customer sentiment", "Find trends in sales data"]),
        ("creative writing", ["Write a short story", "Create a poem"]),
        ("technical support", ["Troubleshoot network issues", "Debug software errors"]),
    ]
    
    print("\nGenerated system prompts:")
    for task, examples_list in tasks:
        prompt = generate_system_prompt(task, examples_list)
        print(f"\n{task.upper()}:")
        print(f"  {prompt[:150]}...")
    
    # Step 5: Extract system prompts from datasets
    print("\n" + "="*80)
    print("EXTRACTING SYSTEM PROMPTS")
    print("="*80)
    
    # Combine datasets
    combined_examples = (code_dataset.examples + 
                        summary_dataset.examples + 
                        multiturn_dataset.examples)
    combined = TrainingDataset(
        examples=combined_examples,
        format=DatasetFormat.CHAT
    )
    
    system_prompts = extract_system_prompts(combined)
    print(f"\nFound {len(system_prompts)} unique system prompts:")
    for i, prompt in enumerate(system_prompts, 1):
        preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        print(f"  {i}. {preview}")
    
    # Step 6: Standardize system prompts
    print("\n" + "="*80)
    print("STANDARDIZING SYSTEM PROMPTS")
    print("="*80)
    
    standard_prompt = "You are a helpful AI assistant specializing in programming and technical tasks. Provide clear, accurate, and well-structured responses."
    
    print(f"\nStandard prompt: {standard_prompt[:100]}...")
    
    standardized = standardize_system_prompts(combined, standard_prompt)
    print(f"\nStandardized {len(standardized)} examples")
    
    # Verify all have same system prompt
    new_prompts = extract_system_prompts(standardized)
    print(f"Unique system prompts after standardization: {len(new_prompts)}")
    
    # Step 7: Create few-shot instruction examples
    print("\n" + "="*80)
    print("FEW-SHOT INSTRUCTION EXAMPLES")
    print("="*80)
    
    few_shot_system = """You are a sentiment analysis expert. Classify text as positive, negative, or neutral.

Examples:
- "I love this product!" -> positive
- "This is terrible" -> negative
- "It's okay" -> neutral

Provide only the classification."""
    
    few_shot_examples = [
        TrainingExample(
            messages=[
                {"role": "system", "content": few_shot_system},
                {"role": "user", "content": "The service was amazing and the staff was friendly!"},
                {"role": "assistant", "content": "positive"}
            ]
        ),
        TrainingExample(
            messages=[
                {"role": "system", "content": few_shot_system},
                {"role": "user", "content": "Worst experience ever, would not recommend."},
                {"role": "assistant", "content": "negative"}
            ]
        ),
    ]
    
    few_shot_dataset = TrainingDataset(
        examples=few_shot_examples,
        format=DatasetFormat.CHAT,
        metadata={"type": "few_shot_classification"}
    )
    
    print(f"\nFew-shot examples: {len(few_shot_dataset)}")
    print("\nFew-shot system prompt includes examples in the prompt itself:")
    print(few_shot_system[:200] + "...")
    
    # Step 8: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nCode generation instructions: {len(code_dataset)} examples")
    print(f"Summarization instructions: {len(summary_dataset)} examples")
    print(f"Multi-turn instructions: {len(multiturn_dataset)} examples")
    print(f"Few-shot examples: {len(few_shot_dataset)} examples")
    print(f"Total instruction examples: {len(code_dataset) + len(summary_dataset) + len(multiturn_dataset) + len(few_shot_dataset)}")
    print("\nInstruction dataset types covered:")
    print("  - Single-turn instructions")
    print("  - Multi-turn conversations")
    print("  - Few-shot learning")
    print("  - Task-specific system prompts")
    print("  - Standardized prompts")


if __name__ == "__main__":
    main()
