"""
Few-shot learning examples for LLM prompt engineering.
======================================================

This example demonstrates:
- Creating and managing few-shot examples
- Different selection strategies
- Example formatting for prompts
- Semantic and diversity-based selection
"""

from kerb.prompt import (
    create_example,
    ExampleSelector,
    select_examples,
    format_examples
)


def create_few_shot_examples():
    """Create few-shot examples for prompts."""
    print("=" * 80)
    print("CREATING FEW-SHOT EXAMPLES")
    print("=" * 80)
    
    # Create examples for a sentiment classification task
    ex1 = create_example(
        input_text="The product exceeded my expectations!",
        output_text="positive",
        metadata={"category": "product_review", "confidence": 0.95}
    )
    
    ex2 = create_example(
        input_text="Terrible customer service, very disappointed.",
        output_text="negative",
        metadata={"category": "service_review", "confidence": 0.90}
    )
    
    ex3 = create_example(
        input_text="It works as described, nothing special.",
        output_text="neutral",
        metadata={"category": "product_review", "confidence": 0.85}
    )
    
    print("\nCreated 3 few-shot examples for sentiment analysis:")
    print(f"1. Input: '{ex1.input}' -> Output: '{ex1.output}'")
    print(f"2. Input: '{ex2.input}' -> Output: '{ex2.output}'")
    print(f"3. Input: '{ex3.input}' -> Output: '{ex3.output}'")


def random_selection():
    """Randomly select examples for variety."""

# %%
# Setup and Imports
# -----------------
    print("\n" + "=" * 80)
    print("RANDOM SELECTION")
    print("=" * 80)
    
    # Create selector with multiple examples
    selector = ExampleSelector()
    
    examples_data = [
        ("The movie was fantastic!", "positive"),
        ("Worst purchase ever made.", "negative"),
        ("Average quality, decent price.", "neutral"),
        ("Absolutely love this product!", "positive"),
        ("Not worth the money.", "negative"),
    ]
    
    for inp, out in examples_data:
        selector.add(create_example(input_text=inp, output_text=out))
    
    # Select 3 random examples
    selected = selector.select(k=3, strategy="random")
    
    print(f"\nRandomly selected {len(selected)} examples:")
    for i, ex in enumerate(selected, 1):
        print(f"{i}. '{ex.input}' -> '{ex.output}'")
    
    print("\nUse case: Prevent overfitting to specific example patterns")



# %%
# First And Last Selection
# ------------------------

def first_and_last_selection():
    """Select first or last examples."""
    print("\n" + "=" * 80)
    print("FIRST AND LAST SELECTION")
    print("=" * 80)
    
    selector = ExampleSelector()
    
    # Add examples in order of complexity
    complexity_examples = [
        ("Good", "positive"),  # Simple
        ("Nice product", "positive"),  # Simple
        ("This is a well-made item", "positive"),  # Medium
        ("The quality of this product surpasses expectations", "positive"),  # Complex
        ("I am thoroughly impressed with the exceptional craftsmanship", "positive"),  # Complex
    ]
    
    for inp, out in complexity_examples:
        selector.add(create_example(input_text=inp, output_text=out))
    
    # Select first k (simplest examples)
    first_examples = selector.select(k=2, strategy="first")
    print("\nFirst 2 examples (simplest):")
    for ex in first_examples:
        print(f"  '{ex.input}'")
    
    # Select last k (most complex examples)
    last_examples = selector.select(k=2, strategy="last")
    print("\nLast 2 examples (most complex):")
    for ex in last_examples:
        print(f"  '{ex.input}'")
    
    print("\nUse case: Control example complexity based on task difficulty")


def diverse_selection():
    """Select diverse examples to cover different patterns."""
    print("\n" + "=" * 80)
    print("DIVERSE SELECTION")
    print("=" * 80)
    
    selector = ExampleSelector()
    
    # Add examples with different characteristics
    diverse_data = [
        ("Great!", "positive"),
        ("Good product", "positive"),
        ("Very satisfied", "positive"),
        ("Bad quality", "negative"),
        ("Disappointed", "negative"),
        ("Could be better", "negative"),
        ("It's okay", "neutral"),
        ("Average", "neutral"),
    ]
    
    for inp, out in diverse_data:
        selector.add(create_example(input_text=inp, output_text=out))
    
    # Select diverse subset
    selected = selector.select(k=4, strategy="diverse")
    
    print(f"\nDiverse selection of {len(selected)} examples:")
    for i, ex in enumerate(selected, 1):
        print(f"{i}. '{ex.input}' -> '{ex.output}'")
    
    print("\nUse case: Maximize coverage of different patterns and styles")



# %%
# Filtered Selection
# ------------------

def filtered_selection():
    """Filter examples before selection."""
    print("\n" + "=" * 80)
    print("FILTERED SELECTION")
    print("=" * 80)
    
    selector = ExampleSelector()
    
    # Add examples with metadata
    examples_with_metadata = [
        ("Love this!", "positive", {"domain": "product"}),
        ("Great service!", "positive", {"domain": "service"}),
        ("Terrible product", "negative", {"domain": "product"}),
        ("Poor support", "negative", {"domain": "service"}),
        ("Okay quality", "neutral", {"domain": "product"}),
    ]
    
    for inp, out, meta in examples_with_metadata:
        selector.add(create_example(input_text=inp, output_text=out, metadata=meta))
    
    # Filter to only product-related examples
    def product_filter(ex):
        return ex.metadata.get("domain") == "product"
    
    selected = selector.select(k=3, strategy="first", filter_fn=product_filter)
    
    print("\nSelected product-related examples only:")
    for ex in selected:
        print(f"  '{ex.input}' -> '{ex.output}' (domain: {ex.metadata['domain']})")
    
    print("\nUse case: Domain-specific or context-aware example selection")


def format_examples_for_prompt():
    """Format examples for inclusion in prompts."""
    print("\n" + "=" * 80)
    print("FORMATTING EXAMPLES FOR PROMPTS")
    print("=" * 80)
    
    # Create examples
    examples = [
        create_example(input_text="This is amazing!", output_text="positive"),
        create_example(input_text="Very disappointed.", output_text="negative"),
        create_example(input_text="It's acceptable.", output_text="neutral"),
    ]
    
    # Default formatting
    formatted = format_examples(examples)
    print("\nDefault format:")
    print(formatted)
    
    # Custom formatting template
    custom_template = "Example: {input} => Sentiment: {output}"
    formatted_custom = format_examples(examples, template=custom_template)
    print("\nCustom format:")
    print(formatted_custom)
    
    # Custom separator
    formatted_numbered = format_examples(
        examples,
        template="{input} -> {output}",
        separator="\n"
    )
    print("\nNumbered format:")
    for i, line in enumerate(formatted_numbered.split('\n'), 1):
        if line.strip():
            print(f"{i}. {line}")



# %%
# Complete Few Shot Prompt
# ------------------------

def complete_few_shot_prompt():
    """Create a complete few-shot prompt."""
    print("\n" + "=" * 80)
    print("COMPLETE FEW-SHOT PROMPT")
    print("=" * 80)
    
    # Create example selector
    selector = ExampleSelector()
    
    # Add training examples
    training_data = [
        ("Extract the name: John Smith lives in NYC", "John Smith"),
        ("Extract the name: Dr. Sarah Johnson is a scientist", "Dr. Sarah Johnson"),
        ("Extract the name: The CEO, Michael Brown, announced", "Michael Brown"),
        ("Extract the name: Alice Williams won the award", "Alice Williams"),
        ("Extract the name: Professor David Lee teaches at MIT", "Professor David Lee"),
    ]
    
    for inp, out in training_data:
        selector.add(create_example(input_text=inp, output_text=out))
    
    # Select 3 examples
    selected = selector.select(k=3, strategy="random")
    
    # Build complete prompt
    system_prompt = "You are a name extraction system. Extract person names from text."
    
    examples_text = format_examples(
        selected,
        template="Input: {input}\nOutput: {output}",
        separator="\n\n"
    )
    
    user_query = "Extract the name: Jennifer Martinez is the new director"
    
    complete_prompt = f"""{system_prompt}

Here are some examples:

{examples_text}

Now extract from:
Input: {user_query}
Output:"""
    
    print("\nComplete few-shot prompt:")
    print(complete_prompt)


def semantic_selection_fallback():
    """Demonstrate semantic selection with fallback."""
    print("\n" + "=" * 80)
    print("SEMANTIC SELECTION (with fallback)")
    print("=" * 80)
    
    selector = ExampleSelector()
    
    # Add code-related examples
    code_examples = [
        ("How do I sort a list?", "Use list.sort() or sorted(list)"),
        ("What is a lambda function?", "Anonymous function: lambda x: x + 1"),
        ("How to read a file?", "Use open('file.txt', 'r') as f"),
        ("What is list comprehension?", "[x*2 for x in range(10)]"),
    ]
    
    for inp, out in code_examples:
        selector.add(create_example(input_text=inp, output_text=out))
    
    query = "How do I iterate over a list?"
    
    print(f"\nQuery: '{query}'")
    print("\nAttempting semantic selection...")
    
    try:
        # Try semantic selection (requires embeddings)
        selected = selector.select(k=2, strategy="semantic", query=query)
        print("Selected semantically similar examples:")
        for ex in selected:
            print(f"  '{ex.input}'")
    except Exception as e:
        # Falls back to random if embeddings not available
        print(f"Semantic selection not available (embedding module required)")
        print("Falling back to random selection...")
        selected = selector.select(k=2, strategy="random")
        print("Selected examples:")
        for ex in selected:
            print(f"  '{ex.input}'")



# %%
# Production Example Bank
# -----------------------

def production_example_bank():
    """Demonstrate managing a production example bank."""
    print("\n" + "=" * 80)
    print("PRODUCTION EXAMPLE BANK")
    print("=" * 80)
    
    # Create example bank for code generation
    example_bank = ExampleSelector()
    
    # Add categorized examples
    examples_with_categories = [
        ("Create a class", "class MyClass:\\n    pass", "structure"),
        ("Define a function", "def my_func():\\n    pass", "structure"),
        ("Error handling", "try:\\n    code()\\nexcept Exception as e:\\n    handle(e)", "error"),
        ("List comprehension", "[x for x in items]", "idiom"),
        ("Dictionary comprehension", "{k: v for k, v in pairs}", "idiom"),
        ("Context manager", "with open('file') as f:\\n    data = f.read()", "idiom"),
    ]
    
    for desc, code, category in examples_with_categories:
        example_bank.add(create_example(
            input_text=desc,
            output_text=code,
            metadata={"category": category}
        ))
    
    print(f"\nExample bank contains {len(example_bank.examples)} examples")
    
    # Select examples by category
    def category_filter(category_name):
        return lambda ex: ex.metadata.get("category") == category_name
    
    idiom_examples = example_bank.select(
        k=3,
        strategy="first",
        filter_fn=category_filter("idiom")
    )
    
    print(f"\nSelected {len(idiom_examples)} idiom examples:")
    for ex in idiom_examples:
        print(f"  {ex.input}: {ex.output}")
    
    print("\nUse case: Maintain a curated example bank for consistent prompting")


def main():
    """Run all few-shot learning examples."""
    print("\n" + "=" * 80)
    print("FEW-SHOT LEARNING EXAMPLES FOR LLM DEVELOPERS")
    print("=" * 80)
    
    create_few_shot_examples()
    random_selection()
    first_and_last_selection()
    diverse_selection()
    filtered_selection()
    format_examples_for_prompt()
    complete_few_shot_prompt()
    semantic_selection_fallback()
    production_example_bank()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
