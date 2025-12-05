"""
Prompt optimization and compression for LLM applications.
=========================================================

This example demonstrates:
- Prompt compression to reduce token usage
- Whitespace optimization
- Prompt analysis and statistics
- Token-efficient prompt design
"""

from kerb.prompt import (
    compress_prompt,
    optimize_whitespace,
    analyze_prompt
)


def optimize_prompt_whitespace():
    """Remove excessive whitespace from prompts."""
    print("=" * 80)
    print("WHITESPACE OPTIMIZATION")
    print("=" * 80)
    
    # Prompt with excessive whitespace
    messy_prompt = """You are a helpful assistant.    Please analyze    this code.

# %%
# Setup and Imports
# -----------------


The code is:    def   process():
    return    True


Provide  feedback    on style and    efficiency."""
    
    print("\nOriginal prompt:")
    print(repr(messy_prompt))
    print(f"Length: {len(messy_prompt)} characters")
    
    # Optimize whitespace
    optimized = optimize_whitespace(messy_prompt)
    
    print("\nOptimized prompt:")
    print(repr(optimized))
    print(f"Length: {len(optimized)} characters")
    print(f"Saved: {len(messy_prompt) - len(optimized)} characters")


def compress_long_prompts():
    """Compress prompts to reduce token usage."""
    print("\n" + "=" * 80)
    print("PROMPT COMPRESSION")
    print("=" * 80)
    
    long_prompt = """You are an artificial intelligence assistant that specializes in 
    helping users with their programming questions and coding challenges. 
    
    You should always be polite and professional in your responses. 
    
    
    When analyzing code, please pay attention to the following aspects:
    - Code correctness and potential bugs
    - Performance and efficiency considerations  
    - Best practices and coding standards
    - Security implications if applicable
    
    
    Please provide detailed explanations for your suggestions."""
    
    print("\nOriginal prompt:")
    print(long_prompt)
    print(f"Length: {len(long_prompt)} characters")
    
    # Compress with whitespace optimization
    compressed = compress_prompt(long_prompt, strategies=["whitespace"])
    
    print("\nCompressed prompt:")
    print(compressed)
    print(f"Length: {len(compressed)} characters")
    print(f"Reduction: {((len(long_prompt) - len(compressed)) / len(long_prompt) * 100):.1f}%")



# %%
# Compress With Length Limit
# --------------------------

def compress_with_length_limit():
    """Compress and truncate prompts to meet token limits."""
    print("\n" + "=" * 80)
    print("COMPRESSION WITH LENGTH LIMIT")
    print("=" * 80)
    
    verbose_prompt = """You are a code review assistant. Your task is to analyze 
    the provided code snippet carefully and thoroughly. You should examine every 
    aspect of the code including syntax, logic, performance, security, and style.
    
    When reviewing code, consider the following important points:
    1. Check for syntax errors and typos
    2. Look for logical errors and edge cases
    3. Evaluate performance and optimization opportunities
    4. Identify potential security vulnerabilities
    5. Assess code style and adherence to best practices
    6. Suggest improvements and alternative approaches
    
    Please provide your feedback in a structured format with clear sections
    for each category of issues you identify. Be specific and constructive
    in your suggestions."""
    
    print(f"\nOriginal length: {len(verbose_prompt)} characters")
    
    # Compress to max 200 characters
    compressed = compress_prompt(verbose_prompt, max_length=200)
    
    print("\nCompressed to 200 characters:")
    print(compressed)
    print(f"Final length: {len(compressed)} characters")


def analyze_prompt_statistics():
    """Analyze prompt characteristics."""
    print("\n" + "=" * 80)
    print("PROMPT ANALYSIS")
    print("=" * 80)
    
    prompt = """You are a Python expert. Analyze this code:

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

Provide feedback on correctness and performance."""
    
    print("\nPrompt:")
    print(prompt)
    
    # Analyze the prompt
    analysis = analyze_prompt(prompt)
    
    # Check for code blocks
    has_code_blocks = any(line.strip().startswith('```') or line.strip().startswith('def ') for line in prompt.split('\n'))
    
    print("\nAnalysis:")
    print(f"  Character count: {analysis['length']}")
    print(f"  Word count: {analysis['words']}")
    print(f"  Line count: {analysis['lines']}")
    print(f"  Estimated tokens: {analysis['tokens_approx']}")
    print(f"  Has code blocks: {has_code_blocks}")
    print(f"  Average word length: {analysis['avg_word_length']:.1f}")



# %%
# Token Efficient Design
# ----------------------

def token_efficient_design():
    """Design token-efficient prompts."""
    print("\n" + "=" * 80)
    print("TOKEN-EFFICIENT PROMPT DESIGN")
    print("=" * 80)
    
    # Verbose version
    verbose = """I would like you to please analyze the following Python function
    and provide me with your thoughts and feedback regarding its implementation.
    Please let me know if there are any issues or improvements that could be made.
    
    Here is the function:

# %%
# Calculate Total
# ---------------

    def calculate_total(items):
        total = 0
        for item in items:
            total = total + item
        return total
    
    I would appreciate your detailed analysis and suggestions."""
    
    # Token-efficient version
    efficient = """Analyze this Python function:


# %%
# Calculate Total
# ---------------

def calculate_total(items):
    total = 0
    for item in items:
        total = total + item
    return total

Suggest improvements."""
    
    print("\nVerbose version:")
    verbose_analysis = analyze_prompt(verbose)
    print(f"  Characters: {verbose_analysis['length']}")
    print(f"  Estimated tokens: {verbose_analysis['tokens_approx']}")
    
    print("\nEfficient version:")
    efficient_analysis = analyze_prompt(efficient)
    print(f"  Characters: {efficient_analysis['length']}")
    print(f"  Estimated tokens: {efficient_analysis['tokens_approx']}")
    
    token_savings = verbose_analysis['tokens_approx'] - efficient_analysis['tokens_approx']
    print(f"\nToken savings: {token_savings} (~{token_savings/verbose_analysis['tokens_approx']*100:.1f}%)")


def optimize_system_prompts():
    """Optimize system prompts for production."""
    print("\n" + "=" * 80)
    print("SYSTEM PROMPT OPTIMIZATION")
    print("=" * 80)
    
    # Original system prompt
    original = """You are a helpful, harmless, and honest AI assistant. You have been 
    designed to assist users with a wide variety of tasks including answering questions, 
    providing explanations, helping with code, and offering suggestions. You should always 
    strive to be accurate and informative in your responses. When you are uncertain about 
    something, you should clearly communicate your uncertainty rather than providing 
    potentially incorrect information. You should also be respectful and professional 
    in all interactions."""
    
    print("Original system prompt:")
    original_analysis = analyze_prompt(original)
    print(f"  Tokens: {original_analysis['tokens_approx']}")
    
    # Optimized version
    optimized = compress_prompt(original, max_length=150)
    
    print("\nOptimized system prompt:")
    print(optimized)
    optimized_analysis = analyze_prompt(optimized)
    print(f"  Tokens: {optimized_analysis['tokens_approx']}")
    print(f"  Savings: {original_analysis['tokens_approx'] - optimized_analysis['tokens_approx']} tokens")



# %%
# Batch Optimization
# ------------------

def batch_optimization():
    """Optimize multiple prompts in batch."""
    print("\n" + "=" * 80)
    print("BATCH PROMPT OPTIMIZATION")
    print("=" * 80)
    
    prompts = {
        "code_review": """Please review this code carefully and provide detailed 
        feedback on any issues you find. Look for bugs, performance problems, 
        and style issues. Thank you!""",
        
        "summarize": """I need you to create a summary of the following text. 
        Please make it concise but ensure all key points are included. 
        The summary should be easy to understand.""",
        
        "translate": """Please translate the following text from English to Spanish. 
        Make sure the translation is accurate and natural sounding. 
        Preserve the tone and meaning of the original."""
    }
    
    print("\nOptimizing prompts:")
    
    for name, prompt in prompts.items():
        original_len = len(prompt)
        optimized = compress_prompt(prompt)
        optimized_len = len(optimized)
        
        print(f"\n  {name}:")
        print(f"    Original: {original_len} chars")
        print(f"    Optimized: {optimized_len} chars")
        print(f"    Saved: {original_len - optimized_len} chars ({(original_len-optimized_len)/original_len*100:.1f}%)")


def maintain_clarity_while_compressing():
    """Balance compression with clarity."""
    print("\n" + "=" * 80)
    print("BALANCING COMPRESSION AND CLARITY")
    print("=" * 80)
    
    detailed_prompt = """You are a senior software engineer conducting a code review.
    Please analyze the following code with attention to:
    
    1. Correctness: Does the code work as intended?
    2. Efficiency: Can performance be improved?
    3. Readability: Is the code easy to understand?
    4. Maintainability: Will this be easy to maintain?
    
    Code to review:
    {{code}}
    
    Provide specific, actionable feedback."""
    
    print("Original detailed prompt:")
    print(detailed_prompt)
    detailed_analysis = analyze_prompt(detailed_prompt)
    print(f"Tokens: {detailed_analysis['tokens_approx']}")
    
    # Light compression - preserve structure
    light_compressed = optimize_whitespace(detailed_prompt)
    
    print("\nLightly compressed (whitespace only):")
    print(light_compressed)
    light_analysis = analyze_prompt(light_compressed)
    print(f"Tokens: {light_analysis['tokens_approx']}")
    
    # Aggressive compression
    aggressive_compressed = compress_prompt(detailed_prompt, max_length=100)
    
    print("\nAggressively compressed:")
    print(aggressive_compressed)
    aggressive_analysis = analyze_prompt(aggressive_compressed)
    print(f"Tokens: {aggressive_analysis['tokens_approx']}")
    
    print("\nRecommendation: Use light compression to save tokens while preserving")
    print("prompt clarity and effectiveness.")



# %%
# Main
# ----

def main():
    """Run all optimization examples."""
    print("\n" + "=" * 80)
    print("PROMPT OPTIMIZATION EXAMPLES FOR LLM DEVELOPERS")
    print("=" * 80)
    
    optimize_prompt_whitespace()
    compress_long_prompts()
    compress_with_length_limit()
    analyze_prompt_statistics()
    token_efficient_design()
    optimize_system_prompts()
    batch_optimization()
    maintain_clarity_while_compressing()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
