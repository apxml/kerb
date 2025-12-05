"""
Prompt Optimization Example
===========================

This example demonstrates how to optimize prompts for token efficiency,
which is essential for LLM applications to reduce costs, improve performance,
and maximize the effective use of context windows.

Main concepts:
- Analyzing prompt token usage
- Identifying optimization opportunities
- Comparing verbose vs. concise prompts
- Template optimization strategies
"""

from kerb.tokenizer import count_tokens, Tokenizer
from kerb.tokenizer.utils import optimize_token_usage


def main():
    """Run prompt optimization examples."""
    
    print("="*80)
    print("PROMPT OPTIMIZATION EXAMPLE")
    print("="*80)
    
    # Example 1: Verbose vs. Concise prompts
    print("\n" + "-"*80)
    print("EXAMPLE 1: Verbose vs. Concise Prompt Comparison")
    print("-"*80)
    
    verbose_prompt = """

# %%
# Setup and Imports
# -----------------
    I would like you to please analyze the following text very carefully and 
    thoroughly, and then provide me with a comprehensive and detailed summary 
    that captures all of the main points and key ideas. Please make sure to 
    include all important information and don't leave anything out. Thank you 
    very much for your help with this task.
    
    Text to analyze: [User's text here]
    """
    
    concise_prompt = """
    Summarize this text, capturing all main points:
    
    [User's text here]
    """
    
    print("Verbose prompt:")
    print(verbose_prompt)
    verbose_tokens = count_tokens(verbose_prompt, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {verbose_tokens}\n")
    
    print("Concise prompt:")
    print(concise_prompt)
    concise_tokens = count_tokens(concise_prompt, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {concise_tokens}\n")
    
    savings = verbose_tokens - concise_tokens
    savings_percent = (savings / verbose_tokens) * 100
    
    print(f"Optimization results:")
    print(f"  Tokens saved: {savings} ({savings_percent:.1f}%)")
    print(f"  For 1000 requests: {savings * 1000:,} tokens saved")
    
    # Example 2: Removing redundant instructions
    print("\n" + "-"*80)
    print("EXAMPLE 2: Eliminating Redundancy")
    print("-"*80)
    
    redundant_prompt = """
    You are a helpful assistant. You should help the user. Be helpful and assist.
    
    The user will ask a question. Answer the user's question. Provide an answer.
    
    Make sure your answer is clear. The answer should be clear and understandable.
    Be concise but also be thorough. Don't be too brief but don't be too long.
    """
    
    optimized_prompt = """
    You are a helpful assistant. Provide clear, concise answers to user questions.
    """
    
    print("Redundant prompt:")
    print(redundant_prompt)
    redundant_tokens = count_tokens(redundant_prompt, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {redundant_tokens}\n")
    
    print("Optimized prompt:")
    print(optimized_prompt)
    optimized_tokens = count_tokens(optimized_prompt, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {optimized_tokens}\n")
    
    print(f"Reduction: {redundant_tokens - optimized_tokens} tokens ({(redundant_tokens - optimized_tokens) / redundant_tokens * 100:.1f}%)")
    
    # Example 3: Efficient instruction formatting
    print("\n" + "-"*80)
    print("EXAMPLE 3: Instruction Formatting Optimization")
    print("-"*80)
    
    wordy_instructions = """
    Please follow these instructions carefully:
    
    1. First, you should read the input text
    2. Then, you need to identify the main topic
    3. After that, you should extract key points
    4. Next, you must organize the information
    5. Finally, you should write a summary
    
    Please make sure to follow all of these steps in order.
    """
    
    concise_instructions = """
    Instructions:
    1. Read input
    2. Identify topic
    3. Extract key points
    4. Organize information
    5. Write summary
    """
    
    print("Wordy instructions:")
    wordy_tokens = count_tokens(wordy_instructions, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {wordy_tokens}\n")
    
    print("Concise instructions:")
    concise_inst_tokens = count_tokens(concise_instructions, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {concise_inst_tokens}\n")
    
    print(f"Tokens saved: {wordy_tokens - concise_inst_tokens} ({(wordy_tokens - concise_inst_tokens) / wordy_tokens * 100:.1f}%)")
    
    # Example 4: Analyzing prompt templates
    print("\n" + "-"*80)
    print("EXAMPLE 4: Prompt Template Analysis")
    print("-"*80)
    
    templates = {
        "Classification": "Classify this text into one of these categories: {categories}. Text: {text}",
        "Summarization": "Summarize: {text}",
        "Translation": "Translate to {language}: {text}",
        "Q&A": "Question: {question}\nContext: {context}\nAnswer:",
        "Code Generation": "Generate {language} code for: {task}",
    }
    
    print(f"{'Template Type':<20} | {'Base Tokens':>12} | {'Example Total':>14}")
    print("-" * 50)
    
    for template_type, template in templates.items():
        base_tokens = count_tokens(template, tokenizer=Tokenizer.CL100K_BASE)
        
        # Simulate with example values
        if "{categories}" in template:
            filled = template.format(categories="positive, negative, neutral", text="This is great!")
        elif "{language}" in template and "{task}" in template:
            filled = template.format(language="Python", task="sort a list")
        elif "{language}" in template:
            filled = template.format(language="Spanish", text="Hello world")
        elif "{question}" in template:
            filled = template.format(question="What is AI?", context="AI is artificial intelligence")
        else:
            filled = template.format(text="Sample text here")
        
        filled_tokens = count_tokens(filled, tokenizer=Tokenizer.CL100K_BASE)
        
        print(f"{template_type:<20} | {base_tokens:>12} | {filled_tokens:>14}")
    
    # Example 5: Optimizing few-shot examples
    print("\n" + "-"*80)
    print("EXAMPLE 5: Few-Shot Example Optimization")
    print("-"*80)
    
    verbose_few_shot = """
    Here are some examples of how to classify sentiment:
    
    Example 1:
    Input: "I absolutely love this product! It's amazing!"
    Output: The sentiment of this text is positive.
    
    Example 2:
    Input: "This is the worst experience I've ever had."
    Output: The sentiment of this text is negative.
    
    Example 3:
    Input: "The product is okay, nothing special."
    Output: The sentiment of this text is neutral.
    
    Now, please classify the following text:
    """
    
    optimized_few_shot = """
    Classify sentiment (positive/negative/neutral):
    
    "I love this!" -> positive
    "Worst experience ever." -> negative
    "It's okay." -> neutral
    
    Classify:
    """
    
    print("Verbose few-shot prompt:")
    verbose_fs_tokens = count_tokens(verbose_few_shot, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {verbose_fs_tokens}\n")
    
    print("Optimized few-shot prompt:")
    optimized_fs_tokens = count_tokens(optimized_few_shot, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {optimized_fs_tokens}\n")
    
    print(f"Optimization:")
    print(f"  Tokens saved: {verbose_fs_tokens - optimized_fs_tokens}")
    print(f"  Reduction: {(verbose_fs_tokens - optimized_fs_tokens) / verbose_fs_tokens * 100:.1f}%")
    
    # Example 6: System message optimization
    print("\n" + "-"*80)
    print("EXAMPLE 6: System Message Optimization")
    print("-"*80)
    
    verbose_system = """
    You are an artificial intelligence assistant that has been specifically designed
    and trained to help users with their questions and tasks. You should always strive
    to be helpful, harmless, and honest in all of your interactions with users. When
    a user asks you a question, you should provide accurate and relevant information
    to the best of your knowledge and abilities. You should also be polite and
    professional in your communication style at all times.
    """
    
    concise_system = """
    You are a helpful AI assistant. Provide accurate, relevant answers. Be polite and professional.
    """
    
    print("Verbose system message:")
    verbose_sys_tokens = count_tokens(verbose_system, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {verbose_sys_tokens}\n")
    
    print("Concise system message:")
    concise_sys_tokens = count_tokens(concise_system, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {concise_sys_tokens}\n")
    
    savings = verbose_sys_tokens - concise_sys_tokens
    print(f"Tokens saved per conversation: {savings}")
    print(f"For 10,000 conversations: {savings * 10000:,} tokens")
    
    # Example 7: Using the optimize_token_usage utility
    print("\n" + "-"*80)
    print("EXAMPLE 7: Token Usage Optimization Utility")
    print("-"*80)
    
    prompts_to_analyze = [
        "Explain quantum computing",
        "Please provide a detailed explanation of quantum computing, including all the technical details and making sure to cover every aspect thoroughly and comprehensively",
    ]
    
    print("Analyzing prompts with optimize_token_usage:\n")
    
    for i, prompt in enumerate(prompts_to_analyze, 1):
        print(f"Prompt {i}:")
        print(f"  {prompt[:70]}...")
        
        analysis = optimize_token_usage(
            prompt,
            max_tokens=50,
            tokenizer=Tokenizer.CL100K_BASE
        )
        
        print(f"  Tokens: {analysis['token_count']}")
        print(f"  Characters: {analysis['char_count']}")
        print(f"  Ratio: {analysis['tokens_per_char']:.4f} tokens/char")
        print(f"  Exceeds limit: {analysis['exceeds_limit']}")
        print(f"  Suggestion: {analysis['suggested_action']}")
        print()
    
    # Example 8: Real-world optimization case study
    print("\n" + "-"*80)
    print("EXAMPLE 8: Real-World Optimization Case Study")
    print("-"*80)
    
    original_production_prompt = """
    You are an advanced AI customer support agent for TechCorp Inc. Your primary 
    responsibility is to assist customers with their inquiries, concerns, and issues 
    related to our products and services. You should always maintain a professional 
    and courteous demeanor. When responding to customers, please ensure that you:
    
    1. Carefully read and understand their question or issue
    2. Provide accurate and helpful information based on our knowledge base
    3. If you don't know the answer, admit it and offer to escalate
    4. Always ask if there's anything else you can help with
    5. Thank the customer for contacting us
    
    Remember to personalize your responses and show empathy where appropriate.
    Please begin each response by acknowledging the customer's concern.
    """
    
    optimized_production_prompt = """
    You are TechCorp's AI support agent. Assist customers professionally with 
    product inquiries and issues. Provide accurate information; escalate if uncertain. 
    Show empathy and personalize responses.
    """
    
    print("Original production prompt:")
    original_prod_tokens = count_tokens(original_production_prompt, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {original_prod_tokens}\n")
    
    print("Optimized production prompt:")
    optimized_prod_tokens = count_tokens(optimized_production_prompt, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Tokens: {optimized_prod_tokens}\n")
    
    print("Impact Analysis:")
    tokens_saved = original_prod_tokens - optimized_prod_tokens
    print(f"  Tokens saved per interaction: {tokens_saved}")
    print(f"  Percentage reduction: {tokens_saved / original_prod_tokens * 100:.1f}%")
    
    # Calculate cost savings
    daily_interactions = 1000
    monthly_interactions = daily_interactions * 30
    monthly_tokens_saved = tokens_saved * monthly_interactions
    
    # Assume GPT-3.5-turbo at $0.0005 per 1K input tokens
    monthly_cost_savings = (monthly_tokens_saved / 1000) * 0.0005
    annual_cost_savings = monthly_cost_savings * 12
    
    print(f"\n  Operational Impact:")
    print(f"    Daily interactions: {daily_interactions:,}")
    print(f"    Monthly tokens saved: {monthly_tokens_saved:,}")
    print(f"    Monthly cost savings: ${monthly_cost_savings:.2f}")
    print(f"    Annual cost savings: ${annual_cost_savings:.2f}")
    
    # Example 9: Optimization best practices summary
    print("\n" + "-"*80)
    print("EXAMPLE 9: Optimization Best Practices")
    print("-"*80)
    
    best_practices = [
        ("Remove filler words", "please, very, really, actually"),
        ("Use concise language", "utilize -> use, implement -> add"),
        ("Eliminate redundancy", "repeat instructions, similar phrases"),
        ("Optimize formatting", "minimal whitespace, compact lists"),
        ("Shorten examples", "essential info only, remove fluff"),
    ]
    
    print("Prompt optimization best practices:\n")
    
    for i, (practice, example) in enumerate(best_practices, 1):
        print(f"{i}. {practice}")
        print(f"   Example: {example}\n")
    
    print("Key Takeaways:")
    print("- Every token counts in production systems")
    print("- Optimize system messages (used in every request)")
    print("- Compress few-shot examples where possible")
    print("- Test that optimized prompts maintain quality")
    print("- Monitor token usage metrics in production")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
