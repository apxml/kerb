"""Cost Estimation Example

This example demonstrates how to estimate API costs based on token usage,
which is critical for LLM applications to manage budgets, optimize spending,
and provide cost transparency to users.

Main concepts:
- Estimating costs for different models
- Comparing input vs. output token costs
- Calculating costs for conversations
- Budget planning and forecasting
"""

from kerb.tokenizer import (
    count_tokens,
    count_tokens_for_messages,
    batch_count_tokens,
    Tokenizer
)
from kerb.tokenizer.utils import estimate_cost


def main():
    """Run cost estimation examples."""
    
    print("="*80)
    print("API COST ESTIMATION EXAMPLE")
    print("="*80)
    
    # Example 1: Basic cost estimation
    print("\n" + "-"*80)
    print("EXAMPLE 1: Basic Cost Estimation")
    print("-"*80)
    
    prompt = "Explain quantum computing in simple terms."
    completion = (
        "Quantum computing is a type of computing that uses quantum mechanics principles. "
        "Unlike classical computers that use bits (0 or 1), quantum computers use qubits "
        "which can be in multiple states simultaneously. This allows them to solve certain "
        "complex problems much faster than classical computers."
    )
    
    input_tokens = count_tokens(prompt, tokenizer=Tokenizer.CL100K_BASE)
    output_tokens = count_tokens(completion, tokenizer=Tokenizer.CL100K_BASE)
    
    print(f"Prompt: {prompt}")
    print(f"Input tokens: {input_tokens}")
    print(f"\nCompletion: {completion}")
    print(f"Output tokens: {output_tokens}")
    
    # Calculate costs for different models
    models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    
    print("\nCost comparison across models:")
    for model in models:
        input_cost = estimate_cost(input_tokens, model=model, is_input=True)
        output_cost = estimate_cost(output_tokens, model=model, is_input=False)
        total_cost = input_cost + output_cost
        
        print(f"\n{model}:")
        print(f"  Input cost:  ${input_cost:.6f}")
        print(f"  Output cost: ${output_cost:.6f}")
        print(f"  Total cost:  ${total_cost:.6f}")
    
    # Example 2: Conversation cost estimation
    print("\n" + "-"*80)
    print("EXAMPLE 2: Multi-turn Conversation Cost")
    print("-"*80)
    
    conversation = [
        {"role": "system", "content": "You are a helpful programming tutor."},
        {"role": "user", "content": "How do I sort a list in Python?"},
        {"role": "assistant", "content": "You can use the sorted() function or the .sort() method. sorted() returns a new list, while .sort() modifies the list in place."},
        {"role": "user", "content": "What's the difference in performance?"},
        {"role": "assistant", "content": "Both have O(n log n) time complexity. The main difference is memory: sorted() creates a new list, while .sort() is more memory efficient."},
        {"role": "user", "content": "Can you show an example?"},
        {"role": "assistant", "content": "Sure! Here's an example:\n\nmy_list = [3, 1, 4, 1, 5]\nsorted_list = sorted(my_list)  # Returns [1, 1, 3, 4, 5]\nmy_list.sort()  # Modifies my_list to [1, 1, 3, 4, 5]"},
    ]
    
    print(f"Conversation: {len(conversation)} messages\n")
    
    # Calculate total tokens
    total_tokens = count_tokens_for_messages(conversation, tokenizer=Tokenizer.CL100K_BASE)
    
    # Estimate input vs output tokens (approximate)
    user_tokens = sum(
        count_tokens(msg["content"], tokenizer=Tokenizer.CL100K_BASE)
        for msg in conversation if msg["role"] in ["user", "system"]
    )
    assistant_tokens = sum(
        count_tokens(msg["content"], tokenizer=Tokenizer.CL100K_BASE)
        for msg in conversation if msg["role"] == "assistant"
    )
    
    # Add message formatting overhead to input
    message_overhead = total_tokens - (user_tokens + assistant_tokens)
    input_tokens = user_tokens + message_overhead
    output_tokens = assistant_tokens
    
    print(f"Token breakdown:")
    print(f"  Input tokens (user + system + overhead): {input_tokens}")
    print(f"  Output tokens (assistant): {output_tokens}")
    print(f"  Total tokens: {total_tokens}")
    
    model = "gpt-3.5-turbo"
    input_cost = estimate_cost(input_tokens, model=model, is_input=True)
    output_cost = estimate_cost(output_tokens, model=model, is_input=False)
    total_cost = input_cost + output_cost
    
    print(f"\nCost for {model}:")
    print(f"  Input cost:  ${input_cost:.6f}")
    print(f"  Output cost: ${output_cost:.6f}")
    print(f"  Total cost:  ${total_cost:.6f}")
    
    # Example 3: Batch processing cost estimation
    print("\n" + "-"*80)
    print("EXAMPLE 3: Batch Processing Cost Estimation")
    print("-"*80)
    
    # Simulate processing a batch of documents
    documents = [
        "Analyze the sentiment of this customer review: The product works great!",
        "Summarize this article: AI is transforming industries...",
        "Translate to Spanish: Hello, how are you today?",
        "Extract entities from: John Smith works at Microsoft in Seattle.",
        "Classify this text: Breaking news about technology stocks.",
    ] * 20  # 100 documents total
    
    print(f"Processing {len(documents)} documents")
    
    # Count input tokens
    input_token_counts = batch_count_tokens(documents, tokenizer=Tokenizer.CL100K_BASE)
    total_input_tokens = sum(input_token_counts)
    
    # Assume average output is 50 tokens per request
    avg_output_tokens = 50
    total_output_tokens = len(documents) * avg_output_tokens
    
    print(f"\nToken estimates:")
    print(f"  Total input tokens: {total_input_tokens:,}")
    print(f"  Average input per doc: {total_input_tokens / len(documents):.1f}")
    print(f"  Estimated output tokens: {total_output_tokens:,}")
    print(f"  Total tokens: {total_input_tokens + total_output_tokens:,}")
    
    # Calculate costs
    model = "gpt-3.5-turbo"
    input_cost = estimate_cost(total_input_tokens, model=model, is_input=True)
    output_cost = estimate_cost(total_output_tokens, model=model, is_input=False)
    total_cost = input_cost + output_cost
    
    print(f"\nEstimated cost for {model}:")
    print(f"  Input:  ${input_cost:.4f}")
    print(f"  Output: ${output_cost:.4f}")
    print(f"  Total:  ${total_cost:.4f}")
    print(f"  Cost per document: ${total_cost / len(documents):.6f}")
    
    # Example 4: Monthly cost projection
    print("\n" + "-"*80)
    print("EXAMPLE 4: Monthly Cost Projection")
    print("-"*80)
    
    # Simulate daily usage patterns
    daily_scenarios = {
        "Customer Support Chat": {
            "conversations_per_day": 500,
            "avg_messages_per_conversation": 6,
            "avg_tokens_per_message": 50,
            "model": "gpt-3.5-turbo",
        },
        "Document Summarization": {
            "documents_per_day": 100,
            "avg_input_tokens": 2000,
            "avg_output_tokens": 200,
            "model": "gpt-4-turbo",
        },
        "Code Generation": {
            "requests_per_day": 50,
            "avg_input_tokens": 300,
            "avg_output_tokens": 400,
            "model": "gpt-4",
        },
    }
    
    print("Monthly cost projections (30 days):\n")
    
    total_monthly_cost = 0
    
    for scenario_name, config in daily_scenarios.items():
        print(f"{scenario_name}:")
        print(f"  Model: {config['model']}")
        
        if "conversations_per_day" in config:
            # Chat scenario
            daily_messages = config["conversations_per_day"] * config["avg_messages_per_conversation"]
            daily_input_tokens = daily_messages * config["avg_tokens_per_message"]
            daily_output_tokens = daily_messages * config["avg_tokens_per_message"]
            print(f"  Conversations/day: {config['conversations_per_day']}")
            print(f"  Avg messages/conversation: {config['avg_messages_per_conversation']}")
        else:
            # Document processing scenario
            if "documents_per_day" in config:
                count = config["documents_per_day"]
                print(f"  Documents/day: {count}")
            else:
                count = config["requests_per_day"]
                print(f"  Requests/day: {count}")
            
            daily_input_tokens = count * config["avg_input_tokens"]
            daily_output_tokens = count * config["avg_output_tokens"]
        
        print(f"  Daily tokens: {daily_input_tokens:,} input + {daily_output_tokens:,} output")
        
        # Calculate monthly costs
        monthly_input_tokens = daily_input_tokens * 30
        monthly_output_tokens = daily_output_tokens * 30
        
        monthly_input_cost = estimate_cost(monthly_input_tokens, model=config["model"], is_input=True)
        monthly_output_cost = estimate_cost(monthly_output_tokens, model=config["model"], is_input=False)
        monthly_cost = monthly_input_cost + monthly_output_cost
        
        print(f"  Monthly cost: ${monthly_cost:.2f}")
        print(f"    Input:  ${monthly_input_cost:.2f}")
        print(f"    Output: ${monthly_output_cost:.2f}")
        print()
        
        total_monthly_cost += monthly_cost
    
    print(f"Total estimated monthly cost: ${total_monthly_cost:.2f}")
    print(f"Total estimated annual cost: ${total_monthly_cost * 12:.2f}")
    
    # Example 5: Cost optimization recommendations
    print("\n" + "-"*80)
    print("EXAMPLE 5: Cost Optimization Analysis")
    print("-"*80)
    
    # Compare cost of using different models for same task
    task = "Summarize customer feedback"
    input_text = "The product is great! Fast shipping, excellent quality. " * 20
    expected_output_tokens = 100
    
    input_tokens = count_tokens(input_text, tokenizer=Tokenizer.CL100K_BASE)
    
    print(f"Task: {task}")
    print(f"Input tokens: {input_tokens}")
    print(f"Expected output tokens: {expected_output_tokens}")
    print(f"\nCost comparison:\n")
    
    models_to_compare = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    costs = []
    
    for model in models_to_compare:
        input_cost = estimate_cost(input_tokens, model=model, is_input=True)
        output_cost = estimate_cost(expected_output_tokens, model=model, is_input=False)
        total = input_cost + output_cost
        costs.append((model, total))
        
        print(f"{model}:")
        print(f"  Cost per request: ${total:.6f}")
        print(f"  Cost per 1,000 requests: ${total * 1000:.2f}")
        print(f"  Cost per 100,000 requests: ${total * 100000:.2f}")
        print()
    
    # Show potential savings
    costs.sort(key=lambda x: x[1])
    cheapest = costs[0]
    most_expensive = costs[-1]
    
    savings = most_expensive[1] - cheapest[1]
    savings_percent = (savings / most_expensive[1]) * 100
    
    print(f"Optimization opportunity:")
    print(f"  Cheapest option: {cheapest[0]} (${cheapest[1]:.6f} per request)")
    print(f"  Most expensive: {most_expensive[0]} (${most_expensive[1]:.6f} per request)")
    print(f"  Savings: ${savings:.6f} per request ({savings_percent:.1f}%)")
    print(f"  Annual savings at 100k requests/month: ${savings * 100000 * 12:.2f}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
