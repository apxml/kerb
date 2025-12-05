"""
LLM-as-Judge Evaluation.
========================

This example demonstrates how to use LLM-as-judge for evaluating outputs
when you don't have ground truth or want human-like quality assessment.
Common use cases:
- Evaluating open-ended generation tasks
- Comparing multiple model outputs
- Assessing qualities that are hard to measure automatically
- Replacing expensive human evaluation
"""

from kerb.evaluation import (
    llm_as_judge,
    pairwise_comparison,
    JudgmentCriterion,
)


def simulate_llm_judge(prompt: str) -> str:
    """
    Simulate an LLM judge response for demonstration.
    In production, replace this with actual LLM API calls.
    """

# %%
# Setup and Imports
# -----------------
    # Simple simulation based on prompt keywords
    if "relevance" in prompt.lower():
        if "python" in prompt.lower():
            return "Rating: 5\nReasoning: The response is highly relevant, directly answering the question about Python with accurate information."
        return "Rating: 3\nReasoning: The response is somewhat relevant but lacks specific details."
    
    if "accuracy" in prompt.lower():
        return "Rating: 5\nReasoning: The information provided is factually accurate and well-supported."
    
    if "coherence" in prompt.lower():
        return "Rating: 4\nReasoning: The response flows logically with good transitions between ideas."
    
    if "completeness" in prompt.lower():
        if "comprehensive" in prompt.lower():
            return "Rating: 5\nReasoning: The response thoroughly covers all aspects of the topic."
        return "Rating: 3\nReasoning: The response addresses the main point but lacks some details."
    
    if "helpfulness" in prompt.lower():
        return "Rating: 4\nReasoning: The response provides practical and useful information."
    
    # Pairwise comparison
    if "which is better" in prompt.lower() or "compare" in prompt.lower():
        if "Output A" in prompt and "comprehensive" in prompt.lower():
            return "Winner: A\nReasoning: Output A is more comprehensive and provides better detail."
        return "Winner: B\nReasoning: Output B is more concise while maintaining clarity."
    
    return "Rating: 4\nReasoning: Good quality response overall."



# %%
# Evaluate Relevance
# ------------------

def evaluate_relevance():
    """Evaluate how relevant outputs are to the prompt."""
    print("=" * 80)
    print("RELEVANCE EVALUATION")
    print("=" * 80)
    
    prompt = "What is Python programming language?"
    
    outputs = [
        {
            "label": "Highly Relevant",
            "text": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple paradigms and has extensive libraries.",
        },
        {
            "label": "Partially Relevant",
            "text": "Programming languages are used to write software. There are many types of programming languages available.",
        },
        {
            "label": "Not Relevant",
            "text": "Machine learning is an important field in computer science that involves training models on data.",
        },
    ]
    
    print(f"\nPrompt: {prompt}")
    print("\n" + "-" * 80)
    
    for output in outputs:
        print(f"\n{output['label']}:")
        print(f"\"{output['text']}\"")
        
        result = llm_as_judge(
            output['text'],
            JudgmentCriterion.RELEVANCE,
            context=prompt,
            llm_function=simulate_llm_judge
        )
        
        print(f"\nRelevance Score: {result.score:.1f}/5.0")
        print(f"Reasoning: {result.details.get('reasoning', 'N/A')}")


def evaluate_accuracy():
    """Evaluate factual accuracy of outputs."""
    print("\n" + "=" * 80)
    print("ACCURACY EVALUATION")
    print("=" * 80)
    
    question = "What is the capital of France?"
    
    answers = [
        {"label": "Accurate", "text": "The capital of France is Paris."},
        {"label": "Inaccurate", "text": "The capital of France is Lyon."},
    ]
    
    print(f"\nQuestion: {question}")
    print("\n" + "-" * 80)
    
    for answer in answers:
        print(f"\n{answer['label']} Answer:")
        print(f"\"{answer['text']}\"")
        
        result = llm_as_judge(
            answer['text'],
            JudgmentCriterion.ACCURACY,
            context=question,
            reference="Paris",
            llm_function=simulate_llm_judge
        )
        
        print(f"\nAccuracy Score: {result.score:.1f}/5.0")
        print(f"Assessment: {result.details.get('reasoning', 'N/A')}")



# %%
# Evaluate Multiple Criteria
# --------------------------

def evaluate_multiple_criteria():
    """Evaluate outputs across multiple judgment criteria."""
    print("\n" + "=" * 80)
    print("MULTI-CRITERIA EVALUATION")
    print("=" * 80)
    
    prompt = "Explain how neural networks work."
    output = """
    Neural networks are computational models inspired by the human brain. They consist
    of interconnected nodes (neurons) organized in layers. Each connection has a weight
    that is adjusted during training. The network learns by processing input data through
    these layers, making predictions, and updating weights based on errors.
    """
    
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output.strip()}")
    print("\n" + "-" * 80)
    print("Evaluating across multiple criteria:")
    print("-" * 80)
    
    criteria = [
        JudgmentCriterion.RELEVANCE,
        JudgmentCriterion.ACCURACY,
        JudgmentCriterion.COMPLETENESS,
        JudgmentCriterion.COHERENCE,
        JudgmentCriterion.HELPFULNESS,
    ]
    
    scores = {}
    for criterion in criteria:
        result = llm_as_judge(
            output,
            criterion,
            context=prompt,
            llm_function=simulate_llm_judge
        )
        scores[criterion.value] = result.score
        print(f"{criterion.value.capitalize()}: {result.score:.1f}/5.0")
    
    average_score = sum(scores.values()) / len(scores)
    print(f"\nAverage Score: {average_score:.2f}/5.0")


def compare_model_outputs():
    """Compare outputs from different models using pairwise comparison."""
    print("\n" + "=" * 80)
    print("PAIRWISE COMPARISON")
    print("=" * 80)
    
    prompt = "Write a brief explanation of recursion."
    
    comparisons = [
        {
            "name": "Completeness",
            "output_a": "Recursion is when a function calls itself.",
            "output_b": "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems. Each recursive call works on a simpler version until reaching a base case.",
        },
        {
            "name": "Clarity",
            "output_a": "Recursive functions utilize self-referential invocation patterns to achieve iterative computation through stack-based execution.",
            "output_b": "A recursive function is one that calls itself to solve a problem step by step.",
        },
    ]
    
    print(f"\nPrompt: {prompt}\n")
    
    for comparison in comparisons:
        print("=" * 80)
        print(f"Comparing: {comparison['name']}")
        print("=" * 80)
        print(f"\nOutput A: \"{comparison['output_a']}\"")
        print(f"Output B: \"{comparison['output_b']}\"")
        
        result = pairwise_comparison(
            comparison['output_a'],
            comparison['output_b'],
            comparison['name'].lower(),
            context=prompt,
            llm_function=simulate_llm_judge
        )
        
        print(f"\nWinner: Output {result.winner}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
        print()



# %%
# Evaluate Creative Writing
# -------------------------

def evaluate_creative_writing():
    """Evaluate creative writing outputs using multiple judges."""
    print("=" * 80)
    print("CREATIVE WRITING EVALUATION")
    print("=" * 80)
    
    prompt = "Write a short opening for a mystery story."
    output = """
    The old mansion stood silent against the stormy sky. Detective Sarah Chen
    pulled her coat tighter as she approached the heavy oak door. Inside, three
    people waited—one of them a murderer. She had until dawn to find the truth.
    """
    
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output.strip()}")
    print("\n" + "-" * 80)
    print("Creative Writing Assessment:")
    print("-" * 80)
    
    # Evaluate creative aspects
    creativity = llm_as_judge(
        output,
        "creativity",
        context=prompt,
        llm_function=simulate_llm_judge
    )
    print(f"Creativity: {creativity.score:.1f}/5.0")
    
    engagement = llm_as_judge(
        output,
        "engagement",
        context=prompt,
        llm_function=simulate_llm_judge
    )
    print(f"Engagement: {engagement.score:.1f}/5.0")
    
    coherence = llm_as_judge(
        output,
        JudgmentCriterion.COHERENCE,
        context=prompt,
        llm_function=simulate_llm_judge
    )
    print(f"Coherence: {coherence.score:.1f}/5.0")


def batch_evaluate_chatbot():
    """Batch evaluate chatbot responses across conversations."""
    print("\n" + "=" * 80)
    print("BATCH CHATBOT EVALUATION")
    print("=" * 80)
    
    conversations = [
        {
            "user": "How do I reset my password?",
            "bot": "You can reset your password by clicking 'Forgot Password' on the login page and following the email instructions.",
        },
        {
            "user": "What are your business hours?",
            "bot": "We're open Monday to Friday, 9 AM to 5 PM EST.",
        },
        {
            "user": "Can you explain quantum physics?",
            "bot": "I can help with account-related questions. For technical topics, please visit our knowledge base.",
        },
    ]
    
    print("\nEvaluating chatbot responses for helpfulness:\n")
    
    total_score = 0
    for i, conv in enumerate(conversations, 1):
        print(f"Conversation {i}:")
        print(f"  User: {conv['user']}")
        print(f"  Bot: {conv['bot']}")
        
        result = llm_as_judge(
            conv['bot'],
            JudgmentCriterion.HELPFULNESS,
            context=conv['user'],
            llm_function=simulate_llm_judge
        )
        
        print(f"  Helpfulness: {result.score:.1f}/5.0")
        total_score += result.score
        print()
    
    average = total_score / len(conversations)
    print(f"Average Helpfulness Score: {average:.2f}/5.0")



# %%
# Main
# ----

def main():
    """Run all LLM-as-judge examples."""
    print("\nNOTE: This example uses simulated LLM responses for demonstration.")
    print("In production, integrate with actual LLM APIs (OpenAI, Anthropic, etc.)\n")
    
    evaluate_relevance()
    evaluate_accuracy()
    evaluate_multiple_criteria()
    compare_model_outputs()
    evaluate_creative_writing()
    batch_evaluate_chatbot()
    
    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. LLM-as-judge enables evaluation without ground truth")
    print("2. Use multiple criteria for comprehensive assessment")
    print("3. Pairwise comparison helps choose better outputs")
    print("4. Suitable for open-ended and creative tasks")
    print("5. Scale evaluation while maintaining quality insights")
    print("\nProduction Tips:")
    print("- Integrate with OpenAI, Anthropic, or other LLM APIs")
    print("- Use temperature=0 for consistent judgments")
    print("- Validate judge outputs for critical applications")
    print("- Consider cost vs. quality tradeoffs")


if __name__ == "__main__":
    main()
