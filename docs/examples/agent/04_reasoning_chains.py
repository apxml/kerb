"""Reasoning Chains Example

This example demonstrates how to build and execute reasoning chains.

Concepts covered:
- Creating sequential chains
- Creating parallel chains
- Conditional chains
- Step dependencies
- Chain composition
"""

from kerb.agent.reasoning import (
    Chain, SequentialChain, ParallelChain, Step, StepResult, StepStatus
)
from typing import Dict, Any


# Define individual step functions
def analyze_sentiment(context: Dict[str, Any]) -> str:
    """Analyze sentiment of text."""
    text = context.get('text', '')
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
    negative_words = ['bad', 'terrible', 'awful', 'poor', 'horrible']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def extract_keywords(context: Dict[str, Any]) -> list:
    """Extract keywords from text."""
    text = context.get('text', '')
    
    # Simple keyword extraction
    words = text.lower().split()
    # Filter out common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
    keywords = [w for w in words if w not in common_words and len(w) > 3]
    
    return list(set(keywords))[:5]  # Return top 5 unique keywords


def count_words(context: Dict[str, Any]) -> int:
    """Count words in text."""
    text = context.get('text', '')
    return len(text.split())


def summarize_analysis(context: Dict[str, Any]) -> str:
    """Create summary from previous analysis."""
    sentiment = context.get('sentiment', 'unknown')
    keywords = context.get('keywords', [])
    word_count = context.get('word_count', 0)
    
    summary = f"Text Analysis Summary:\n"
    summary += f"- Sentiment: {sentiment}\n"
    summary += f"- Word Count: {word_count}\n"
    summary += f"- Key Topics: {', '.join(keywords) if keywords else 'none found'}"
    
    return summary


def check_word_count(context: Dict[str, Any]) -> bool:
    """Condition: Check if text is long enough for detailed analysis."""
    word_count = context.get('word_count', 0)
    return word_count > 5


def main():
    """Run reasoning chains example."""
    
    print("="*80)
    print("REASONING CHAINS EXAMPLE")
    print("="*80)
    
    # Example text to analyze
    sample_text = "This is an excellent and amazing product. The quality is great and I'm very happy with my purchase."
    
    print(f"\nSample Text:")
    print(f'"{sample_text}"')
    
    # ========================================================================
    # SEQUENTIAL CHAIN
    # ========================================================================
    print("\n" + "="*80)
    print("1. SEQUENTIAL CHAIN")
    print("="*80)
    
    # Create steps
    step1 = Step(
        name="word_count",
        func=count_words,
        description="Count words in text"
    )
    
    step2 = Step(
        name="sentiment",
        func=analyze_sentiment,
        description="Analyze sentiment",
        depends_on=["word_count"]
    )
    
    step3 = Step(
        name="keywords",
        func=extract_keywords,
        description="Extract keywords",
        depends_on=["word_count"]
    )
    
    step4 = Step(
        name="summary",
        func=summarize_analysis,
        description="Create summary",
        depends_on=["sentiment", "keywords", "word_count"]
    )
    
    # Create sequential chain
    seq_chain = SequentialChain(
        steps=[step1, step2, step3, step4],
        name="TextAnalysisChain"
    )
    
    print(f"\nCreated chain: {seq_chain.name}")
    print(f"   Steps: {len(seq_chain.steps)}")
    
    # Execute chain
    context = {'text': sample_text}
    result = seq_chain.execute(context)
    
    print("\nSequential Execution Results:")
    print("-"*80)
    for step_name, step_result in result.get('results', {}).items():
        print(f"\n{step_name}:")
        print(f"  Status: {step_result.status.value if hasattr(step_result, 'status') else 'unknown'}")
        output = step_result.output if hasattr(step_result, 'output') else step_result
        print(f"  Output: {output}")
    
    # ========================================================================
    # PARALLEL CHAIN
    # ========================================================================
    print("\n" + "="*80)
    print("2. PARALLEL CHAIN")
    print("="*80)
    
    # Create parallel steps (can run independently)
    parallel_step1 = Step(
        name="sentiment_check",
        func=analyze_sentiment,
        description="Check sentiment in parallel"
    )
    
    parallel_step2 = Step(
        name="keyword_extraction",
        func=extract_keywords,
        description="Extract keywords in parallel"
    )
    
    parallel_step3 = Step(
        name="word_counting",
        func=count_words,
        description="Count words in parallel"
    )
    
    # Create parallel chain
    parallel_chain = ParallelChain(
        steps=[parallel_step1, parallel_step2, parallel_step3],
        name="ParallelAnalysis"
    )
    
    print(f"\nCreated parallel chain: {parallel_chain.name}")
    print(f"   Parallel steps: {len(parallel_chain.steps)}")
    
    # Execute parallel chain
    parallel_result = parallel_chain.execute(context)
    
    print("\nParallel Execution Results:")
    print("-"*80)
    for step_name, step_result in parallel_result.get('results', {}).items():
        print(f"\n{step_name}:")
        output = step_result.output if hasattr(step_result, 'output') else step_result
        print(f"  Output: {output}")
    
    # ========================================================================
    # CONDITIONAL CHAIN
    # ========================================================================
    print("\n" + "="*80)
    print("3. CONDITIONAL CHAIN")
    print("="*80)
    
    # Create conditional step
    conditional_step = Step(
        name="detailed_analysis",
        func=lambda ctx: f"Detailed analysis for {ctx.get('word_count', 0)} words",
        description="Perform detailed analysis only if text is long enough",
        condition=check_word_count
    )
    
    # Test with long text
    long_context = {'text': sample_text, 'word_count': 20}
    result_long = conditional_step.execute(long_context)
    
    print("\nTest 1: Long text (20 words)")
    print(f"   Condition met: {check_word_count(long_context)}")
    print(f"   Status: {result_long.status.value}")
    print(f"   Output: {result_long.output}")
    
    # Test with short text
    short_context = {'text': "Too short", 'word_count': 2}
    result_short = conditional_step.execute(short_context)
    
    print("\nTest 2: Short text (2 words)")
    print(f"   Condition met: {check_word_count(short_context)}")
    print(f"   Status: {result_short.status.value}")
    print(f"   Skipped: {result_short.status == StepStatus.SKIPPED}")
    
    # ========================================================================
    # COMPOSED CHAIN
    # ========================================================================
    print("\n" + "="*80)
    print("4. COMPOSED CHAIN")
    print("="*80)
    
    # Combine chains
    preprocessing_step = Step(
        name="preprocess",
        func=lambda ctx: {'text': ctx['text'].lower()},
        description="Preprocess text"
    )
    
    composed_chain = SequentialChain(
        steps=[preprocessing_step],
        name="ComposedAnalysis"
    )
    
    # Add analysis steps
    for step in [step1, step2, step3]:
        composed_chain.add_step(step)
    
    print(f"\nCreated composed chain: {composed_chain.name}")
    print(f"   Total steps: {len(composed_chain.steps)}")
    
    # Execute
    composed_result = composed_chain.execute(context)
    
    print("\nComposed Chain Result:")
    print("-"*80)
    print(f"Steps executed: {len(composed_result.get('results', {}))}")
    sentiment_result = composed_result.get('results', {}).get('sentiment')
    if sentiment_result:
        output = sentiment_result.output if hasattr(sentiment_result, 'output') else sentiment_result
        print(f"Final sentiment: {output}")
    
    print("\n" + "="*80)
    print("Reasoning chains example completed!")
    print("="*80)
    
    # Summary
    print("\nImportant concepts demonstrated:")
    print("-"*80)
    print("1. Sequential chains execute steps in order")
    print("2. Parallel chains execute steps concurrently")
    print("3. Steps can have dependencies on other steps")
    print("4. Conditional steps only execute when conditions are met")
    print("5. Chains can be composed and combined")
    print("6. Context flows through the chain, accumulating results")


if __name__ == "__main__":
    main()

