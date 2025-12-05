"""
Quality Assessment for LLM Outputs.
===================================

This example demonstrates how to assess the quality of LLM-generated content
without requiring ground truth references. Common use cases:
- Detecting hallucinations in generated content
- Assessing faithfulness to source documents (RAG systems)
- Evaluating answer relevance to questions
- Measuring coherence and fluency
"""

from kerb.evaluation import (
    detect_hallucination,
    assess_faithfulness,
    assess_answer_relevance,
    assess_coherence,
    assess_fluency,
)


def detect_hallucinations_in_rag():
    """Detect when model generates unsupported claims in RAG applications."""
    print("=" * 80)
    print("HALLUCINATION DETECTION IN RAG SYSTEMS")
    print("=" * 80)
    
    # Context retrieved from knowledge base
    context = """

# %%
# Setup and Imports
# -----------------
    Python was created by Guido van Rossum and first released in 1991.
    It emphasizes code readability with significant indentation.
    Python supports multiple programming paradigms including procedural,
    object-oriented, and functional programming.
    """
    
    test_outputs = [
        {
            "label": "Accurate",
            "text": "Python was created by Guido van Rossum in 1991 and emphasizes code readability.",
        },
        {
            "label": "Partially hallucinated",
            "text": "Python was created by Guido van Rossum in 1991 and is the fastest programming language.",
        },
        {
            "label": "Highly hallucinated",
            "text": "Python was created by Dennis Ritchie in 1985 for systems programming.",
        },
    ]
    
    print("\nContext from knowledge base:")
    print(context.strip())
    print("\n" + "-" * 80)
    
    for test in test_outputs:
        print(f"\n{test['label']} Output:")
        print(f"\"{test['text']}\"")
        
        result = detect_hallucination(test['text'], context)
        
        print(f"\nHallucination Score: {result.score:.3f} (0=none, 1=high)")
        print(f"Assessment: {'PASS' if result.passed else 'FAIL'}")
        if result.details:
            print(f"Details: {result.details}")


def assess_document_faithfulness():
    """Assess faithfulness of summaries and generated content to source."""
    print("\n" + "=" * 80)
    print("FAITHFULNESS ASSESSMENT")
    print("=" * 80)
    
    source_document = """
    Climate change is causing global temperatures to rise at an unprecedented rate.
    Scientists agree that human activities, particularly fossil fuel burning,
    are the primary drivers. The effects include melting ice caps, rising sea levels,
    and more frequent extreme weather events. Immediate action is needed to reduce
    carbon emissions and transition to renewable energy sources.
    """
    
    summaries = [
        {
            "label": "Faithful summary",
            "text": "Climate change, driven by human fossil fuel use, is raising temperatures and causing environmental impacts like melting ice and extreme weather.",
        },
        {
            "label": "Unfaithful summary",
            "text": "Climate change is a natural cycle that has occurred throughout Earth's history and is not primarily caused by human activity.",
        },
    ]
    
    print("\nSource Document:")
    print(source_document.strip())
    print("\n" + "-" * 80)
    
    for summary in summaries:
        print(f"\n{summary['label']}:")
        print(f"\"{summary['text']}\"")
        
        result = assess_faithfulness(summary['text'], source_document)
        
        print(f"\nFaithfulness Score: {result.score:.3f}")
        print(f"Assessment: {'FAITHFUL' if result.passed else 'UNFAITHFUL'}")



# %%
# Evaluate Answer Relevance
# -------------------------

def evaluate_answer_relevance():
    """Evaluate whether answers are relevant to questions asked."""
    print("\n" + "=" * 80)
    print("ANSWER RELEVANCE EVALUATION")
    print("=" * 80)
    
    qa_examples = [
        {
            "question": "What are the health benefits of exercise?",
            "answer": "Exercise improves cardiovascular health, strengthens muscles, boosts mental health, and helps maintain a healthy weight.",
            "expected": "Relevant",
        },
        {
            "question": "What are the health benefits of exercise?",
            "answer": "Many people enjoy going to the gym in the morning. Fitness centers offer various equipment and classes.",
            "expected": "Partially relevant",
        },
        {
            "question": "What are the health benefits of exercise?",
            "answer": "The stock market has been volatile recently due to economic uncertainty and inflation concerns.",
            "expected": "Not relevant",
        },
    ]
    
    print("\nEvaluating answer relevance for Q&A system:\n")
    
    for i, example in enumerate(qa_examples, 1):
        print(f"Example {i} (Expected: {example['expected']})")
        print(f"Question: {example['question']}")
        print(f"Answer: {example['answer']}")
        
        result = assess_answer_relevance(example['answer'], example['question'])
        
        print(f"Relevance Score: {result.score:.3f}")
        print(f"Assessment: {'RELEVANT' if result.passed else 'NOT RELEVANT'}")
        print()


def measure_text_coherence():
    """Measure coherence and logical flow of generated text."""
    print("=" * 80)
    print("COHERENCE ASSESSMENT")
    print("=" * 80)
    
    texts = [
        {
            "label": "Coherent",
            "text": """
            Machine learning is a subset of artificial intelligence. It enables computers
            to learn from data without explicit programming. This learning process involves
            training models on datasets, which then make predictions on new data.
            """,
        },
        {
            "label": "Less coherent",
            "text": """
            Machine learning uses algorithms. Paris is the capital of France. 
            Neural networks have layers. The weather is nice today. Data is important.
            """,
        },
    ]
    
    print("\nEvaluating text coherence:\n")
    
    for text_sample in texts:
        print(f"{text_sample['label']} Text:")
        print(text_sample['text'].strip())
        
        result = assess_coherence(text_sample['text'])
        
        print(f"\nCoherence Score: {result.score:.3f}")
        print(f"Assessment: {'COHERENT' if result.passed else 'INCOHERENT'}")
        print("-" * 80 + "\n")



# %%
# Measure Text Fluency
# --------------------

def measure_text_fluency():
    """Measure fluency and naturalness of generated text."""
    print("=" * 80)
    print("FLUENCY ASSESSMENT")
    print("=" * 80)
    
    texts = [
        {
            "label": "Fluent",
            "text": "The quick brown fox jumps gracefully over the lazy dog.",
        },
        {
            "label": "Less fluent",
            "text": "The quick fox brown over jumps the dog lazy.",
        },
        {
            "label": "Very fluent",
            "text": "Artificial intelligence is revolutionizing industries by automating complex tasks and providing valuable insights from data.",
        },
    ]
    
    print("\nEvaluating text fluency:\n")
    
    for text_sample in texts:
        print(f"{text_sample['label']}:")
        print(f"\"{text_sample['text']}\"")
        
        result = assess_fluency(text_sample['text'])
        
        print(f"Fluency Score: {result.score:.3f}")
        print(f"Assessment: {'FLUENT' if result.passed else 'NOT FLUENT'}")
        print()


def comprehensive_quality_check():
    """Run comprehensive quality checks on generated content."""
    print("=" * 80)
    print("COMPREHENSIVE QUALITY ASSESSMENT")
    print("=" * 80)
    
    context = "Quantum computing uses quantum bits or qubits, which can exist in superposition states."
    question = "How does quantum computing differ from classical computing?"
    
    generated_answer = """
    Quantum computing differs from classical computing by using qubits instead of 
    classical bits. These qubits can exist in superposition, allowing quantum computers
    to process multiple states simultaneously and solve certain problems exponentially faster.
    """
    
    print("\nContext:", context)
    print("Question:", question)
    print("Generated Answer:", generated_answer.strip())
    
    print("\n" + "-" * 80)
    print("Quality Assessment Results:")
    print("-" * 80)
    
    # Run all quality checks
    faithfulness = assess_faithfulness(generated_answer, context)
    print(f"Faithfulness: {faithfulness.score:.3f} ({'PASS' if faithfulness.passed else 'FAIL'})")
    
    relevance = assess_answer_relevance(generated_answer, question)
    print(f"Relevance: {relevance.score:.3f} ({'PASS' if relevance.passed else 'FAIL'})")
    
    coherence = assess_coherence(generated_answer)
    print(f"Coherence: {coherence.score:.3f} ({'PASS' if coherence.passed else 'FAIL'})")
    
    fluency = assess_fluency(generated_answer)
    print(f"Fluency: {fluency.score:.3f} ({'PASS' if fluency.passed else 'FAIL'})")
    
    hallucination = detect_hallucination(generated_answer, context)
    print(f"Hallucination: {hallucination.score:.3f} ({'PASS' if hallucination.passed else 'FAIL'})")
    
    # Calculate overall quality score
    overall_quality = (
        faithfulness.score + relevance.score + coherence.score + 
        fluency.score + (1 - hallucination.score)
    ) / 5
    
    print(f"\nOverall Quality Score: {overall_quality:.3f}")



# %%
# Main
# ----

def main():
    """Run all quality assessment examples."""
    detect_hallucinations_in_rag()
    assess_document_faithfulness()
    evaluate_answer_relevance()
    print()
    measure_text_coherence()
    measure_text_fluency()
    comprehensive_quality_check()
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Hallucination detection is critical for RAG applications")
    print("2. Faithfulness ensures generated content aligns with sources")
    print("3. Answer relevance prevents off-topic responses")
    print("4. Coherence and fluency measure text quality")
    print("5. Combine metrics for comprehensive quality assessment")


if __name__ == "__main__":
    main()
