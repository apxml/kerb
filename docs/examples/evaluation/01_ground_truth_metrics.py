"""Ground Truth Metrics for Translation and Summarization Evaluation.

This example demonstrates how to use reference-based metrics to evaluate
LLM outputs against ground truth answers. Common use cases:
- Machine translation quality assessment
- Summarization evaluation
- Question answering accuracy
- Text generation comparison
"""

from kerb.evaluation import (
    calculate_bleu,
    calculate_rouge,
    calculate_meteor,
    calculate_exact_match,
    calculate_f1_score,
    calculate_semantic_similarity,
)


def evaluate_translation_quality():
    """Evaluate machine translation output against reference translation."""
    print("=" * 80)
    print("TRANSLATION QUALITY EVALUATION")
    print("=" * 80)
    
    # Source: "Bonjour, comment allez-vous?"
    reference = "Hello, how are you?"
    candidates = [
        "Hello, how are you?",  # Perfect match
        "Hello, how are you doing?",  # Good but longer
        "Hi, how are you?",  # Synonym used
        "Hello, what's up?",  # Different phrasing
    ]
    
    print("\nReference translation:", reference)
    print("\nEvaluating different translation candidates:\n")
    
    for i, candidate in enumerate(candidates, 1):
        print(f"Candidate {i}: \"{candidate}\"")
        
        # BLEU score (precision-focused, n-gram overlap)
        bleu = calculate_bleu(candidate, reference)
        print(f"  BLEU: {bleu:.3f}")
        
        # ROUGE-L (longest common subsequence)
        rouge_l = calculate_rouge(candidate, reference, rouge_type="rouge-l")
        print(f"  ROUGE-L F1: {rouge_l['fmeasure']:.3f}")
        
        # F1 score (token-level precision and recall)
        f1 = calculate_f1_score(candidate, reference)
        print(f"  F1 Score: {f1:.3f}")
        
        # Exact match
        exact = calculate_exact_match(candidate, reference)
        print(f"  Exact Match: {exact}")
        
        print()


def evaluate_summarization():
    """Evaluate summarization quality with multiple reference summaries."""
    print("=" * 80)
    print("SUMMARIZATION EVALUATION")
    print("=" * 80)
    
    source_text = """
    Artificial intelligence (AI) is transforming the technology landscape.
    Machine learning, a subset of AI, enables systems to learn from data.
    Deep learning models have achieved breakthrough results in computer vision
    and natural language processing tasks.
    """
    
    reference_summaries = [
        "AI and machine learning are transforming technology with deep learning breakthroughs.",
        "Machine learning and deep learning are advancing AI capabilities significantly.",
    ]
    
    generated_summary = "AI is revolutionizing technology through machine learning and deep learning advances."
    
    print("\nGenerated Summary:")
    print(f"\"{generated_summary}\"\n")
    
    print("Evaluation against multiple references:\n")
    
    # BLEU with multiple references
    bleu = calculate_bleu(generated_summary, reference_summaries)
    print(f"BLEU Score: {bleu:.3f}")
    
    # ROUGE scores
    rouge_scores = calculate_rouge(generated_summary, reference_summaries[0])
    print(f"ROUGE-1 F1: {rouge_scores['fmeasure']:.3f}")
    
    rouge_scores = calculate_rouge(generated_summary, reference_summaries[0], rouge_type="rouge-2")
    print(f"ROUGE-2 F1: {rouge_scores['fmeasure']:.3f}")
    
    rouge_scores = calculate_rouge(generated_summary, reference_summaries[0], rouge_type="rouge-l")
    print(f"ROUGE-L F1: {rouge_scores['fmeasure']:.3f}")
    
    # METEOR (considers synonyms and word order)
    meteor = calculate_meteor(generated_summary, reference_summaries[0])
    print(f"METEOR Score: {meteor:.3f}")
    
    # Semantic similarity (meaning-based comparison)
    semantic_sim = calculate_semantic_similarity(generated_summary, reference_summaries[0])
    print(f"Semantic Similarity: {semantic_sim:.3f}")


def evaluate_qa_answers():
    """Evaluate question answering accuracy."""
    print("\n" + "=" * 80)
    print("QUESTION ANSWERING EVALUATION")
    print("=" * 80)
    
    qa_pairs = [
        {
            "question": "What is the capital of France?",
            "reference": "Paris",
            "answer": "Paris",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "reference": "William Shakespeare",
            "answer": "Shakespeare",
        },
        {
            "question": "What is 2 + 2?",
            "reference": "4",
            "answer": "four",
        },
    ]
    
    print("\nEvaluating QA system answers:\n")
    
    total_exact = 0
    total_f1 = 0
    
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"Reference: {qa['reference']}")
        print(f"Answer: {qa['answer']}")
        
        exact = calculate_exact_match(qa['answer'], qa['reference'])
        f1 = calculate_f1_score(qa['answer'], qa['reference'])
        
        print(f"  Exact Match: {exact}")
        print(f"  F1 Score: {f1:.3f}")
        
        total_exact += exact
        total_f1 += f1
        print()
    
    print(f"Overall Metrics:")
    print(f"  Exact Match Rate: {total_exact / len(qa_pairs):.1%}")
    print(f"  Average F1 Score: {total_f1 / len(qa_pairs):.3f}")


def compare_metric_characteristics():
    """Demonstrate different characteristics of each metric."""
    print("\n" + "=" * 80)
    print("METRIC CHARACTERISTICS COMPARISON")
    print("=" * 80)
    
    reference = "The quick brown fox jumps over the lazy dog"
    
    test_cases = [
        ("Identical", "The quick brown fox jumps over the lazy dog"),
        ("Reordered", "The lazy dog jumps over the quick brown fox"),
        ("Synonyms", "The fast brown fox leaps over the sleepy dog"),
        ("Partial", "The quick brown fox"),
        ("Different", "A cat sits on the mat"),
    ]
    
    print("\nReference:", reference)
    print("\nMetric behavior with different candidate types:\n")
    print(f"{'Type':<12} {'BLEU':<8} {'ROUGE-L':<10} {'F1':<8} {'Semantic':<10}")
    print("-" * 60)
    
    for case_type, candidate in test_cases:
        bleu = calculate_bleu(candidate, reference)
        rouge_l = calculate_rouge(candidate, reference, rouge_type="rouge-l")
        f1 = calculate_f1_score(candidate, reference)
        semantic = calculate_semantic_similarity(candidate, reference)
        
        print(f"{case_type:<12} {bleu:<8.3f} {rouge_l['fmeasure']:<10.3f} "
              f"{f1:<8.3f} {semantic:<10.3f}")
    
    print("\nKey Insights:")
    print("- BLEU: Sensitive to exact n-gram matches, penalizes length differences")
    print("- ROUGE-L: Focuses on longest common subsequence, handles reordering better")
    print("- F1: Token-level precision/recall, good for partial matches")
    print("- Semantic: Captures meaning similarity even with different words")


def main():
    """Run all ground truth metric examples."""
    evaluate_translation_quality()
    print("\n")
    evaluate_summarization()
    evaluate_qa_answers()
    compare_metric_characteristics()
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Use BLEU for translation tasks requiring precise n-gram matches")
    print("2. ROUGE is excellent for summarization evaluation")
    print("3. F1 score works well for QA and partial matches")
    print("4. Semantic similarity captures meaning beyond exact words")
    print("5. Combine multiple metrics for comprehensive evaluation")


if __name__ == "__main__":
    main()
