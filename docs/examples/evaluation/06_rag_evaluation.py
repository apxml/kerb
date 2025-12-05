"""
RAG (Retrieval-Augmented Generation) Evaluation.
================================================

This example demonstrates comprehensive evaluation strategies for RAG systems,
which combine retrieval and generation. Common use cases:
- Evaluating document retrieval quality
- Assessing answer faithfulness to retrieved context
- Detecting hallucinations in RAG outputs
- End-to-end RAG system evaluation
"""

from kerb.evaluation import (
    assess_faithfulness,
    detect_hallucination,
    assess_answer_relevance,
    calculate_f1_score,
    run_benchmark,
    TestCase,
)


class SimpleRAGSystem:
    """Simple RAG system simulator for demonstration."""
    
    def __init__(self):
        # Simulated knowledge base
        self.documents = {
            "doc1": "Python was created by Guido van Rossum and released in 1991. It emphasizes code readability.",
            "doc2": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
            "doc3": "The Transformer architecture, introduced in 2017, revolutionized NLP with self-attention mechanisms.",
            "doc4": "RAG systems combine retrieval and generation to produce more accurate and grounded responses.",
            "doc5": "Climate change is primarily caused by greenhouse gas emissions from human activities.",
        }
    
    def retrieve(self, query: str, top_k: int = 2):
        """Simple keyword-based retrieval."""

# %%
# Setup and Imports
# -----------------
        scores = {}
        query_lower = query.lower()
        
        for doc_id, doc_text in self.documents.items():
            doc_lower = doc_text.lower()
            # Simple scoring: count matching words
            query_words = set(query_lower.split())
            doc_words = set(doc_lower.split())
            overlap = len(query_words & doc_words)
            scores[doc_id] = overlap
        
        # Return top-k documents
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, self.documents[doc_id]) for doc_id, _ in sorted_docs[:top_k]]
    

# %%
# Generate
# --------

    def generate(self, query: str, context: str) -> str:
        """Simple generation based on context."""
        # Extract key information from context
        if "python" in query.lower():
            return "Python was created by Guido van Rossum in 1991 and emphasizes code readability."
        elif "machine learning" in query.lower():
            return "Machine learning is a subset of AI that enables systems to learn from data."
        elif "transformer" in query.lower():
            return "The Transformer architecture was introduced in 2017 and revolutionized NLP."
        elif "rag" in query.lower():
            return "RAG systems combine retrieval and generation for accurate responses."
        elif "climate" in query.lower():
            return "Climate change is primarily caused by human greenhouse gas emissions."
        else:
            return "I don't have specific information about that in my knowledge base."
    
    def answer_question(self, query: str):
        """Full RAG pipeline: retrieve then generate."""
        retrieved_docs = self.retrieve(query)
        context = " ".join([doc for _, doc in retrieved_docs])
        answer = self.generate(query, context)
        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "context": context
        }



# %%
# Evaluate Retrieval Quality
# --------------------------

def evaluate_retrieval_quality():
    """Evaluate the quality of document retrieval."""
    print("=" * 80)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 80)
    
    rag = SimpleRAGSystem()
    
    test_queries = [
        {
            "query": "Who created Python?",
            "relevant_docs": ["doc1"],
        },
        {
            "query": "What is machine learning?",
            "relevant_docs": ["doc2"],
        },
        {
            "query": "Tell me about Transformers in NLP",
            "relevant_docs": ["doc3"],
        },
    ]
    
    print("\nEvaluating retrieval accuracy...\n")
    
    total_precision = 0
    total_recall = 0
    
    for test in test_queries:
        query = test["query"]
        relevant_docs = set(test["relevant_docs"])
        
        retrieved = rag.retrieve(query, top_k=2)
        retrieved_ids = set([doc_id for doc_id, _ in retrieved])
        
        # Calculate precision and recall
        true_positives = len(relevant_docs & retrieved_ids)
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0
        
        total_precision += precision
        total_recall += recall
        
        print(f"Query: {query}")
        print(f"  Retrieved: {list(retrieved_ids)}")
        print(f"  Relevant: {list(relevant_docs)}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print()
    
    print("-" * 80)
    print(f"Average Precision: {total_precision / len(test_queries):.2f}")
    print(f"Average Recall: {total_recall / len(test_queries):.2f}")


def evaluate_answer_faithfulness():
    """Evaluate whether generated answers are faithful to retrieved context."""
    print("\n" + "=" * 80)
    print("ANSWER FAITHFULNESS EVALUATION")
    print("=" * 80)
    
    rag = SimpleRAGSystem()
    
    queries = [
        "Who created Python and when?",
        "What is machine learning?",
        "How does climate change occur?",
    ]
    
    print("\nEvaluating whether answers stay faithful to retrieved context...\n")
    
    for query in queries:
        result = rag.answer_question(query)
        
        print(f"Query: {query}")
        print(f"Retrieved Context: {result['context'][:100]}...")
        print(f"Generated Answer: {result['answer']}")
        
        # Evaluate faithfulness
        faithfulness = assess_faithfulness(result['answer'], result['context'])
        
        print(f"Faithfulness Score: {faithfulness.score:.3f}")
        print(f"Assessment: {'FAITHFUL' if faithfulness.passed else 'UNFAITHFUL'}")
        print("-" * 80 + "\n")



# %%
# Detect Rag Hallucinations
# -------------------------

def detect_rag_hallucinations():
    """Detect hallucinations in RAG-generated answers."""
    print("=" * 80)
    print("HALLUCINATION DETECTION IN RAG")
    print("=" * 80)
    
    rag = SimpleRAGSystem()
    
    # Test with queries that might lead to hallucination
    test_cases = [
        {
            "query": "What is Python?",
            "context": "Python was created by Guido van Rossum and released in 1991.",
            "answer": "Python was created by Guido van Rossum in 1991 and emphasizes code readability.",
            "expected": "Low hallucination (faithful to context)",
        },
        {
            "query": "What is Python?",
            "context": "Python was created by Guido van Rossum and released in 1991.",
            "answer": "Python is the fastest programming language in the world and was created in 2000.",
            "expected": "High hallucination (contradicts context)",
        },
    ]
    
    print("\nTesting hallucination detection...\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test['expected']}")
        print(f"Query: {test['query']}")
        print(f"Context: {test['context']}")
        print(f"Answer: {test['answer']}")
        
        result = detect_hallucination(test['answer'], test['context'])
        
        print(f"\nHallucination Score: {result.score:.3f} (0=none, 1=high)")
        print(f"Status: {'PASS' if result.passed else 'FAIL - Likely hallucination'}")
        print("-" * 80 + "\n")


def evaluate_answer_relevance_rag():
    """Evaluate whether RAG answers are relevant to the question."""
    print("=" * 80)
    print("ANSWER RELEVANCE EVALUATION")
    print("=" * 80)
    
    rag = SimpleRAGSystem()
    
    queries = [
        "Who created Python?",
        "What is machine learning?",
        "Explain transformers in NLP",
    ]
    
    print("\nEvaluating answer relevance to questions...\n")
    
    total_relevance = 0
    
    for query in queries:
        result = rag.answer_question(query)
        
        print(f"Question: {query}")
        print(f"Answer: {result['answer']}")
        
        relevance = assess_answer_relevance(result['answer'], query)
        
        print(f"Relevance Score: {relevance.score:.3f}")
        print(f"Status: {'RELEVANT' if relevance.passed else 'NOT RELEVANT'}")
        print()
        
        total_relevance += relevance.score
    
    avg_relevance = total_relevance / len(queries)
    print("-" * 80)
    print(f"Average Relevance Score: {avg_relevance:.3f}")



# %%
# End To End Rag Evaluation
# -------------------------

def end_to_end_rag_evaluation():
    """Comprehensive end-to-end evaluation of RAG system."""
    print("\n" + "=" * 80)
    print("END-TO-END RAG EVALUATION")
    print("=" * 80)
    
    rag = SimpleRAGSystem()
    
    # Create test cases with expected answers
    test_cases = [
        TestCase(
            id="rag_1",
            input="Who created Python?",
            expected_output="Guido van Rossum"
        ),
        TestCase(
            id="rag_2",
            input="What is machine learning?",
            expected_output="subset of AI that learns from data"
        ),
        TestCase(
            id="rag_3",
            input="When was Transformer introduced?",
            expected_output="2017"
        ),
    ]
    
    print(f"\nRunning comprehensive evaluation on {len(test_cases)} test cases...\n")
    
    def rag_generator(query: str) -> str:
        """RAG system wrapped for benchmarking."""
        result = rag.answer_question(query)
        return result['answer']
    

# %%
# Rag Evaluator
# -------------

    def rag_evaluator(output: str, expected: str) -> float:
        """Evaluate RAG output using F1 score."""
        return calculate_f1_score(output, expected)
    
    # Run benchmark
    results = run_benchmark(
        test_cases,
        rag_generator,
        rag_evaluator,
        threshold=0.3,
        name="RAG System Benchmark"
    )
    
    print("-" * 80)
    print("Overall RAG System Performance:")
    print("-" * 80)
    print(f"Total Tests: {results.total_tests}")
    print(f"Passed: {results.passed_tests}")
    print(f"Failed: {results.failed_tests}")
    print(f"Pass Rate: {results.pass_rate:.1f}%")
    print(f"Average Score: {results.average_score:.3f}")


def evaluate_rag_with_multiple_metrics():
    """Evaluate RAG system using multiple complementary metrics."""
    print("\n" + "=" * 80)
    print("MULTI-METRIC RAG EVALUATION")
    print("=" * 80)
    
    rag = SimpleRAGSystem()
    query = "What is machine learning?"
    
    result = rag.answer_question(query)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved Context: {result['context'][:150]}...")
    print(f"Generated Answer: {result['answer']}")
    print("\n" + "-" * 80)
    print("Evaluation Results:")
    print("-" * 80)
    
    # 1. Faithfulness to context
    faithfulness = assess_faithfulness(result['answer'], result['context'])
    print(f"1. Faithfulness: {faithfulness.score:.3f} ({'PASS' if faithfulness.passed else 'FAIL'})")
    print(f"   (Is the answer grounded in retrieved context?)")
    
    # 2. Answer relevance
    relevance = assess_answer_relevance(result['answer'], query)
    print(f"\n2. Relevance: {relevance.score:.3f} ({'PASS' if relevance.passed else 'FAIL'})")
    print(f"   (Does the answer address the question?)")
    
    # 3. Hallucination detection
    hallucination = detect_hallucination(result['answer'], result['context'])
    print(f"\n3. Hallucination: {hallucination.score:.3f} ({'PASS' if hallucination.passed else 'FAIL'})")
    print(f"   (Does the answer contain unsupported claims?)")
    
    # Overall quality score
    quality_score = (faithfulness.score + relevance.score + (1 - hallucination.score)) / 3
    print(f"\n" + "=" * 80)
    print(f"Overall RAG Quality Score: {quality_score:.3f}")
    
    if quality_score >= 0.7:
        print("Assessment: EXCELLENT - High quality RAG output")
    elif quality_score >= 0.5:
        print("Assessment: GOOD - Acceptable RAG output")
    else:
        print("Assessment: NEEDS IMPROVEMENT - RAG output has issues")



# %%
# Compare Rag Configurations
# --------------------------

def compare_rag_configurations():
    """Compare different RAG configurations."""
    print("\n" + "=" * 80)
    print("COMPARING RAG CONFIGURATIONS")
    print("=" * 80)
    
    print("\nComparing RAG with different retrieval depths (top-k)...")
    print("Configuration A: top_k=1 (retrieve 1 document)")
    print("Configuration B: top_k=2 (retrieve 2 documents)")
    print("Configuration C: top_k=3 (retrieve 3 documents)")
    
    rag = SimpleRAGSystem()
    queries = [
        "What is Python?",
        "What is machine learning?",
        "What is a Transformer?",
    ]
    
    configs = {
        "top_k=1": [],
        "top_k=2": [],
        "top_k=3": [],
    }
    
    for query in queries:
        for k in [1, 2, 3]:
            retrieved = rag.retrieve(query, top_k=k)
            context = " ".join([doc for _, doc in retrieved])
            answer = rag.generate(query, context)
            
            # Evaluate
            faithfulness = assess_faithfulness(answer, context)
            configs[f"top_k={k}"].append(faithfulness.score)
    
    print("\n" + "-" * 80)
    print("Average Faithfulness Scores:")
    print("-" * 80)
    
    for config, scores in configs.items():
        avg_score = sum(scores) / len(scores)
        print(f"{config}: {avg_score:.3f}")
    
    best_config = max(configs.items(), key=lambda x: sum(x[1]) / len(x[1]))
    print(f"\nBest Configuration: {best_config[0]}")


def main():
    """Run all RAG evaluation examples."""
    evaluate_retrieval_quality()
    evaluate_answer_faithfulness()
    detect_rag_hallucinations()
    evaluate_answer_relevance_rag()
    end_to_end_rag_evaluation()
    evaluate_rag_with_multiple_metrics()
    compare_rag_configurations()
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. RAG evaluation requires assessing both retrieval and generation")
    print("2. Faithfulness ensures answers are grounded in retrieved context")
    print("3. Hallucination detection is critical for RAG reliability")
    print("4. Answer relevance checks if the question was actually answered")
    print("5. Use multiple metrics for comprehensive RAG evaluation")
    print("\nRAG-Specific Metrics:")
    print("- Retrieval Precision/Recall: Are the right documents retrieved?")
    print("- Faithfulness: Does the answer align with retrieved context?")
    print("- Answer Relevance: Does the answer address the question?")
    print("- Hallucination Rate: Are there unsupported claims?")


if __name__ == "__main__":
    main()
