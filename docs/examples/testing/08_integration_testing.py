"""
Integration Testing Example
===========================

This example demonstrates end-to-end integration testing of complete LLM pipelines
and applications.

Main concepts:
- Testing complete LLM workflows
- Combining multiple testing techniques
- Integration test suites
- End-to-end validation
- Multi-component testing
- Production-like testing scenarios

Use cases for LLM developers:
- Testing RAG pipelines
- Validating chatbot flows
- Testing agent systems
- Verifying multi-step processes
- Integration with external systems
- End-to-end quality assurance
"""

from typing import List, Dict, Any
from kerb.testing import (
    MockLLM,
    MockBehavior,
    TestDataset,
    assert_response_contains,
    assert_response_json,
    assert_response_quality
)
from kerb.testing.performance import measure_latency
from kerb.testing.snapshots import SnapshotManager
from pathlib import Path


class MockRetriever:
    """Mock retriever for testing RAG systems."""
    
    def __init__(self, documents: Dict[str, str]):
        self.documents = documents
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant documents."""

# %%
# Setup and Imports
# -----------------
        # Simple keyword-based retrieval for demo
        results = []
        query_words = set(query.lower().split())
        
        for doc_id, content in self.documents.items():
            doc_words = set(content.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                results.append((doc_id, content, overlap))
        
        # Sort by relevance and return top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return [content for _, content, _ in results[:top_k]]


class RAGPipeline:
    """Simple RAG pipeline for testing."""
    
    def __init__(self, retriever: MockRetriever, llm: MockLLM):
        self.retriever = retriever
        self.llm = llm
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        # Step 1: Retrieve relevant documents
        docs = self.retriever.retrieve(question, top_k=2)
        
        # Step 2: Build prompt with context
        context = "\n\n".join(docs)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Step 3: Generate response
        response = self.llm.generate(prompt)
        
        return {
            "question": question,
            "retrieved_docs": docs,
            "answer": response.content,
            "metadata": {
                "num_docs": len(docs),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens
            }
        }


class ChatbotFlow:
    """Multi-turn chatbot for testing."""
    
    def __init__(self, llm: MockLLM):
        self.llm = llm
        self.conversation_history = []
    
    def chat(self, user_message: str) -> str:
        """Process a chat message."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build prompt from history
        prompt = self._build_prompt()
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        return response.content
    

# %%
#  Build Prompt
# -------------

    def _build_prompt(self) -> str:
        """Build prompt from conversation history."""
        messages = []
        for msg in self.conversation_history:
            role = msg["role"].capitalize()
            messages.append(f"{role}: {msg['content']}")
        return "\n".join(messages)
    
    def reset(self):
        """Reset conversation."""
        self.conversation_history = []



# %%
# Main
# ----

def main():
    """Run integration testing examples."""
    
    print("="*80)
    print("INTEGRATION TESTING EXAMPLE")
    print("="*80)
    
    # Example 1: Testing RAG pipeline
    print("\n1. TESTING RAG PIPELINE")
    print("-"*80)
    
    # Setup mock components
    knowledge_base = {
        "doc1": "Python is a high-level programming language created by Guido van Rossum.",
        "doc2": "Machine learning is a subset of AI that learns from data.",
        "doc3": "Neural networks are computational models inspired by the brain.",
        "doc4": "Python is widely used in data science and machine learning."
    }
    
    retriever = MockRetriever(knowledge_base)
    
    # Mock LLM with pattern-based responses
    rag_llm = MockLLM(
        responses={
            r"Python.*programming": "Python is a programming language known for its simplicity.",
            r"machine learning.*data": "Machine learning enables systems to learn from data.",
            r"neural network": "Neural networks are AI models inspired by biological neurons.",
        },
        behavior=MockBehavior.PATTERN,
        default_response="Based on the context provided, here is the answer.",
        latency=0.05
    )
    
    # Create RAG pipeline
    rag = RAGPipeline(retriever, rag_llm)
    
    # Test the pipeline
    test_questions = [
        "What is Python?",
        "Tell me about machine learning",
        "Explain neural networks"
    ]
    
    print("Testing RAG pipeline:")
    for question in test_questions:
        result = rag.query(question)
        
        print(f"\n  Question: {question}")
        print(f"  Retrieved docs: {result['metadata']['num_docs']}")
        print(f"  Answer: {result['answer'][:60]}...")
        
        # Validate
        assert len(result['retrieved_docs']) > 0, "No docs retrieved"
        assert len(result['answer']) > 0, "Empty answer"
        print(f"  Validation: PASS")
    
    # Example 2: Testing multi-turn conversation
    print("\n2. TESTING MULTI-TURN CONVERSATION")
    print("-"*80)
    
    # Setup chatbot
    chat_llm = MockLLM(
        responses=[
            "Hello! How can I help you today?",
            "Python is a programming language.",
            "You're welcome! Let me know if you need anything else."
        ],
        behavior=MockBehavior.SEQUENTIAL,
        latency=0.02
    )
    
    chatbot = ChatbotFlow(chat_llm)
    
    # Test conversation flow
    conversation = [
        "Hi there!",
        "What is Python?",
        "Thank you!"
    ]
    
    print("Testing conversation flow:")
    for user_msg in conversation:
        response = chatbot.chat(user_msg)
        print(f"\n  User: {user_msg}")
        print(f"  Bot: {response}")
        
        # Validate response
        assert len(response) > 0, "Empty response"
        assert_response_quality(response, min_words=3)
    
    print(f"\n  Conversation length: {len(chatbot.conversation_history)} messages")
    print(f"  Validation: PASS")
    
    # Example 3: Testing with dataset
    print("\n3. INTEGRATION TEST WITH DATASET")
    print("-"*80)
    
    # Create test dataset
    test_dataset = TestDataset(name="integration_tests")
    test_dataset.add_example(
        input="What is AI?",
        output="artificial intelligence",
        metadata={"category": "qa"}
    )
    test_dataset.add_example(
        input="Explain ML",
        output="machine learning",
        metadata={"category": "qa"}
    )
    
    # Test LLM
    test_llm = MockLLM(
        responses={
            r"AI": "AI stands for artificial intelligence.",
            r"ML": "ML stands for machine learning.",
        },
        behavior=MockBehavior.PATTERN,
        latency=0.01
    )
    
    print("Running integration tests from dataset:")
    passed = 0
    failed = 0
    
    for i, example in enumerate(test_dataset):
        response = test_llm.generate(example['input']).content
        
        # Check if response contains expected output
        expected_in_response = example['output'].lower() in response.lower()
        
        if expected_in_response:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
        
        print(f"\n  Test {i+1}: {example['input']}")
        print(f"    Expected term: {example['output']}")
        print(f"    Status: {status}")
    
    print(f"\n  Results: {passed} passed, {failed} failed")
    
    # Example 4: Performance and quality combined
    print("\n4. PERFORMANCE + QUALITY TESTING")
    print("-"*80)
    
    perf_llm = MockLLM(
        responses="This is a high-quality response with good content and appropriate length.",
        behavior=MockBehavior.FIXED,
        latency=0.03
    )
    
    def llm_call(prompt):
        return perf_llm.generate(prompt).content
    
    # Measure performance
    print("Measuring performance...")
    metrics = measure_latency(
        fn=llm_call,
        args=["Test prompt"],
        num_runs=10,
        warmup=2
    )
    
    print(f"  Avg latency: {metrics.avg_latency:.3f}s")
    print(f"  P95 latency: {metrics.p95_latency:.3f}s")
    
    # Check quality
    print("\nChecking quality...")
    response = llm_call("Test prompt")
    
    try:
        assert_response_quality(response, min_words=5)
        assert_response_contains(response, "response")
        print("  Quality checks: PASS")
    except AssertionError as e:
        print(f"  Quality checks: FAIL - {e}")
    
    # Check performance SLA
    sla_met = metrics.p95_latency < 0.100  # 100ms SLA
    print(f"  SLA compliance: {'PASS' if sla_met else 'FAIL'}")
    
    # Example 5: Snapshot-based integration testing
    print("\n5. SNAPSHOT-BASED INTEGRATION TESTING")
    print("-"*80)
    
    snapshot_dir = Path("temp_integration_snapshots")
    snapshot_manager = SnapshotManager(snapshot_dir)
    
    # Create baseline
    baseline_llm = MockLLM(
        responses="Baseline integrated response for testing",
        behavior=MockBehavior.FIXED
    )
    
    baseline_result = rag.llm.generate("Test query").content
    snapshot_manager.create_snapshot(
        name="rag_baseline",
        content=baseline_result,
        metadata={"pipeline": "RAG", "version": "1.0"}
    )
    
    print("Created baseline snapshot")
    
    # Test against snapshot
    current_result = rag.llm.generate("Test query").content
    matches, diff = snapshot_manager.compare_snapshot("rag_baseline", current_result)
    
    print(f"Snapshot comparison: {'MATCH' if matches else 'CHANGED'}")
    
    # Example 6: Comprehensive test suite
    print("\n6. COMPREHENSIVE INTEGRATION TEST SUITE")
    print("-"*80)
    
    class IntegrationTestSuite:
        """Integration test suite."""
        

# %%
#   Init  
# --------

        def __init__(self, name: str):
            self.name = name
            self.tests = []
            self.results = []
        

# %%
# Add Test
# --------

        def add_test(self, test_name: str, test_func):
            """Add a test to the suite."""
            self.tests.append((test_name, test_func))
        
        def run(self):
            """Run all tests."""
            print(f"\nRunning test suite: {self.name}")
            print("-" * 60)
            
            for test_name, test_func in self.tests:
                try:
                    test_func()
                    self.results.append((test_name, "PASS", None))
                    print(f"  {test_name}: PASS")
                except Exception as e:
                    self.results.append((test_name, "FAIL", str(e)))
                    print(f"  {test_name}: FAIL - {e}")
        

# %%
# Summary
# -------

        def summary(self):
            """Print test summary."""
            passed = sum(1 for _, status, _ in self.results if status == "PASS")
            failed = sum(1 for _, status, _ in self.results if status == "FAIL")
            
            print(f"\n  Summary: {passed} passed, {failed} failed")
            return passed, failed
    
    # Create test suite
    suite = IntegrationTestSuite("LLM Application Tests")
    
    # Add tests
    def test_rag_retrieval():
        assert len(retriever.retrieve("Python")) > 0
    
    def test_llm_response():
        assert len(rag_llm.generate("test").content) > 0
    
    def test_response_quality():
        assert_response_quality(
            rag_llm.generate("test").content,
            min_words=1
        )
    
    def test_chatbot_history():
        assert len(chatbot.conversation_history) > 0
    
    suite.add_test("RAG retrieval works", test_rag_retrieval)
    suite.add_test("LLM generates response", test_llm_response)
    suite.add_test("Response quality check", test_response_quality)
    suite.add_test("Chatbot maintains history", test_chatbot_history)
    
    # Run suite
    suite.run()
    passed, failed = suite.summary()
    
    # Example 7: Error handling testing
    print("\n7. ERROR HANDLING TESTING")
    print("-"*80)
    
    error_llm = MockLLM(
        responses="",  # Empty response to trigger error handling
        behavior=MockBehavior.FIXED
    )
    
    def safe_pipeline_call(pipeline, question):
        """Safely call pipeline with error handling."""
        try:
            result = pipeline.query(question)
            if not result['answer']:
                return {"error": "Empty response", "fallback": "Default answer"}
            return result
        except Exception as e:
            return {"error": str(e), "fallback": "Error occurred"}
    
    print("Testing error handling:")
    
    # Create pipeline with error-prone LLM
    error_rag = RAGPipeline(retriever, error_llm)
    result = safe_pipeline_call(error_rag, "Test question")
    
    if "error" in result:
        print(f"  Error detected: {result['error']}")
        print(f"  Fallback used: {result.get('fallback')}")
        print("  Error handling: PASS")
    else:
        print("  No errors encountered")
    
    # Example 8: End-to-end validation
    print("\n8. END-TO-END VALIDATION")
    print("-"*80)
    

# %%
# Validate E2E Pipeline
# ---------------------

    def validate_e2e_pipeline(
        input_query: str,
        expected_characteristics: Dict[str, Any]
    ) -> bool:
        """Validate entire pipeline end-to-end."""
        
        # Run through pipeline
        result = rag.query(input_query)
        
        # Validate characteristics
        checks = []
        
        # Check answer exists
        has_answer = len(result['answer']) > 0
        checks.append(("Has answer", has_answer))
        
        # Check retrieval worked
        has_docs = len(result['retrieved_docs']) > 0
        checks.append(("Retrieved docs", has_docs))
        
        # Check response quality
        min_length = expected_characteristics.get('min_length', 10)
        meets_length = len(result['answer']) >= min_length
        checks.append(("Meets length", meets_length))
        
        # Print results
        for check_name, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"    {check_name}: {status}")
        
        return all(passed for _, passed in checks)
    
    print("End-to-end validation:")
    e2e_passed = validate_e2e_pipeline(
        "What is Python?",
        {"min_length": 10}
    )
    
    print(f"\n  Overall E2E: {'PASS' if e2e_passed else 'FAIL'}")
    
    # Cleanup
    print("\n9. CLEANUP")
    print("-"*80)
    import shutil
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
        print("Cleaned up snapshot directory")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)
    print("\nKey takeaways:")
    print("- Integration tests validate complete workflows")
    print("- Combine multiple testing techniques")
    print("- Test real-world scenarios end-to-end")
    print("- Validate both functionality and performance")
    print("- Include error handling in tests")
    print("- Create comprehensive test suites")


if __name__ == "__main__":
    main()
