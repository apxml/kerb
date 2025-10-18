"""Agent Monitoring and Evaluation Example

This example demonstrates how to monitor and evaluate agent performance.

Concepts covered:
- Agent tracing and logging
- Performance metrics
- Agent evaluation
- Benchmarking agents
- Comparing multiple agents
- Debugging agent behavior
"""

from kerb.agent.patterns import ReActAgent
from kerb.agent.monitoring import AgentTracer, evaluate_agent, benchmark_agent
from kerb.agent import AgentResult
from kerb.agent.core import AgentStep, AgentStatus
import time
from typing import List, Dict, Any


def fast_llm(prompt: str) -> str:
    """Fast mock LLM."""
    time.sleep(0.1)
    return "Quick response"


def slow_llm(prompt: str) -> str:
    """Slow mock LLM."""
    time.sleep(0.3)
    return "Detailed response"


def accurate_llm(prompt: str) -> str:
    """Accurate mock LLM."""
    if "calculate" in prompt.lower():
        return "The answer is 42"
    return "Accurate response based on prompt"


def main():
    """Run agent monitoring example."""
    
    print("="*80)
    print("AGENT MONITORING AND EVALUATION EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # AGENT TRACING
    # ========================================================================
    print("\n" + "="*80)
    print("1. AGENT TRACING")
    print("="*80)
    
    tracer = AgentTracer()
    
    # Create agent
    traced_agent = ReActAgent(
        name="TracedAgent",
        llm_func=fast_llm,
        max_iterations=3
    )
    
    print(f"\nTracing agent: {traced_agent.name}")
    
    # Start tracing
    tracer.log('agent_start', {'agent_name': traced_agent.name})
    
    # Run agent
    print("\nRunning agent...")
    result = traced_agent.run("Solve this problem")
    
    # Record trace data
    tracer.log('agent_step', {
        'agent_name': traced_agent.name,
        'step': 1,
        'action': 'process',
        'duration': result.total_time,
        'status': result.status.value
    })
    
    tracer.log('agent_end', {'agent_name': traced_agent.name, 'total_duration': result.total_time})
    
    # Get trace
    traces = tracer.get_traces()
    
    print(f"\nTrace Summary:")
    print("-"*80)
    print(f"Events logged: {len(traces)}")
    for trace in traces:
        print(f"  [{trace['event']}] {trace['timestamp']}")
        print(f"     {trace['data']}")
    
    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================
    print("\n" + "="*80)
    print("2. PERFORMANCE METRICS")
    print("="*80)
    
    def calculate_metrics(result: AgentResult) -> Dict[str, Any]:
        """Calculate performance metrics."""
        return {
            'total_time': result.total_time,
            'steps_count': len(result.steps),
            'avg_step_time': result.total_time / len(result.steps) if result.steps else 0,
            'success_rate': 1.0 if result.status == AgentStatus.COMPLETED else 0.0,
            'output_length': len(result.output)
        }
    
    # Run agent and calculate metrics
    agent = ReActAgent(name="MetricsAgent", llm_func=fast_llm, max_iterations=5)
    result = agent.run("Analyze this data")
    
    metrics = calculate_metrics(result)
    
    print(f"\nPerformance Metrics:")
    print("-"*80)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value}")
    
    # ========================================================================
    # AGENT EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("3. AGENT EVALUATION")
    print("="*80)
    
    # Create evaluation criteria
    eval_criteria = {
        'accuracy': lambda r: 1.0 if 'correct' in r.output.lower() else 0.5,
        'speed': lambda r: 1.0 if r.total_time < 1.0 else 0.5,
        'completeness': lambda r: len(r.steps) / 5.0  # Normalized to max 5 steps
    }
    
    eval_agent = ReActAgent(name="EvalAgent", llm_func=accurate_llm, max_iterations=3)
    
    print(f"\nEvaluating: {eval_agent.name}")
    print(f"   Criteria: {list(eval_criteria.keys())}")
    
    # Run and evaluate
    eval_result = eval_agent.run("Calculate the answer")
    
    scores = {}
    for criterion, score_func in eval_criteria.items():
        scores[criterion] = score_func(eval_result)
    
    avg_score = sum(scores.values()) / len(scores)
    
    print(f"\nEvaluation Scores:")
    print("-"*80)
    for criterion, score in scores.items():
        print(f"   {criterion}: {score:.2f}")
    print(f"\n   Average Score: {avg_score:.2f}")
    
    # ========================================================================
    # BENCHMARKING
    # ========================================================================
    print("\n" + "="*80)
    print("4. BENCHMARKING")
    print("="*80)
    
    # Create test cases
    test_cases = [
        "Simple task 1",
        "Simple task 2",
        "Simple task 3"
    ]
    
    bench_agent = ReActAgent(name="BenchAgent", llm_func=fast_llm, max_iterations=2)
    
    print(f"\nBenchmarking: {bench_agent.name}")
    print(f"   Test cases: {len(test_cases)}")
    
    # Run benchmark
    benchmark_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        start = time.time()
        result = bench_agent.run(test_case)
        duration = time.time() - start
        
        benchmark_results.append({
            'test_case': test_case,
            'duration': duration,
            'steps': len(result.steps),
            'status': result.status.value
        })
        
        print(f"   Test {i}: {duration:.4f}s")
    
    # Calculate statistics
    avg_duration = sum(r['duration'] for r in benchmark_results) / len(benchmark_results)
    min_duration = min(r['duration'] for r in benchmark_results)
    max_duration = max(r['duration'] for r in benchmark_results)
    
    print(f"\nBenchmark Statistics:")
    print("-"*80)
    print(f"   Average time: {avg_duration:.4f}s")
    print(f"   Min time: {min_duration:.4f}s")
    print(f"   Max time: {max_duration:.4f}s")
    print(f"   Total tests: {len(benchmark_results)}")
    
    # ========================================================================
    # COMPARING AGENTS
    # ========================================================================
    print("\n" + "="*80)
    print("5. COMPARING AGENTS")
    print("="*80)
    
    # Create agents with different characteristics
    agent_fast = ReActAgent(name="FastAgent", llm_func=fast_llm, max_iterations=3)
    agent_slow = ReActAgent(name="SlowAgent", llm_func=slow_llm, max_iterations=3)
    agent_accurate = ReActAgent(name="AccurateAgent", llm_func=accurate_llm, max_iterations=3)
    
    agents = [agent_fast, agent_slow, agent_accurate]
    
    print(f"\nComparing {len(agents)} agents:")
    for agent in agents:
        print(f"   - {agent.name}")
    
    # Test all agents on same task
    test_goal = "Process this request"
    comparison_results = []
    
    print(f"\n📝 Test goal: {test_goal}")
    print("-"*80)
    
    for agent in agents:
        result = agent.run(test_goal)
        comparison_results.append({
            'name': agent.name,
            'time': result.total_time,
            'steps': len(result.steps),
            'status': result.status.value,
            'output_len': len(result.output)
        })
        
        print(f"\n{agent.name}:")
        print(f"   Time: {result.total_time:.4f}s")
        print(f"   Steps: {len(result.steps)}")
        print(f"   Status: {result.status.value}")
    
    # Find best agent
    fastest = min(comparison_results, key=lambda x: x['time'])
    most_steps = max(comparison_results, key=lambda x: x['steps'])
    
    print(f"\n🏆 Comparison Results:")
    print("-"*80)
    print(f"   Fastest: {fastest['name']} ({fastest['time']:.4f}s)")
    print(f"   Most thorough: {most_steps['name']} ({most_steps['steps']} steps)")
    
    # ========================================================================
    # ERROR MONITORING
    # ========================================================================
    print("\n" + "="*80)
    print("6. ERROR MONITORING")
    print("="*80)
    
    error_log = []
    
    def error_llm(prompt: str) -> str:
        """LLM that sometimes fails."""
        import random
        if random.random() < 0.3:  # 30% failure rate
            raise ValueError("Simulated LLM error")
        return "Success"
    
    error_agent = ReActAgent(name="ErrorAgent", llm_func=error_llm, max_iterations=5)
    
    print(f"\nTesting error monitoring...")
    print("-"*80)
    
    # Run multiple times to catch errors
    attempts = 10
    successes = 0
    failures = 0
    
    for i in range(attempts):
        try:
            result = error_agent.run(f"Task {i}")
            successes += 1
        except Exception as e:
            failures += 1
            error_log.append({
                'attempt': i,
                'error': str(e),
                'type': type(e).__name__
            })
    
    print(f"\nError Monitoring Results:")
    print(f"   Total attempts: {attempts}")
    print(f"   Successes: {successes}")
    print(f"   Failures: {failures}")
    print(f"   Success rate: {(successes/attempts)*100:.1f}%")
    
    if error_log:
        print(f"\nErrors logged: {len(error_log)}")
        for error in error_log[:3]:  # Show first 3
            print(f"   Attempt {error['attempt']}: {error['type']}")
    
    # ========================================================================
    # EXECUTION SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("7. EXECUTION SUMMARY")
    print("="*80)
    
    class AgentMonitor:
        """Monitor for tracking agent executions."""
        
        def __init__(self):
            self.executions = []
        
        def record_execution(self, agent_name: str, result: AgentResult):
            """Record an execution."""
            self.executions.append({
                'agent': agent_name,
                'status': result.status.value,
                'time': result.total_time,
                'steps': len(result.steps),
                'timestamp': time.time()
            })
        
        def get_summary(self) -> Dict[str, Any]:
            """Get execution summary."""
            if not self.executions:
                return {'total': 0}
            
            return {
                'total': len(self.executions),
                'avg_time': sum(e['time'] for e in self.executions) / len(self.executions),
                'avg_steps': sum(e['steps'] for e in self.executions) / len(self.executions),
                'success_rate': sum(1 for e in self.executions 
                                   if e['status'] == 'completed') / len(self.executions)
            }
    
    monitor = AgentMonitor()
    
    # Simulate multiple executions
    test_agent = ReActAgent(name="MonitoredAgent", llm_func=fast_llm, max_iterations=3)
    
    for i in range(5):
        result = test_agent.run(f"Task {i}")
        monitor.record_execution(test_agent.name, result)
    
    summary = monitor.get_summary()
    
    print(f"\nExecution Summary:")
    print("-"*80)
    print(f"   Total executions: {summary['total']}")
    print(f"   Average time: {summary['avg_time']:.4f}s")
    print(f"   Average steps: {summary['avg_steps']:.2f}")
    print(f"   Success rate: {summary['success_rate']*100:.1f}%")
    
    print("\n" + "="*80)
    print("Monitoring and evaluation example completed!")
    print("="*80)
    
    # Summary
    print("\nImportant concepts demonstrated:")
    print("-"*80)
    print("1. AgentTracer tracks execution flow and timing")
    print("2. Performance metrics provide quantitative evaluation")
    print("3. Custom evaluation criteria assess agent quality")
    print("4. Benchmarking measures performance across test cases")
    print("5. Comparing agents helps select the best for a task")
    print("6. Error monitoring tracks failures and success rates")
    print("7. Execution summaries provide aggregate insights")


if __name__ == "__main__":
    main()
