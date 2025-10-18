"""Agent monitoring, evaluation, and debugging.

This module provides tools for tracing agent execution, evaluating performance,
benchmarking, and comparing different agents.
"""

import json
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .core import Agent, AgentResult


# ============================================================================
# Monitoring and Tracing
# ============================================================================

@dataclass
class AgentTracer:
    """Trace agent execution for monitoring."""
    traces: List[Dict[str, Any]] = field(default_factory=list)
    
    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a trace event."""
        self.traces.append({
            'event': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        })
    
    def get_traces(self, event_type: str = None) -> List[Dict[str, Any]]:
        """Get traces, optionally filtered by type."""
        if event_type is None:
            return self.traces
        return [t for t in self.traces if t['event'] == event_type]
    
    def clear(self) -> None:
        """Clear all traces."""
        self.traces.clear()
    
    def visualize(self) -> str:
        """Visualize execution trace.
        
        Returns:
            Formatted trace visualization
        """
        lines = ["=== Agent Execution Trace ===\n"]
        
        for i, trace in enumerate(self.traces):
            lines.append(f"{i+1}. [{trace['event']}] {trace['timestamp']}")
            lines.append(f"   {json.dumps(trace['data'], indent=2)}\n")
        
        return "\n".join(lines)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_agent(
    agent: Agent,
    test_cases: List[Tuple[str, str]],
    metric_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """Evaluate agent on test cases.
    
    Args:
        agent: Agent to evaluate
        test_cases: List of (goal, expected_output) tuples
        metric_func: Function to compute metrics
        
    Returns:
        Evaluation results
    """
    results = []
    
    for goal, expected in test_cases:
        result = agent.run(goal)
        
        if metric_func:
            score = metric_func(result.output, expected)
        else:
            # Simple exact match
            score = 1.0 if result.output == expected else 0.0
        
        results.append({
            'goal': goal,
            'output': result.output,
            'expected': expected,
            'score': score,
            'steps': len(result.steps),
            'time': result.total_time
        })
    
    avg_score = sum(r['score'] for r in results) / len(results) if results else 0
    
    return {
        'average_score': avg_score,
        'num_test_cases': len(test_cases),
        'results': results
    }


def benchmark_agent(
    agent: Agent,
    tasks: List[str],
    num_runs: int = 3
) -> Dict[str, Any]:
    """Benchmark agent performance.
    
    Args:
        agent: Agent to benchmark
        tasks: List of tasks
        num_runs: Number of runs per task
        
    Returns:
        Benchmark results
    """
    task_results = []
    
    for task in tasks:
        times = []
        step_counts = []
        
        for _ in range(num_runs):
            result = agent.run(task)
            times.append(result.total_time)
            step_counts.append(len(result.steps))
        
        task_results.append({
            'task': task,
            'avg_time': sum(times) / len(times),
            'avg_steps': sum(step_counts) / len(step_counts),
            'min_time': min(times),
            'max_time': max(times)
        })
    
    return {
        'num_tasks': len(tasks),
        'num_runs_per_task': num_runs,
        'task_results': task_results
    }


def compare_agents(
    agents: List[Agent],
    test_cases: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """Compare multiple agents.
    
    Args:
        agents: List of agents to compare
        test_cases: List of (goal, expected) tuples
        
    Returns:
        Comparison results
    """
    agent_results = {}
    
    for agent in agents:
        eval_result = evaluate_agent(agent, test_cases)
        agent_results[agent.name] = eval_result
    
    # Find best agent
    best_agent = max(
        agent_results.items(),
        key=lambda x: x[1]['average_score']
    )
    
    return {
        'agent_results': agent_results,
        'best_agent': best_agent[0],
        'best_score': best_agent[1]['average_score']
    }
