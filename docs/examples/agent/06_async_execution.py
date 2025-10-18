"""Async Agent Execution Example

This example demonstrates asynchronous agent execution patterns.

Main concepts:
- Running agents asynchronously
- Concurrent execution of multiple agents
- Streaming agent steps
- Async/await patterns with agents
- Performance benefits of async execution
"""

import asyncio
import time
from kerb.agent.patterns import ReActAgent as Agent
from kerb.agent.execution import AgentExecutor, run_agent_loop, run_agent_async


def mock_llm_slow(prompt: str) -> str:
    """Mock LLM with simulated delay."""
    time.sleep(0.5)  # Simulate API latency
    
    if "weather" in prompt.lower():
        return "The weather is sunny."
    elif "stock" in prompt.lower():
        return "The stock price is $150."
    elif "news" in prompt.lower():
        return "Latest news: Tech stocks rising."
    else:
        return f"Processed: {prompt[:50]}"


async def async_llm(prompt: str) -> str:
    """Async mock LLM."""
    await asyncio.sleep(0.5)  # Simulate async API call
    
    if "analyze" in prompt.lower():
        return "Analysis complete: Data shows positive trends."
    elif "summarize" in prompt.lower():
        return "Summary: Key points extracted successfully."
    else:
        return f"Async result for: {prompt[:50]}"


async def main():
    """Run async agent example."""
    
    print("="*80)
    print("ASYNC AGENT EXECUTION EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # SYNC vs ASYNC COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("1. SYNC vs ASYNC COMPARISON")
    print("="*80)
    
    # Create agents
    agent1 = Agent(name="SyncAgent1", llm_func=mock_llm_slow, max_iterations=2)
    agent2 = Agent(name="SyncAgent2", llm_func=mock_llm_slow, max_iterations=2)
    agent3 = Agent(name="SyncAgent3", llm_func=mock_llm_slow, max_iterations=2)
    
    # Synchronous execution
    print("\nSYNCHRONOUS EXECUTION (Sequential)")
    print("-"*80)
    
    start_sync = time.time()
    
    result1 = agent1.run("What's the weather?")
    result2 = agent2.run("What's the stock price?")
    result3 = agent3.run("What's the latest news?")
    
    sync_time = time.time() - start_sync
    
    print(f"Completed 3 agents")
    print(f"Time taken: {sync_time:.2f}s")
    
    # Asynchronous execution
    print("\nASYNCHRONOUS EXECUTION (Concurrent)")
    print("-"*80)
    
    start_async = time.time()
    
    # Create executors
    executor1 = AgentExecutor(agent1)
    executor2 = AgentExecutor(agent2)
    executor3 = AgentExecutor(agent3)
    
    # Run concurrently
    results = await asyncio.gather(
        executor1.run_async("What's the weather?"),
        executor2.run_async("What's the stock price?"),
        executor3.run_async("What's the latest news?")
    )
    
    async_time = time.time() - start_async
    
    print(f"Completed 3 agents concurrently")
    print(f"Time taken: {async_time:.2f}s")
    print(f"Speedup: {sync_time/async_time:.2f}x faster")
    
    # ========================================================================
    # ASYNC AGENT WITH ASYNC LLM
    # ========================================================================
    print("\n" + "="*80)
    print("2. ASYNC AGENT WITH ASYNC LLM")
    print("="*80)
    
    # Note: This demonstrates the pattern, though the Agent class needs async support
    async def run_with_async_llm(goal: str) -> dict:
        """Simulate async agent execution."""
        print(f"\n▶Processing: {goal}")
        
        result = await async_llm(goal)
        
        return {
            'goal': goal,
            'output': result,
            'status': 'completed'
        }
    
    # Run multiple async operations
    async_goals = [
        "Analyze the quarterly data",
        "Summarize the report",
        "Analyze customer feedback"
    ]
    
    print("\nRunning multiple async operations:")
    async_results = await asyncio.gather(*[
        run_with_async_llm(goal) for goal in async_goals
    ])
    
    print("\n📊 Results:")
    print("-"*80)
    for result in async_results:
        print(f"\n{result['goal']}:")
        print(f"  Status: {result['status']}")
        print(f"  Output: {result['output']}")
    
    # ========================================================================
    # STREAMING EXECUTION
    # ========================================================================
    print("\n" + "="*80)
    print("3. STREAMING EXECUTION")
    print("="*80)
    
    stream_agent = Agent(
        name="StreamAgent",
        llm_func=mock_llm_slow,
        max_iterations=3
    )
    
    executor = AgentExecutor(stream_agent)
    
    print("\n📡 Streaming agent steps as they execute...")
    print("-"*80)
    
    step_count = 0
    async for step in executor.run_stream("Analyze the data"):
        step_count += 1
        print(f"\n[Step {step_count}]")
        if step.thought:
            print(f"  Thought: {step.thought}")
        if step.action:
            print(f"  Action: {step.action}")
        if step.observation:
            print(f"  Observation: {step.observation}")
    
    print(f"\nStreamed {step_count} steps")
    
    # ========================================================================
    # CONCURRENT AGENT TEAMS
    # ========================================================================
    print("\n" + "="*80)
    print("4. CONCURRENT AGENT TEAMS")
    print("="*80)
    
    # Create specialized agents
    researcher = Agent(name="Researcher", llm_func=mock_llm_slow, max_iterations=2)
    analyst = Agent(name="Analyst", llm_func=mock_llm_slow, max_iterations=2)
    writer = Agent(name="Writer", llm_func=mock_llm_slow, max_iterations=2)
    
    # Create executors
    researchers_executor = AgentExecutor(researcher)
    analyst_executor = AgentExecutor(analyst)
    writer_executor = AgentExecutor(writer)
    
    print("\nRunning agent team concurrently...")
    print("-"*80)
    
    team_start = time.time()
    
    team_results = await asyncio.gather(
        researchers_executor.run_async("Research Python trends"),
        analyst_executor.run_async("Analyze the data"),
        writer_executor.run_async("Write a summary"),
        return_exceptions=True  # Don't fail if one agent fails
    )
    
    team_time = time.time() - team_start
    
    print(f"\nTeam completed in {team_time:.2f}s")
    
    agents_list = [researcher, analyst, writer]
    for i, result in enumerate(team_results):
        if isinstance(result, Exception):
            print(f"\n{agents_list[i].name}: Error - {result}")
        else:
            print(f"\n{agents_list[i].name}:")
            print(f"   Status: {result.status.value}")
            print(f"   Output: {result.output[:60]}...")
    
    # ========================================================================
    # ERROR HANDLING IN ASYNC
    # ========================================================================
    print("\n" + "="*80)
    print("5. ASYNC ERROR HANDLING")
    print("="*80)
    
    def error_llm(prompt: str) -> str:
        """LLM that sometimes errors."""
        if "error" in prompt.lower():
            raise ValueError("Simulated LLM error")
        return "Success"
    
    error_agent = Agent(name="ErrorAgent", llm_func=error_llm, max_iterations=2)
    error_executor = AgentExecutor(error_agent)
    
    print("\nTesting error handling...")
    
    # Test successful execution
    try:
        success_result = await error_executor.run_async("Process this")
        print(f"Success case: {success_result.status.value}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Test error case
    try:
        error_result = await error_executor.run_async("This will error")
        print(f"Completed: {error_result.status.value}")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}")
    
    # ========================================================================
    # TIMEOUT HANDLING
    # ========================================================================
    print("\n" + "="*80)
    print("6. TIMEOUT HANDLING")
    print("="*80)
    
    def slow_llm(prompt: str) -> str:
        """Very slow LLM."""
        time.sleep(3)
        return "Finally done"
    
    slow_agent = Agent(name="SlowAgent", llm_func=slow_llm, max_iterations=1)
    slow_executor = AgentExecutor(slow_agent)
    
    print("\nTesting timeout (2 second limit)...")
    
    try:
        result = await asyncio.wait_for(
            slow_executor.run_async("Slow task"),
            timeout=2.0
        )
        print(f"Completed: {result.output}")
    except asyncio.TimeoutError:
        print("Task timed out as expected")
    
    print("\n" + "="*80)
    print("Async execution example completed!")
    print("="*80)
    
    # Summary
    print("\nImportant concepts demonstrated:")
    print("-"*80)
    print("1. Async execution allows concurrent agent execution")
    print("2. Can achieve significant speedup with parallel agents")
    print("3. AgentExecutor provides async execution methods")
    print("4. Streaming allows real-time step monitoring")
    print("5. Error handling and timeouts work naturally with async/await")
    print("6. Ideal for multi-agent systems and I/O-bound operations")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
