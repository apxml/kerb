"""
ReAct Agent Example
===================

This example demonstrates the ReAct (Reasoning and Acting) pattern.

The ReAct pattern follows: Thought -> Action -> Observation -> Thought -> ...

Main concepts:
- Creating a ReActAgent
- Understanding the thought-action-observation loop
- Using tools within the ReAct pattern
- Extracting final answers
"""

from kerb.agent.patterns import ReActAgent
from kerb.agent import Tool


def mock_llm_react(prompt: str) -> str:
    """Mock LLM that responds in ReAct format.
    
    In production, you would use a real LLM with ReAct prompting.
    """

# %%
# Setup and Imports
# -----------------
    # Detect which phase we're in based on prompt
    if "what is 15 * 7" in prompt.lower() or "calculate" in prompt.lower():
        return """Thought: I need to calculate 15 multiplied by 7.
Action: calculate
Action Input: 15 * 7"""
    
    elif "105" in prompt:
        return """Thought: I have the calculation result. The answer is 105.
Final Answer: 15 * 7 = 105"""
    
    else:
        return """Thought: Let me think about this step by step.
Action: proceed
Action Input: continue"""



# %%
# Calculate
# ---------

def calculate(expression: str) -> str:
    """Calculator tool."""
    try:
        # Safe evaluation for basic math
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def search(query: str) -> str:
    """Mock search tool."""
    responses = {
        "python": "Python is a high-level programming language.",
        "react": "ReAct is a reasoning pattern for LLM agents.",
        "weather": "The weather is sunny and 72°F."
    }
    
    for key, value in responses.items():
        if key in query.lower():
            return value
    
    return f"No results found for: {query}"



# %%
# Main
# ----

def main():
    """Run ReAct agent example."""
    
    print("="*80)
    print("REACT AGENT EXAMPLE")
    print("="*80)
    
    # Create tools
    calc_tool = Tool(
        name="calculate",
        description="Performs mathematical calculations",
        func=calculate,
        parameters={"expression": {"type": "string", "description": "Math expression"}}
    )
    
    search_tool = Tool(
        name="search",
        description="Searches for information",
        func=search,
        parameters={"query": {"type": "string", "description": "Search query"}}
    )
    
    # Create ReAct agent
    agent = ReActAgent(
        name="MathAgent",
        llm_func=mock_llm_react,
        tools=[calc_tool, search_tool],
        max_iterations=5
    )
    
    print(f"\nCreated ReAct agent: {agent.name}")
    print(f"📦 Tools available: {', '.join([t.name for t in agent.tools])}")
    
    # Run the agent
    goal = "What is 15 * 7?"
    print(f"\n🎯 Goal: {goal}")
    print("\n" + "-"*80)
    
    result = agent.run(goal)
    
    # Display the ReAct loop
    print("\nREACT LOOP")
    print("-"*80)
    for i, step in enumerate(result.steps, 1):
        print(f"\n[Step {i}]")
        if step.thought:
            print(f"Thought: {step.thought}")
        if step.action:
            print(f"Action: {step.action}({step.action_input})")
        if step.observation:
            print(f"Observation: {step.observation}")
    
    # Display final result
    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    print(f"Status: {result.status.value}")
    print(f"Total Time: {result.total_time:.4f}s")
    print(f"Total Steps: {len(result.steps)}")
    print(f"\nAnswer: {result.output}")
    
    print("\n" + "="*80)
    print("ReAct example completed!")
    print("="*80)


if __name__ == "__main__":
    main()
