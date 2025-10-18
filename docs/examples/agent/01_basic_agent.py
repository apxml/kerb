"""Basic Agent Example

This example demonstrates how to create and run a simple agent.

Main concepts:
- Creating a ReActAgent instance
- Setting up a simple LLM function
- Running an agent with a goal
- Inspecting agent results
"""

from kerb.agent.patterns import ReActAgent
from kerb.agent import AgentResult


def simple_llm(prompt: str) -> str:
    """A simple mock LLM function for demonstration.
    
    In production, you would use a real LLM like OpenAI, Anthropic, etc.
    """
    # Mock responses based on prompt content
    if "weather" in prompt.lower():
        return "Thought: I should provide weather information.\nFinal Answer: The weather is sunny with a temperature of 72°F."
    elif "calculate" in prompt.lower() or "math" in prompt.lower():
        return "Thought: I need to provide the calculation result.\nFinal Answer: The result is 42."
    else:
        return f"Thought: I'll process this request.\nFinal Answer: I have processed your request: {prompt[:50]}..."


def main():
    """Run basic agent example."""
    
    print("="*80)
    print("BASIC AGENT EXAMPLE")
    print("="*80)
    
    # Create a ReAct agent (concrete implementation of Agent)
    agent = ReActAgent(
        name="BasicAgent",
        llm_func=simple_llm,
        max_iterations=5
    )
    
    print(f"\nCreated agent: {agent.name}")
    
    # Run the agent with a goal
    goal = "What is the weather like today?"
    print(f"\nGoal: {goal}")
    print("\n" + "-"*80)
    
    result = agent.run(goal)
    
    # Display results
    print("\nRESULTS")
    print("-"*80)
    print(f"Status: {result.status.value}")
    print(f"Total Time: {result.total_time:.4f}s")
    print(f"Steps Taken: {len(result.steps)}")
    print(f"\nOutput:\n{result.output}")
    
    # Show execution steps
    if result.steps:
        print("\nEXECUTION STEPS")
        print("-"*80)
        for i, step in enumerate(result.steps, 1):
            print(f"\nStep {i}:")
            if step.thought:
                print(f"  Thought: {step.thought}")
            if step.action:
                print(f"  Action: {step.action}")
            if step.observation:
                print(f"  Observation: {step.observation}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

