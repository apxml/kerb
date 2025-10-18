"""Multi-Agent Teams Example

This example demonstrates multi-agent coordination and teamwork.

Main concepts:
- Creating agent teams
- Running agents in parallel
- Running agents sequentially
- Task delegation between agents
- Aggregating results from multiple agents
- Agent conversations
"""

from kerb.agent.patterns import ReActAgent as Agent
from kerb.agent.teams import AgentTeam, Conversation, delegate_task, aggregate_results


def researcher_llm(prompt: str) -> str:
    """LLM function for researcher agent."""
    if "research" in prompt.lower() or "find" in prompt.lower():
        return "I found the following information: Python is a popular programming language."
    return "Conducting research..."


def writer_llm(prompt: str) -> str:
    """LLM function for writer agent."""
    if "write" in prompt.lower() or "article" in prompt.lower():
        return "I have written a comprehensive article based on the research findings."
    return "Writing content..."


def editor_llm(prompt: str) -> str:
    """LLM function for editor agent."""
    if "edit" in prompt.lower() or "review" in prompt.lower():
        return "I have reviewed and edited the content. It's now polished and ready."
    return "Editing content..."


def analyst_llm(prompt: str) -> str:
    """LLM function for analyst agent."""
    if "analyze" in prompt.lower() or "data" in prompt.lower():
        return "Analysis complete: The data shows a positive trend."
    return "Analyzing data..."


def main():
    """Run multi-agent teams example."""
    
    print("="*80)
    print("MULTI-AGENT TEAMS EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # CREATE AGENT TEAM
    # ========================================================================
    print("\nCreating Agent Team")
    print("-"*80)
    
    # Create individual agents
    researcher = Agent(
        name="Researcher",
        llm_func=researcher_llm,
        max_iterations=3
    )
    
    writer = Agent(
        name="Writer",
        llm_func=writer_llm,
        max_iterations=3
    )
    
    editor = Agent(
        name="Editor",
        llm_func=editor_llm,
        max_iterations=3
    )
    
    analyst = Agent(
        name="Analyst",
        llm_func=analyst_llm,
        max_iterations=3
    )
    
    print("Created agents:")
    for agent in [researcher, writer, editor, analyst]:
        print(f"   - {agent.name}")
    
    # Create team
    team = AgentTeam(agents=[researcher, writer, editor])
    
    print(f"\nCreated team with {len(team.agents)} agents")
    
    # ========================================================================
    # PARALLEL EXECUTION
    # ========================================================================
    print("\n" + "="*80)
    print("1. PARALLEL EXECUTION")
    print("="*80)
    
    goal_parallel = "Research Python programming"
    print(f"\nGoal: {goal_parallel}")
    print("\nRunning all agents in parallel on the same goal...")
    
    parallel_results = team.run_parallel(goal_parallel)
    
    print("\nParallel Results:")
    print("-"*80)
    for i, result in enumerate(parallel_results, 1):
        agent_name = team.agents[i-1].name
        print(f"\n{agent_name}:")
        print(f"  Status: {result.status.value}")
        print(f"  Output: {result.output}")
        print(f"  Steps: {len(result.steps)}")
    
    # ========================================================================
    # SEQUENTIAL EXECUTION (PIPELINE)
    # ========================================================================
    print("\n" + "="*80)
    print("2. SEQUENTIAL EXECUTION (Pipeline)")
    print("="*80)
    
    goal_sequential = "Create an article about Python"
    print(f"\nGoal: {goal_sequential}")
    print("\nRunning agents sequentially (output flows forward)...")
    print("   Pipeline: Researcher → Writer → Editor")
    
    sequential_results = team.run_sequential(goal_sequential)
    
    print("\nSequential Results:")
    print("-"*80)
    for i, result in enumerate(sequential_results, 1):
        agent_name = team.agents[i-1].name
        print(f"\n[Step {i}] {agent_name}:")
        print(f"  Status: {result.status.value}")
        print(f"  Output: {result.output[:100]}...")
        if i < len(sequential_results):
            print(f"  → Passed to next agent")
    
    # ========================================================================
    # TASK DELEGATION
    # ========================================================================
    print("\n" + "="*80)
    print("3. TASK DELEGATION")
    print("="*80)
    
    print(f"\n{researcher.name} delegates to {analyst.name}")
    
    delegated_task = "Analyze the research findings"
    delegation_result = delegate_task(
        task=delegated_task,
        from_agent=researcher,
        to_agent=analyst,
        context={'source': 'research_data'}
    )
    
    print(f"\nDelegation complete:")
    print(f"   Task: {delegated_task}")
    print(f"   From: {researcher.name}")
    print(f"   To: {analyst.name}")
    print(f"   Result: {delegation_result.output}")
    
    # ========================================================================
    # AGENT CONVERSATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("4. AGENT CONVERSATIONS")
    print("="*80)
    
    conversation = Conversation()
    
    # Simulate a conversation
    conversation.add_message(researcher.name, "I've completed the research on Python.")
    conversation.add_message(writer.name, "Great! I'll use that to write the article.")
    conversation.add_message(researcher.name, "Let me know if you need more details.")
    conversation.add_message(writer.name, "Article is ready for review.")
    conversation.add_message(editor.name, "I'll review it now.")
    
    print("\nConversation History:")
    print("-"*80)
    for i, msg in enumerate(conversation.get_history(), 1):
        print(f"\n[{i}] {msg['agent']}:")
        print(f"    {msg['content']}")
        print(f"    {msg['timestamp']}")
    
    # Get recent messages
    recent = conversation.get_history(n=2)
    print(f"\nLast 2 messages:")
    for msg in recent:
        print(f"   {msg['agent']}: {msg['content']}")
    
    # ========================================================================
    # AGGREGATE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("5. AGGREGATE RESULTS")
    print("="*80)
    
    # Create multiple agents for a task
    agents_for_task = [researcher, analyst]
    task_team = AgentTeam(agents=agents_for_task)
    
    task_goal = "Evaluate Python for data science"
    print(f"\nTask: {task_goal}")
    print(f"Agents: {', '.join([a.name for a in agents_for_task])}")
    
    task_results = task_team.run_parallel(task_goal)
    
    print("\nIndividual Results:")
    for i, result in enumerate(task_results):
        print(f"   {agents_for_task[i].name}: {result.output[:60]}...")
    
    # Aggregate results
    aggregated = aggregate_results(task_results)
    
    print("\nAggregated Result:")
    print("-"*80)
    print(f"Status: {aggregated.status.value}")
    print(f"Total Steps: {len(aggregated.steps)}")
    print(f"Combined Output: {aggregated.output[:100]}...")
    
    # ========================================================================
    # DYNAMIC TEAM BUILDING
    # ========================================================================
    print("\n" + "="*80)
    print("6. DYNAMIC TEAM BUILDING")
    print("="*80)
    
    # Create empty team and add agents dynamically
    dynamic_team = AgentTeam()
    
    print("\nBuilding team dynamically:")
    for agent in [researcher, writer, editor]:
        dynamic_team.add_agent(agent)
        print(f"   + Added {agent.name}")
    
    print(f"\nTeam size: {len(dynamic_team.agents)} agents")
    
    # Add a coordinator
    coordinator = Agent(
        name="Coordinator",
        llm_func=lambda p: "Coordinating team efforts...",
        max_iterations=2
    )
    
    dynamic_team.coordinator = coordinator
    print(f"Set coordinator: {coordinator.name}")
    
    print("\n" + "="*80)
    print("Multi-agent teams example completed!")
    print("="*80)
    
    # Summary
    print("\nImportant concepts demonstrated:")
    print("-"*80)
    print("1. AgentTeam coordinates multiple agents")
    print("2. Parallel execution: all agents work on the same goal simultaneously")
    print("3. Sequential execution: output flows from one agent to the next")
    print("4. Task delegation allows agents to delegate work to each other")
    print("5. Conversations track agent-to-agent communication")
    print("6. Results can be aggregated from multiple agents")
    print("7. Teams can be built dynamically with coordinators")


if __name__ == "__main__":
    main()
