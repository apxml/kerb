"""Agent Memory Example

This example demonstrates agent memory systems.

Concepts covered:
- Working memory for current task
- Episodic memory for past experiences
- Saving and loading agent state
- Memory-augmented agents
- Context retention across executions
"""

from kerb.agent.patterns import ReActAgent
from kerb.agent.memory import AgentMemory, WorkingMemory, EpisodicMemory
from kerb.agent.core import AgentState, AgentStep
from typing import Dict, Any


def memory_aware_llm(prompt: str, memory: Dict[str, Any] = None) -> str:
    """LLM that uses memory context."""
    memory = memory or {}
    
    # Check working memory
    if memory.get('recent_facts'):
        facts = memory['recent_facts']
        if 'user_name' in facts:
            return f"Hello {facts['user_name']}, I remember you!"
    
    # Check episodic memory
    if memory.get('past_conversations'):
        return "Based on our previous conversations, I can help you better."
    
    return "Processing your request..."


def main():
    """Run agent memory example."""
    
    print("="*80)
    print("AGENT MEMORY EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # WORKING MEMORY
    # ========================================================================
    print("\n" + "="*80)
    print("1. WORKING MEMORY")
    print("="*80)
    
    working_memory = WorkingMemory(capacity=5)
    
    print("\nAdding items to working memory...")
    print("-"*80)
    
    # Add items (as dict entries)
    working_memory.add({"type": "user_name", "value": "Alice"})
    working_memory.add({"type": "user_preference", "value": "Python"})
    working_memory.add({"type": "current_task", "value": "Learn about agents"})
    working_memory.add({"type": "context", "value": "Educational"})
    
    print(f"Added {len(working_memory.items)} items")
    
    # Retrieve items
    print("\nRetrieving from working memory:")
    for item in working_memory.items:
        if isinstance(item, dict) and item.get('type') == 'user_name':
            print(f"   User name: {item.get('value')}")
        elif isinstance(item, dict) and item.get('type') == 'user_preference':
            print(f"   Preference: {item.get('value')}")
        elif isinstance(item, dict) and item.get('type') == 'current_task':
            print(f"   Task: {item.get('value')}")
    
    # Test capacity limit
    print(f"\nMemory capacity: {working_memory.capacity}")
    print("   Adding more items to test capacity...")
    
    for i in range(3):
        working_memory.add({"type": f"extra_item_{i}", "value": f"value_{i}"})
    
    print(f"   Total items (after overflow): {len(working_memory.items)}")
    recent_types = [item.get('type') if isinstance(item, dict) else str(item) for item in working_memory.items[:3]]
    print(f"   Recent items: {recent_types}")
    
    # ========================================================================
    # EPISODIC MEMORY
    # ========================================================================
    print("\n" + "="*80)
    print("2. EPISODIC MEMORY")
    print("="*80)
    
    episodic_memory = EpisodicMemory()
    
    print("\nAdding episodes...")
    print("-"*80)
    
    # Add episodes (past interactions)
    episode1 = {
        'query': 'What is Python?',
        'response': 'Python is a programming language.',
        'outcome': 'helpful'
    }
    
    episode2 = {
        'query': 'How do I use agents?',
        'response': 'Create an Agent instance and call run().',
        'outcome': 'helpful'
    }
    
    episode3 = {
        'query': 'What about memory?',
        'response': 'Agents can use WorkingMemory and EpisodicMemory.',
        'outcome': 'helpful'
    }
    
    episodic_memory.add_episode(episode1)
    episodic_memory.add_episode(episode2)
    episodic_memory.add_episode(episode3)
    
    print(f"Added {len(episodic_memory.episodes)} episodes")
    
    # Retrieve recent episodes
    recent = episodic_memory.get_recent_episodes(n=2)
    
    print("\nRecent episodes (last 2):")
    for i, episode in enumerate(recent, 1):
        print(f"\n   Episode {i}:")
        print(f"     Query: {episode['query']}")
        print(f"     Response: {episode['response'][:50]}...")
    
    # Search episodes
    print("\nSearching episodes for 'Python'...")
    python_episodes = [e for e in episodic_memory.episodes 
                       if 'Python' in e.get('query', '') or 'Python' in e.get('response', '')]
    
    print(f"   Found {len(python_episodes)} matching episodes")
    
    # ========================================================================
    # AGENT WITH MEMORY
    # ========================================================================
    print("\n" + "="*80)
    print("3. AGENT WITH MEMORY")
    print("="*80)
    
    # Create agent memory
    agent_memory = AgentMemory()
    working_mem = WorkingMemory(capacity=10)
    episodic_mem = EpisodicMemory()
    
    # Store some context
    working_mem.add({"type": "user_name", "value": "Bob"})
    working_mem.add({"type": "expertise_level", "value": "intermediate"})
    
    def memory_llm(prompt: str) -> str:
        """LLM that uses agent memory."""
        # Check working memory for context
        if working_mem.items:
            context_items = [f"{item.get('type', 'item')}: {item.get('value', item)}" 
                           for item in working_mem.items]
            context = ", ".join(context_items)
            return f"Based on context ({context}), I can help with {prompt.lower()}"
        return f"I can help with {prompt}"
    
    # Create agent with memory
    memory_agent = ReActAgent(
        name="MemoryAgent",
        llm_func=memory_llm,
        max_iterations=2
    )
    
    print("\nRunning memory-aware agent...")
    print("-"*80)
    
    result = memory_agent.run("Help me with Python")
    
    print(f"\nAgent response:")
    print(f"   {result.output}")
    
    # Store the interaction in episodic memory
    episodic_mem.add_episode({
        'query': "Help me with Python",
        'response': result.output,
        'timestamp': 'now',
        'outcome': 'successful'
    })
    
    print(f"\nStored interaction in episodic memory")
    print(f"   Total episodes: {len(episodic_mem.episodes)}")
    
    # ========================================================================
    # SAVING AND LOADING STATE
    # ========================================================================
    print("\n" + "="*80)
    print("4. SAVING AND LOADING STATE")
    print("="*80)
    
    # Create agent state
    state = AgentState(
        goal="Process data",
        max_iterations=5
    )
    
    # Add some steps
    state.add_step(AgentStep(
        step_number=1,
        thought="Analyzing the problem",
        action="analyze",
        observation="Data processed"
    ))
    
    state.add_step(AgentStep(
        step_number=2,
        thought="Extracting insights",
        action="extract",
        observation="Insights found"
    ))
    
    # Update beliefs (agent's knowledge)
    state.beliefs['data_analyzed'] = True
    state.beliefs['insights_count'] = 5
    
    print("\n💾 Agent State:")
    print("-"*80)
    print(f"   Goal: {state.goal}")
    print(f"   Current step: {state.current_step}")
    print(f"   Total steps: {len(state.steps)}")
    print(f"   Beliefs: {state.beliefs}")
    
    # Simulate saving (in practice, you'd serialize to disk)
    saved_state = {
        'goal': state.goal,
        'steps': [step.to_dict() for step in state.steps],
        'current_step': state.current_step,
        'beliefs': state.beliefs,
        'status': state.status.value
    }
    
    print("\nState saved (serialized)")
    print(f"   Keys: {list(saved_state.keys())}")
    
    # Simulate loading
    print("\nLoading state...")
    print(f"   Restored goal: {saved_state['goal']}")
    print(f"   Restored steps: {len(saved_state['steps'])}")
    print(f"   Restored beliefs: {saved_state['beliefs']}")
    
    # ========================================================================
    # MEMORY PERSISTENCE
    # ========================================================================
    print("\n" + "="*80)
    print("5. MEMORY PERSISTENCE ACROSS SESSIONS")
    print("="*80)
    
    # Session 1
    print("\nSession 1:")
    print("-"*80)
    
    session1_working = WorkingMemory()
    session1_episodic = EpisodicMemory()
    
    session1_working.add({"type": "user_id", "value": "user123"})
    session1_working.add({"type": "session_start", "value": "2024-01-01"})
    
    session1_episodic.add_episode({
        'action': 'login',
        'result': 'success'
    })
    
    print(f"   Working memory items: {len(session1_working.items)}")
    print(f"   Episodes: {len(session1_episodic.episodes)}")
    
    # Export memory
    exported_memory = {
        'working': session1_working.items,
        'episodic': session1_episodic.episodes
    }
    
    print(f"\nMemory exported")
    
    # Session 2 (simulate new session)
    print("\nSession 2 (after restart):")
    print("-"*80)
    
    session2_working = WorkingMemory()
    session2_episodic = EpisodicMemory()
    
    # Restore memory
    for item in exported_memory['working']:
        session2_working.add(item)
    
    for episode in exported_memory['episodic']:
        session2_episodic.add_episode(episode)
    
    print(f"   Restored working memory: {len(session2_working.items)} items")
    print(f"   Restored episodes: {len(session2_episodic.episodes)}")
    user_id = None
    for item in session2_working.items:
        if isinstance(item, dict) and item.get('type') == 'user_id':
            user_id = item.get('value')
    print(f"   User ID: {user_id}")
    
    print("\nMemory persisted across sessions!")
    
    print("\n" + "="*80)
    print("Agent memory example completed!")
    print("="*80)
    
    # Summary
    print("\nImportant concepts demonstrated:")
    print("-"*80)
    print("1. WorkingMemory stores current context (limited capacity)")
    print("2. EpisodicMemory stores past interactions and experiences")
    print("3. AgentMemory combines working and episodic memory")
    print("4. Memory can be saved and loaded for persistence")
    print("5. Agents can use memory to maintain context")
    print("6. Memory enables continuity across sessions")


if __name__ == "__main__":
    main()
