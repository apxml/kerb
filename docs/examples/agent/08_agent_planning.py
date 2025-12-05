"""
Agent Planning Example
======================

This example demonstrates agent planning and plan execution.

Concepts covered:
- Creating plans with multiple steps
- Plan execution
- Replanning when plans fail
- Plan validation
- Plan-and-Execute agent pattern
"""

from kerb.agent.patterns import ReActAgent, PlanAndExecuteAgent
from kerb.agent.planning import Planner, Plan, create_plan
from typing import Dict, Any, List


def planner_llm(prompt: str) -> str:
    """LLM for planning."""
    if "plan" in prompt.lower():
        return """1. Research the topic

# %%
# Setup and Imports
# -----------------
2. Analyze findings
3. Write summary
4. Review and finalize"""
    return "Creating plan..."


def executor_llm(prompt: str) -> str:
    """LLM for execution."""
    if "research" in prompt.lower():
        return "Thought: I'll research the topic.\nFinal Answer: Research completed: Found 5 relevant sources"
    elif "analyze" in prompt.lower():
        return "Thought: I'll analyze the data.\nFinal Answer: Analysis done: Key insights identified"
    elif "write" in prompt.lower():
        return "Thought: I'll write the summary.\nFinal Answer: Summary written: 200 words"
    elif "review" in prompt.lower():
        return "Thought: I'll review the content.\nFinal Answer: Review complete: No issues found"
    return "Step executed"



# %%
# Main
# ----

def main():
    """Run agent planning example."""
    
    print("="*80)
    print("AGENT PLANNING EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # CREATE A PLAN
    # ========================================================================
    print("\n" + "="*80)
    print("1. CREATING A PLAN")
    print("="*80)
    
    # Manual plan creation
    plan = Plan(
        goal="Write a research report",
        steps=[
            "Research the topic",
            "Analyze findings",
            "Write the report",
            "Review and edit"
        ]
    )
    
    print(f"\nCreated plan: {plan.goal}")
    print(f"   Total steps: {len(plan.steps)}")
    
    print("\nPlan steps:")
    print("-"*80)
    for i, step in enumerate(plan.steps, 1):
        print(f"   {i}. {step}")
    
    # ========================================================================
    # USING A PLANNER
    # ========================================================================
    print("\n" + "="*80)
    print("2. USING A PLANNER")
    print("="*80)
    
    planner = Planner(llm_func=planner_llm)
    
    goal = "Create a data analysis pipeline"
    print(f"\nGoal: {goal}")
    print("\nPlanner generating steps...")
    
    generated_plan = planner.create_plan(goal)
    
    print(f"\nPlan generated:")
    print(f"   Goal: {generated_plan.goal}")
    print(f"   Steps: {len(generated_plan.steps)}")
    
    print("\nGenerated steps:")
    print("-"*80)
    for i, step in enumerate(generated_plan.steps, 1):
        print(f"   {i}. {step}")
    
    # ========================================================================
    # PLAN EXECUTION
    # ========================================================================
    print("\n" + "="*80)
    print("3. PLAN EXECUTION")
    print("="*80)
    
    execution_plan = Plan(
        goal="Complete research task",
        steps=[
            "Research topic",
            "Analyze findings",
            "Write summary"
        ]
    )
    
    print(f"\nExecuting plan: {execution_plan.goal}")
    print("-"*80)
    
    # Simulate execution
    results = []
    step_num = 1
    while not execution_plan.is_complete():
        step = execution_plan.next_step()
        if step:
            print(f"\n[Step {step_num}] {step}")
            
            # Execute action (simplified)
            result = executor_llm(f"Execute: {step}")
            results.append({
                'step': step_num,
                'action': step,
                'result': result,
                'status': 'completed'
            })
            
            print(f"   Result: {result[:80]}...")
            step_num += 1
    
    print(f"\nPlan execution completed")
    print(f"   Steps executed: {len(results)}")
    
    # ========================================================================
    # REPLANNING
    # ========================================================================
    print("\n" + "="*80)
    print("4. REPLANNING")
    print("="*80)
    
    original_plan = Plan(
        goal="Deploy application",
        steps=[
            "Run tests",
            "Build application",
            "Deploy to production"
        ]
    )
    
    print(f"\nOriginal plan: {original_plan.goal}")
    for i, step in enumerate(original_plan.steps, 1):
        print(f"   {i}. {step}")
    
    # Simulate failure at step 2
    print(f"\nStep 2 failed: Build errors detected")
    print(f"\nReplanning...")
    
    # Create recovery plan
    recovery_plan = Plan(
        goal="Recover from build failure",
        steps=[
            "Debug build issues",
            "Fix errors",
            "Rebuild application",
            "Deploy to production"
        ]
    )
    
    print(f"\nRecovery plan created:")
    for i, step in enumerate(recovery_plan.steps, 1):
        print(f"   {i}. {step}")
    
    # ========================================================================
    # PLAN-AND-EXECUTE AGENT
    # ========================================================================
    print("\n" + "="*80)
    print("5. PLAN-AND-EXECUTE AGENT")
    print("="*80)
    
    # Create plan-and-execute agent
    pae_agent = PlanAndExecuteAgent(
        name="PlanExecuteAgent",
        llm_func=planner_llm,
        max_iterations=5
    )
    
    goal = "Analyze customer feedback"
    print(f"\nGoal: {goal}")
    print(f"\nAgent: {pae_agent.name}")
    print("\nRunning Plan-and-Execute agent...")
    print("-"*80)
    
    result = pae_agent.run(goal)
    
    print(f"\nRESULTS:")
    print("-"*80)
    print(f"Status: {result.status.value}")
    print(f"Steps taken: {len(result.steps)}")
    print(f"Output: {result.output[:100]}...")
    
    # Show execution flow
    print("\nExecution Flow:")
    print("-"*80)
    for i, step in enumerate(result.steps, 1):
        print(f"\n[{i}] ", end="")
        if step.thought:
            print(f"Thought: {step.thought[:60]}...")
        if step.action:
            print(f"    Action: {step.action}")
    
    print("\n" + "="*80)
    print("Agent planning example completed!")
    print("="*80)
    
    # Summary
    print("\nImportant concepts demonstrated:")
    print("-"*80)
    print("1. Plans consist of goals and ordered steps")
    print("2. Planners generate plans from goals using LLMs")
    print("3. Plans execute steps sequentially")
    print("4. Failed steps can trigger replanning")
    print("5. Plan-and-Execute agents combine planning and execution")
    print("6. Plans track progress through current_step")


if __name__ == "__main__":
    main()
