"""Advanced prompt engineering patterns for LLM applications.

This example demonstrates:
- Chain-of-thought prompting
- Role-based prompts
- Multi-step reasoning workflows
- Prompt chaining for complex tasks
- Context-aware prompt construction
"""

from kerb.prompt import (
    render_template,
    create_version,
    register_prompt,
    get_prompt,
    create_example,
    format_examples,
    ExampleSelector
)


def chain_of_thought_prompting():
    """Use chain-of-thought to improve reasoning."""
    print("=" * 80)
    print("CHAIN-OF-THOUGHT PROMPTING")
    print("=" * 80)
    
    # Standard prompt (without CoT)
    standard = """Solve: If a train travels 120 miles in 2 hours, how far will it 
travel in 5 hours at the same speed?"""
    
    print("Standard prompt:")
    print(standard)
    
    # Chain-of-thought prompt
    cot_template = """Solve this problem step by step:

Problem: {{problem}}

Let's approach this systematically:
1. First, identify what we know
2. Then, determine what we need to find
3. Show your reasoning for each step
4. Finally, state the answer

Solution:"""
    
    cot_prompt = render_template(
        cot_template,
        {"problem": "If a train travels 120 miles in 2 hours, how far will it travel in 5 hours at the same speed?"}
    )
    
    print("\nChain-of-thought prompt:")
    print(cot_prompt)
    print("\nBenefit: Improved accuracy on reasoning tasks")


def few_shot_chain_of_thought():
    """Combine few-shot learning with chain-of-thought."""
    print("\n" + "=" * 80)
    print("FEW-SHOT CHAIN-OF-THOUGHT")
    print("=" * 80)
    
    # Create examples with reasoning steps
    examples = [
        create_example(
            input_text="Q: A store has 15 apples. They sell 6. How many remain?",
            output_text="""Let's solve step by step:
1. Starting amount: 15 apples
2. Amount sold: 6 apples  
3. Calculation: 15 - 6 = 9
Answer: 9 apples remain"""
        ),
        create_example(
            input_text="Q: If 3 pencils cost $6, how much do 5 pencils cost?",
            output_text="""Let's solve step by step:
1. Cost of 3 pencils: $6
2. Cost per pencil: $6 ÷ 3 = $2
3. Cost of 5 pencils: $2 × 5 = $10
Answer: $10"""
        )
    ]
    
    # Format examples
    examples_text = format_examples(
        examples,
        template="{input}\n{output}",
        separator="\n\n"
    )
    
    # Create complete prompt
    prompt_template = """Solve math problems with step-by-step reasoning.

{examples}

Now solve this problem:
Q: {{question}}

Let's solve step by step:"""
    
    complete_prompt = render_template(
        prompt_template,
        {
            "examples": examples_text,
            "question": "A box contains 24 chocolates. If 4 friends share them equally, how many does each get?"
        }
    )
    
    print(complete_prompt)
    print("\nBenefit: Demonstrates reasoning process through examples")


def role_based_prompts():
    """Create role-based prompts for different personas."""
    print("\n" + "=" * 80)
    print("ROLE-BASED PROMPTS")
    print("=" * 80)
    
    # Define different expert roles
    roles = {
        "code_reviewer": {
            "role": "senior software engineer",
            "expertise": "code quality, performance, and best practices",
            "style": "constructive and detailed"
        },
        "security_analyst": {
            "role": "cybersecurity expert",
            "expertise": "security vulnerabilities and threat analysis",
            "style": "thorough and risk-focused"
        },
        "architect": {
            "role": "software architect",
            "expertise": "system design, scalability, and architecture patterns",
            "style": "high-level and strategic"
        }
    }
    
    template = """You are a {{role}} with expertise in {{expertise}}.
Your communication style is {{style}}.

Analyze the following:
{{content}}

Provide your expert assessment:"""
    
    # Code review perspective
    code_review_prompt = render_template(template, {
        **roles["code_reviewer"],
        "content": "def process(data):\n    return [x*2 for x in data]"
    })
    
    print("Code Reviewer Role:")
    print(code_review_prompt)
    
    # Security analyst perspective  
    security_prompt = render_template(template, {
        **roles["security_analyst"],
        "content": "User input: request.form['username']"
    })
    
    print("\nSecurity Analyst Role:")
    print(security_prompt)
    
    print("\nBenefit: Tailored responses based on expert perspective")


def multi_step_workflow():
    """Chain prompts for multi-step tasks."""
    print("\n" + "=" * 80)
    print("MULTI-STEP WORKFLOW")
    print("=" * 80)
    
    # Step 1: Analysis
    analysis_template = """Analyze this code for potential issues:

{{code}}

List all issues found:"""
    
    # Step 2: Prioritization
    prioritization_template = """Given these code issues:

{{issues}}

Prioritize them by severity (Critical, High, Medium, Low):"""
    
    # Step 3: Solution
    solution_template = """For this {{priority}} priority issue:

{{issue}}

Provide a detailed solution with code example:"""
    
    code = "def authenticate(user, pwd):\n    if user == pwd:\n        return True"
    
    # Simulate step 1
    step1_prompt = render_template(analysis_template, {"code": code})
    print("Step 1 - Analysis Prompt:")
    print(step1_prompt)
    
    # Simulate step 2 (with hypothetical step 1 output)
    issues = "1. Security issue: comparing username with password\n2. No password hashing"
    step2_prompt = render_template(prioritization_template, {"issues": issues})
    print("\nStep 2 - Prioritization Prompt:")
    print(step2_prompt)
    
    # Simulate step 3
    step3_prompt = render_template(solution_template, {
        "priority": "Critical",
        "issue": "comparing username with password"
    })
    print("\nStep 3 - Solution Prompt:")
    print(step3_prompt)
    
    print("\nBenefit: Break complex tasks into manageable steps")


def context_aware_prompts():
    """Build context-aware prompts based on user state."""
    print("\n" + "=" * 80)
    print("CONTEXT-AWARE PROMPTS")
    print("=" * 80)
    
    # User context
    user_context = {
        "skill_level": "intermediate",
        "previous_topics": ["functions", "classes"],
        "current_goal": "learn decorators",
        "learning_style": "visual with examples"
    }
    
    # Beginner template
    beginner_template = """Let's learn {{topic}} from the basics.

I'll explain it simply with lots of examples. We'll start with the fundamentals
and build up gradually.

{{topic}} explained:"""
    
    # Intermediate template
    intermediate_template = """Building on your knowledge of {{previous_topics}}, 
let's explore {{topic}}.

Since you prefer {{learning_style}}, I'll provide practical examples showing:
- How {{topic}} works
- Common use cases
- Best practices

{{topic}} tutorial:"""
    
    # Advanced template
    advanced_template = """Advanced {{topic}} concepts:

Assuming familiarity with {{previous_topics}}, we'll cover:
- Advanced patterns
- Performance considerations  
- Production use cases

Deep dive into {{topic}}:"""
    
    # Select appropriate template based on skill level
    if user_context["skill_level"] == "beginner":
        template = beginner_template
    elif user_context["skill_level"] == "intermediate":
        template = intermediate_template
    else:
        template = advanced_template
    
    prompt = render_template(template, {
        "topic": user_context["current_goal"],
        "previous_topics": ", ".join(user_context["previous_topics"]),
        "learning_style": user_context["learning_style"]
    })
    
    print(f"Context: {user_context['skill_level']} learner")
    print(f"Customized prompt:")
    print(prompt)
    
    print("\nBenefit: Personalized prompts based on user state")


def prompt_versioning_for_workflows():
    """Manage prompt versions for different workflow stages."""
    print("\n" + "=" * 80)
    print("PROMPT VERSIONING FOR WORKFLOWS")
    print("=" * 80)
    
    # Create versions for different stages
    stages = [
        ("research", "1.0", "Research {{topic}}: Find key concepts and definitions"),
        ("outline", "1.0", "Create outline for {{topic}}: Main sections and subtopics"),
        ("draft", "1.0", "Write draft on {{topic}}: Include examples and explanations"),
        ("review", "1.0", "Review content on {{topic}}: Check accuracy and clarity"),
        ("polish", "1.0", "Polish final version: Improve flow and readability")
    ]
    
    for name, version, template in stages:
        v = create_version(
            name=f"writing_workflow_{name}",
            version=version,
            template=template,
            description=f"{name.capitalize()} stage of writing workflow"
        )
        register_prompt(v)
    
    print("Created writing workflow with 5 stages:")
    for name, _, _ in stages:
        print(f"  - {name}")
    
    # Execute workflow
    topic = "Python decorators"
    print(f"\nExecuting workflow for topic: '{topic}'")
    
    for name, _, _ in stages:
        prompt_version = get_prompt(f"writing_workflow_{name}", "1.0")
        prompt = prompt_version.render({"topic": topic})
        print(f"\n{name.upper()} stage:")
        print(f"  {prompt}")
    
    print("\nBenefit: Consistent multi-stage workflows")


def adaptive_prompting():
    """Adapt prompts based on previous responses."""
    print("\n" + "=" * 80)
    print("ADAPTIVE PROMPTING")
    print("=" * 80)
    
    # Initial prompt
    initial_template = """Explain {{concept}} to a {{level}} developer.

{{concept}}:"""
    
    # Follow-up if response was too complex
    simplify_template = """Your previous explanation of {{concept}} was too technical.

Please explain {{concept}} using:
- Simple language
- Concrete examples
- No jargon

Simplified explanation:"""
    
    # Follow-up if response was too simple
    elaborate_template = """Your explanation of {{concept}} was good, but I need more depth.

Please provide:
- Advanced details
- Edge cases
- Performance implications

Detailed explanation:"""
    
    concept = "async/await"
    
    # Initial attempt
    initial = render_template(initial_template, {
        "concept": concept,
        "level": "intermediate"
    })
    print("Initial prompt:")
    print(initial)
    
    # Simulate adaptation based on response quality
    print("\nIf response too complex, use:")
    simplified = render_template(simplify_template, {"concept": concept})
    print(simplified)
    
    print("\nIf response too simple, use:")
    detailed = render_template(elaborate_template, {"concept": concept})
    print(detailed)
    
    print("\nBenefit: Iteratively refine responses")


def structured_output_prompts():
    """Generate prompts for structured outputs."""
    print("\n" + "=" * 80)
    print("STRUCTURED OUTPUT PROMPTS")
    print("=" * 80)
    
    # JSON output prompt
    json_template = """Analyze this code and return JSON:

{{code}}

Return format:
{
  "issues": [{"type": "...", "severity": "...", "description": "..."}],
  "metrics": {"complexity": ..., "lines": ...},
  "suggestions": ["..."]
}

Analysis:"""
    
    # Markdown output prompt
    markdown_template = """Review this code and format as Markdown:

{{code}}

Use this structure:
## Issues Found
- List issues here

## Metrics
- Complexity: 
- Lines:

## Recommendations
1. First recommendation
2. Second recommendation

Review:"""
    
    code = "def f(x): return x*x"
    
    print("JSON output prompt:")
    json_prompt = render_template(json_template, {"code": code})
    print(json_prompt)
    
    print("\nMarkdown output prompt:")
    md_prompt = render_template(markdown_template, {"code": code})
    print(md_prompt)
    
    print("\nBenefit: Consistent, parseable outputs")


def main():
    """Run all advanced prompt engineering examples."""
    print("\n" + "=" * 80)
    print("ADVANCED PROMPT ENGINEERING FOR LLM DEVELOPERS")
    print("=" * 80)
    
    chain_of_thought_prompting()
    few_shot_chain_of_thought()
    role_based_prompts()
    multi_step_workflow()
    context_aware_prompts()
    prompt_versioning_for_workflows()
    adaptive_prompting()
    structured_output_prompts()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
