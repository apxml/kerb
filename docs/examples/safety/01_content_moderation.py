"""
Content Moderation Example
==========================

This example demonstrates how to use content moderation functions to detect
and filter harmful content in LLM applications.

Main concepts:
- Using moderate_content for comprehensive checks
- Checking specific categories (toxicity, hate speech, profanity)
- Filtering LLM outputs before showing to users
- Setting different safety levels for different contexts
"""

from kerb.safety import (
    moderate_content,
    check_toxicity,
    check_hate_speech,
    check_profanity,
    SafetyLevel,
    ContentCategory
)


def simulate_llm_response(prompt: str) -> str:
    """Simulate LLM responses for demonstration."""
    responses = {
        "greeting": "Hello! How can I help you today?",
        "angry": "I hate dealing with stupid questions like this!",
        "professional": "Thank you for your inquiry. I'll be happy to assist you.",
        "offensive": "You're an idiot if you don't understand this.",
    }
    return responses.get(prompt, "I'm here to help!")


def main():
    """Run content moderation example."""

# %%
# Setup and Imports
# -----------------
    
    print("="*80)
    print("CONTENT MODERATION EXAMPLE")
    print("="*80)
    
    # Example 1: Basic content moderation
    print("\nExample 1: Basic Content Moderation")
    print("-"*80)
    
    safe_text = "Hello! How can I help you today?"
    result = moderate_content(safe_text)
    
    print(f"Text: {safe_text}")
    print(f"Safe: {result.safe}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Toxicity Level: {result.toxicity_level.value}")
    
    # Example 2: Detecting toxic content
    print("\n\nExample 2: Detecting Toxic Content")
    print("-"*80)
    
    toxic_text = "I hate dealing with stupid questions like this!"
    result = moderate_content(toxic_text)
    
    print(f"Text: {toxic_text}")
    print(f"Safe: {result.safe}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Toxicity Level: {result.toxicity_level.value}")
    print(f"Flagged Categories: {[cat.value for cat in result.flagged_categories]}")
    
    # Example 3: Specific category checks
    print("\n\nExample 3: Specific Category Checks")
    print("-"*80)
    
    offensive_text = "You're an idiot if you don't understand this."
    
    # Check toxicity
    toxicity_result = check_toxicity(offensive_text)
    print(f"\nText: {offensive_text}")
    print(f"Toxicity Check - Safe: {toxicity_result.safe}, Score: {toxicity_result.score:.3f}")
    
    # Check profanity
    profanity_result = check_profanity(offensive_text)
    print(f"Profanity Check - Safe: {profanity_result.safe}, Score: {profanity_result.score:.3f}")
    
    # Example 4: Different safety levels
    print("\n\nExample 4: Different Safety Levels")
    print("-"*80)
    
    borderline_text = "This is ridiculous and annoying."
    
    for level in [SafetyLevel.PERMISSIVE, SafetyLevel.MODERATE, SafetyLevel.STRICT]:
        result = moderate_content(borderline_text, level=level)
        print(f"\n{level.value.upper()} Level:")
        print(f"  Safe: {result.safe}")
        print(f"  Score: {result.overall_score:.3f}")
    
    # Example 5: Filtering LLM outputs
    print("\n\nExample 5: Filtering LLM Outputs")
    print("-"*80)
    
    test_prompts = ["greeting", "angry", "professional", "offensive"]
    
    for prompt in test_prompts:
        llm_output = simulate_llm_response(prompt)
        result = moderate_content(llm_output, level=SafetyLevel.MODERATE)
        
        print(f"\nLLM Output: {llm_output}")
        print(f"Safe to Show User: {result.safe}")
        
        if not result.safe:
            print(f"Action: FILTER - Flagged for {[cat.value for cat in result.flagged_categories]}")
            print("Showing default response instead: 'I apologize, I cannot provide that response.'")
        else:
            print("Action: ALLOW - Showing to user")
    
    # Example 6: Selective category moderation
    print("\n\nExample 6: Selective Category Moderation")
    print("-"*80)
    
    text = "This damn feature is frustrating!"
    
    # Only check toxicity and hate speech, not profanity
    result = moderate_content(
        text,
        categories=[ContentCategory.TOXICITY, ContentCategory.HATE_SPEECH]
    )
    
    print(f"Text: {text}")
    print(f"Checking only: Toxicity and Hate Speech")
    print(f"Safe: {result.safe}")
    print(f"Category Scores:")
    for category, score in result.categories.items():
        print(f"  {category.value}: {score:.3f}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
