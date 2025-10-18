"""Pydantic Model Parsing from LLM Outputs

This example demonstrates how to parse LLM outputs into validated Pydantic models,
ensuring type safety and data validation for structured LLM responses.

Main concepts:
- Defining Pydantic models for LLM outputs
- Parsing text to Pydantic instances
- Converting Pydantic models to JSON Schema
- Validation and error handling
"""

from typing import List, Optional
from kerb.parsing import parse_to_pydantic, pydantic_to_schema, ParseMode

try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    print("This example requires pydantic. Install with: pip install pydantic")
    exit(0)


# Define Pydantic models for various LLM tasks
class UserProfile(BaseModel):
    """User profile information."""
    name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email address")
    age: int = Field(..., ge=0, le=150, description="User's age")
    roles: List[str] = Field(default_factory=list, description="User roles")
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    text: str = Field(..., description="The analyzed text")
    sentiment: str = Field(..., description="Sentiment: positive, negative, or neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    key_phrases: List[str] = Field(default_factory=list, description="Key phrases")


class EntityExtraction(BaseModel):
    """Named entity extraction result."""
    text: str = Field(..., description="Original text")
    entities: List[dict] = Field(default_factory=list, description="Extracted entities")
    
    class Entity(BaseModel):
        type: str
        value: str
        confidence: float


class TaskBreakdown(BaseModel):
    """Task breakdown for planning."""
    task: str = Field(..., description="Main task")
    subtasks: List[str] = Field(..., description="List of subtasks")
    priority: str = Field(..., description="Priority: high, medium, or low")
    estimated_time: Optional[int] = Field(None, description="Estimated time in minutes")


def simulate_llm_response(task_type: str) -> str:
    """Simulate LLM responses for different task types."""
    
    if task_type == "user_profile":
        return """Here's the user profile:

```json
{
  "name": "Alice Johnson",
  "email": "alice@example.com",
  "age": 28,
  "roles": ["developer", "team_lead", "mentor"]
}
```"""
    
    elif task_type == "sentiment":
        return """{
  "text": "This product is amazing! Best purchase ever.",
  "sentiment": "positive",
  "confidence": 0.95,
  "key_phrases": ["amazing", "best purchase"]
}"""
    
    elif task_type == "entities":
        return """Entity extraction result:

{
  "text": "Apple CEO Tim Cook announced the new iPhone in San Francisco.",
  "entities": [
    {"type": "ORGANIZATION", "value": "Apple", "confidence": 0.98},
    {"type": "PERSON", "value": "Tim Cook", "confidence": 0.96},
    {"type": "PRODUCT", "value": "iPhone", "confidence": 0.99},
    {"type": "LOCATION", "value": "San Francisco", "confidence": 0.97}
  ]
}"""
    
    elif task_type == "task_breakdown":
        return """{
  "task": "Build a web scraper",
  "subtasks": [
    "Design URL structure and pagination logic",
    "Implement HTTP request handling with retries",
    "Parse HTML and extract target data",
    "Store results in database",
    "Add error handling and logging"
  ],
  "priority": "high",
  "estimated_time": 240
}"""
    
    return "{}"


def main():
    """Run Pydantic model parsing examples."""
    
    print("="*80)
    print("PYDANTIC MODEL PARSING FROM LLM OUTPUTS")
    print("="*80)
    
    # Example 1: Parse user profile
    print("\nExample 1: User Profile Parsing")
    print("-"*80)
    
    llm_output = simulate_llm_response("user_profile")
    print(f"LLM Output:\n{llm_output}\n")
    
    result = parse_to_pydantic(llm_output, UserProfile)
    
    if result.success:
        user = result.data
        print(f"Parsed User Profile:")
        print(f"  Name: {user.name}")
        print(f"  Email: {user.email}")
        print(f"  Age: {user.age}")
        print(f"  Roles: {', '.join(user.roles)}")
    else:
        print(f"Error: {result.error}")
    
    # Example 2: Sentiment analysis
    print("\n\nExample 2: Sentiment Analysis Parsing")
    print("-"*80)
    
    llm_output = simulate_llm_response("sentiment")
    print(f"LLM Output:\n{llm_output}\n")
    
    result = parse_to_pydantic(llm_output, SentimentAnalysis, mode=ParseMode.LENIENT)
    
    if result.success:
        sentiment = result.data
        print(f"Sentiment Analysis:")
        print(f"  Text: {sentiment.text}")
        print(f"  Sentiment: {sentiment.sentiment}")
        print(f"  Confidence: {sentiment.confidence:.2%}")
        print(f"  Key Phrases: {', '.join(sentiment.key_phrases)}")
    
    # Example 3: Entity extraction
    print("\n\nExample 3: Named Entity Extraction")
    print("-"*80)
    
    llm_output = simulate_llm_response("entities")
    print(f"LLM Output:\n{llm_output}\n")
    
    result = parse_to_pydantic(llm_output, EntityExtraction)
    
    if result.success:
        extraction = result.data
        print(f"Original Text: {extraction.text}\n")
        print(f"Extracted Entities:")
        for entity in extraction.entities:
            print(f"  - {entity['type']}: {entity['value']} (confidence: {entity['confidence']:.2%})")
    
    # Example 4: Task breakdown
    print("\n\nExample 4: Task Breakdown Parsing")
    print("-"*80)
    
    llm_output = simulate_llm_response("task_breakdown")
    print(f"LLM Output:\n{llm_output}\n")
    
    result = parse_to_pydantic(llm_output, TaskBreakdown)
    
    if result.success:
        task = result.data
        print(f"Task: {task.task}")
        print(f"Priority: {task.priority}")
        print(f"Estimated Time: {task.estimated_time} minutes\n")
        print("Subtasks:")
        for i, subtask in enumerate(task.subtasks, 1):
            print(f"  {i}. {subtask}")
    
    # Example 5: Generate JSON Schema from Pydantic model
    print("\n\nExample 5: Pydantic Model to JSON Schema")
    print("-"*80)
    
    schema = pydantic_to_schema(SentimentAnalysis)
    print("JSON Schema for SentimentAnalysis:")
    print(f"  Title: {schema.get('title', 'N/A')}")
    print(f"  Properties: {', '.join(schema.get('properties', {}).keys())}")
    print(f"  Required: {schema.get('required', [])}")
    
    # Example 6: Error handling with validation
    print("\n\nExample 6: Validation Error Handling")
    print("-"*80)
    
    # Invalid email format
    invalid_profile = '{"name": "Bob", "email": "invalid-email", "age": 30, "roles": []}'
    print(f"Invalid Profile:\n{invalid_profile}\n")
    
    result = parse_to_pydantic(invalid_profile, UserProfile)
    
    if result.success:
        print(f"Parsed successfully: {result.data}")
    else:
        print(f"Validation Error: {result.error}")
    
    # Age out of range
    invalid_age = '{"name": "Charlie", "email": "charlie@example.com", "age": 200, "roles": []}'
    print(f"\nInvalid Age:\n{invalid_age}\n")
    
    result = parse_to_pydantic(invalid_age, UserProfile)
    
    if result.success:
        print(f"Parsed successfully: {result.data}")
    else:
        print(f"Validation Error: {result.error}")
    
    # Example 7: Practical use case - structured extraction
    print("\n\nExample 7: Structured Information Extraction")
    print("-"*80)
    
    # Simulate extracting structured data from unstructured text
    print("Use Case: Extract user information from customer support conversation\n")
    
    llm_output = """Based on the conversation, I extracted:

{
  "name": "Emma Davis",
  "email": "emma.davis@company.com",
  "age": 32,
  "roles": ["customer", "premium_member"]
}"""
    
    print(f"LLM Output:\n{llm_output}\n")
    
    result = parse_to_pydantic(llm_output, UserProfile)
    
    if result.success:
        user = result.data
        print("Successfully extracted and validated user profile:")
        print(f"  {user.name} ({user.email})")
        print(f"  Member status: {', '.join(user.roles)}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
