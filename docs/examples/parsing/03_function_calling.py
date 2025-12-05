"""
Function Calling and Tool Use Formatting
========================================

This example demonstrates how to format function definitions for LLM function calling,
parse function calls from LLM outputs, and handle tool use patterns.

Main concepts:
- Formatting function definitions for LLMs
- Creating tool definitions (OpenAI format)
- Parsing function calls from LLM responses
- Formatting function results for LLM consumption
"""

from kerb.parsing import (
    format_function_call,
    format_tool_call,
    parse_function_call,
    format_function_result,
    ParseMode
)


# Example functions that an LLM might call
def get_weather(location: str, units: str = "celsius") -> dict:
    """Get weather information for a location."""
    # Simulated weather data
    weather_data = {
        "New York": {"temp": 22, "condition": "Sunny", "humidity": 65},
        "London": {"temp": 15, "condition": "Cloudy", "humidity": 80},
        "Tokyo": {"temp": 28, "condition": "Clear", "humidity": 55}
    }
    return weather_data.get(location, {"temp": 20, "condition": "Unknown", "humidity": 50})


def search_database(query: str, limit: int = 10, filters: dict = None) -> list:
    """Search database with optional filters."""

# %%
# Setup and Imports
# -----------------
    # Simulated search results
    return [
        {"id": 1, "title": f"Result for {query}", "score": 0.95},
        {"id": 2, "title": f"Another match for {query}", "score": 0.87}
    ]



# %%
# Send Email
# ----------

def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    return {"status": "sent", "message_id": "msg_12345", "to": to}


def simulate_llm_with_function_call(prompt: str) -> str:
    """Simulate LLM responses with function calls."""
    
    if "weather" in prompt.lower():
        return """{
  "function": "get_weather",
  "arguments": {
    "location": "New York",
    "units": "celsius"
  }
}"""
    
    elif "search" in prompt.lower():
        return """I'll search the database for you.

```json
{
  "function": "search_database",
  "arguments": {
    "query": "machine learning tutorials",
    "limit": 5,
    "filters": {"category": "education"}
  }
}
```"""
    
    elif "email" in prompt.lower():
        return """{
  "function": "send_email",
  "arguments": {
    "to": "team@example.com",
    "subject": "Project Update",
    "body": "The project is on track for delivery next week."
  }
}"""
    
    return "{}"



# %%
# Main
# ----

def main():
    """Run function calling examples."""
    
    print("="*80)
    print("FUNCTION CALLING AND TOOL USE FORMATTING")
    print("="*80)
    
    # Example 1: Format function definition
    print("\nExample 1: Format Function Definition")
    print("-"*80)
    
    weather_func = format_function_call(
        name="get_weather",
        description="Get current weather information for a specific location",
        parameters={
            "location": {
                "type": "string",
                "description": "City name (e.g., 'New York', 'London')"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units",
                "default": "celsius"
            }
        },
        required=["location"]
    )
    
    print("Weather Function Definition:")
    print(f"  Name: {weather_func['name']}")
    print(f"  Description: {weather_func['description']}")
    print(f"  Required params: {weather_func['parameters'].get('required', [])}")
    print(f"  Parameters: {', '.join(weather_func['parameters']['properties'].keys())}")
    
    # Example 2: Format tool definition (OpenAI format)
    print("\n\nExample 2: Format Tool Definition (OpenAI Format)")
    print("-"*80)
    
    search_tool = format_tool_call(
        name="search_database",
        description="Search the knowledge database with optional filters",
        parameters={
            "query": {
                "type": "string",
                "description": "Search query string"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 10
            },
            "filters": {
                "type": "object",
                "description": "Optional filters to apply",
                "properties": {
                    "category": {"type": "string"},
                    "date_range": {"type": "string"}
                }
            }
        },
        required=["query"]
    )
    
    print("Search Tool Definition:")
    print(f"  Type: {search_tool['type']}")
    print(f"  Function Name: {search_tool['function']['name']}")
    print(f"  Description: {search_tool['function']['description']}")
    
    # Example 3: Parse function call from LLM output
    print("\n\nExample 3: Parse Function Call from LLM")
    print("-"*80)
    
    llm_output = simulate_llm_with_function_call("What's the weather in New York?")
    print(f"LLM Output:\n{llm_output}\n")
    
    result = parse_function_call(llm_output)
    
    if result.success:
        call_data = result.data
        print(f"Parsed Function Call:")
        print(f"  Function: {call_data.get('function')}")
        print(f"  Arguments: {call_data.get('arguments')}")
        
        # Execute the function
        func_name = call_data.get('function')
        args = call_data.get('arguments', {})
        
        if func_name == "get_weather":
            weather = get_weather(**args)
            print(f"\nExecution Result: {weather}")
    
    # Example 4: Parse function call with markdown
    print("\n\nExample 4: Parse Function Call with Markdown")
    print("-"*80)
    
    llm_output = simulate_llm_with_function_call("search for tutorials")
    print(f"LLM Output:\n{llm_output}\n")
    
    result = parse_function_call(llm_output, mode=ParseMode.LENIENT)
    
    if result.success:
        call_data = result.data
        print(f"Parsed Function Call:")
        print(f"  Function: {call_data.get('function')}")
        print(f"  Arguments:")
        for key, value in call_data.get('arguments', {}).items():
            print(f"    - {key}: {value}")
    
    # Example 5: Format function result for LLM
    print("\n\nExample 5: Format Function Result for LLM")
    print("-"*80)
    
    function_name = "get_weather"
    function_result = {"temp": 22, "condition": "Sunny", "humidity": 65}
    
    formatted_result = format_function_result(
        result=function_result,
        name=function_name
    )
    
    print(f"Function: {function_name}")
    print(f"Result: {function_result}\n")
    print(f"Formatted for LLM:\n{formatted_result}")
    
    # Example 6: Complete function calling workflow
    print("\n\nExample 6: Complete Function Calling Workflow")
    print("-"*80)
    
    # Step 1: User request
    user_request = "Send an email to the team about project status"
    print(f"User Request: {user_request}\n")
    
    # Step 2: LLM generates function call
    llm_output = simulate_llm_with_function_call(user_request)
    print(f"LLM Function Call:\n{llm_output}\n")
    
    # Step 3: Parse function call
    result = parse_function_call(llm_output)
    
    if result.success:
        call_data = result.data
        func_name = call_data.get('function')
        args = call_data.get('arguments', {})
        
        print(f"Executing: {func_name}(**{args})\n")
        
        # Step 4: Execute function
        if func_name == "send_email":
            exec_result = send_email(**args)
            print(f"Execution Result: {exec_result}\n")
            
            # Step 5: Format result for LLM
            formatted = format_function_result(exec_result, name=func_name)
            print(f"Formatted Result for LLM:\n{formatted}")
    
    # Example 7: Multiple function definitions
    print("\n\nExample 7: Multiple Function Definitions")
    print("-"*80)
    
    available_functions = [
        format_function_call(
            name="get_weather",
            description="Get weather for a location",
            parameters={"location": {"type": "string"}},
            required=["location"]
        ),
        format_function_call(
            name="search_database",
            description="Search knowledge base",
            parameters={"query": {"type": "string"}},
            required=["query"]
        ),
        format_function_call(
            name="send_email",
            description="Send an email",
            parameters={
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            required=["to", "subject", "body"]
        )
    ]
    
    print("Available Functions:")
    for func in available_functions:
        print(f"  - {func['name']}: {func['description']}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
