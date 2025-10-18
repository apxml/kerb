"""Agent Tools Example

This example demonstrates how to create and use tools with agents.

Concepts covered:
- Creating tools with the Tool class
- Using create_tool function
- Tool parameters and validation
- Tool execution and error handling
- Registering tools globally
"""

from kerb.agent.patterns import ReActAgent
from kerb.agent import Tool, create_tool
from kerb.agent.tools import ToolRegistry, ToolResult, ToolStatus
from typing import Dict, Any


# Method 1: Create tool using Tool class
def get_weather(location: str) -> str:
    """Get weather for a location."""
    weather_data = {
        "san francisco": "Sunny, 72°F",
        "new york": "Cloudy, 65°F",
        "london": "Rainy, 58°F",
        "tokyo": "Clear, 68°F"
    }
    
    location_key = location.lower()
    if location_key in weather_data:
        return f"Weather in {location}: {weather_data[location_key]}"
    return f"Weather data not available for {location}"


# Method 2: Use create_tool function
def get_stock_price(ticker: str) -> str:
    """Get stock price for a ticker."""
    # Mock stock prices
    prices = {
        "AAPL": "$180.25",
        "GOOGL": "$142.50",
        "MSFT": "$375.80",
        "TSLA": "$242.15"
    }
    
    ticker_upper = ticker.upper()
    if ticker_upper in prices:
        return f"{ticker_upper}: {prices[ticker_upper]}"
    return f"Price not available for {ticker_upper}"


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def count_characters(text: str) -> str:
    """Count characters in text."""
    return f"Character count: {len(text)}"


def mock_llm(prompt: str) -> str:
    """Mock LLM that suggests tool usage."""
    if "weather" in prompt.lower():
        return "Thought: I need to check the weather.\nAction: get_weather\nAction Input: Tokyo"
    elif "stock" in prompt.lower():
        return "Thought: I need stock information.\nAction: stock_price\nAction Input: AAPL"
    elif "calculate" in prompt.lower() or "math" in prompt.lower():
        return "Thought: I need to do a calculation.\nAction: calculator\nAction Input: 15 * 7"
    else:
        return "Thought: Processing request.\nFinal Answer: Request processed"


def main():
    """Run tools example."""
    
    print("="*80)
    print("AGENT TOOLS EXAMPLE")
    print("="*80)
    
    # Create tools using Tool class
    weather_tool = Tool(
        name="get_weather",
        description="Get current weather for a location",
        func=get_weather,
        parameters={
            "location": {
                "type": "string",
                "description": "City or location name"
            }
        },
        examples=["get_weather('San Francisco')", "get_weather('Tokyo')"]
    )
    
    print("\nTOOL CREATION")
    print("-"*80)
    print(f"Created tool: {weather_tool.name}")
    print(f"   Description: {weather_tool.description}")
    print(f"   Parameters: {list(weather_tool.parameters.keys())}")
    
    # Create tools using create_tool function
    stock_tool = create_tool(
        name="stock_price",
        func=get_stock_price,
        description="Get current stock price for a ticker symbol",
        parameters={
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., AAPL, GOOGL)"
            }
        }
    )
    
    calc_tool = create_tool(
        name="calculator",
        func=calculate,
        description="Perform mathematical calculations",
        parameters={
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        }
    )
    
    text_tool = create_tool(
        name="text_length",
        func=count_characters,
        description="Count the number of characters in text",
        parameters={
            "text": {
                "type": "string",
                "description": "Text to count"
            }
        }
    )
    
    # Test tool execution
    print("\nTOOL EXECUTION")
    print("-"*80)
    
    result1 = weather_tool.execute(location="San Francisco")
    print(f"Test 1: {weather_tool.name}('San Francisco')")
    print(f"  Status: {result1.status.value}")
    print(f"  Output: {result1.output}")
    
    result2 = stock_tool.execute(ticker="AAPL")
    print(f"\nTest 2: stock_price('AAPL')")
    print(f"  Status: {result2.status.value}")
    print(f"  Output: {result2.output}")
    
    result3 = calc_tool.execute(expression="15 * 7 + 10")
    print(f"\nTest 3: calculator('15 * 7 + 10')")
    print(f"  Status: {result3.status.value}")
    print(f"  Output: {result3.output}")
    
    # Create a tool registry
    print("\nTOOL REGISTRY")
    print("-"*80)
    
    registry = ToolRegistry()
    registry.register(weather_tool)
    registry.register(stock_tool)
    registry.register(calc_tool)
    registry.register(text_tool)
    
    print(f"Registered {len(registry.list_tools())} tools:")
    for tool_name in registry.list_tools():
        print(f"   - {tool_name}")
    
    # Use tools with an agent
    print("\nAGENT WITH TOOLS")
    print("-"*80)
    
    agent = ReActAgent(
        name="ToolAgent",
        llm_func=mock_llm,
        tools=[weather_tool, stock_tool, calc_tool, text_tool],
        max_iterations=3
    )
    
    print(f"Created agent: {agent.name}")
    print(f"Tools available: {len(agent.tools)}")
    
    # Run agent
    goal = "What's the weather in Tokyo?"
    print(f"\nGoal: {goal}")
    
    result = agent.run(goal)
    
    print("\nRESULTS")
    print("-"*80)
    print(f"Status: {result.status.value}")
    print(f"Output: {result.output}")
    
    # Tool error handling
    print("\nERROR HANDLING")
    print("-"*80)
    
    # Test with invalid input
    error_result = calc_tool.execute(expression="invalid / 0")
    print(f"Test: calculator('invalid / 0')")
    print(f"  Status: {error_result.status.value}")
    print(f"  Error: {error_result.error}")
    
    print("\n" + "="*80)
    print("Tools example completed!")
    print("="*80)
    
    # Summary
    print("\nImportant concepts demonstrated:")
    print("-"*80)
    print("1. Tools can be created using the Tool class or create_tool function")
    print("2. Tools have parameters, descriptions, and examples")
    print("3. Tool execution returns a ToolResult with status and output")
    print("4. ToolRegistry manages multiple tools")
    print("5. Agents can use multiple tools to accomplish goals")
    print("6. Tools handle errors gracefully with ToolStatus")


if __name__ == "__main__":
    main()

