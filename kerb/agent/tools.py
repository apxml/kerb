"""Tool and function calling abstractions for agents.

This module provides comprehensive tool management and execution utilities
for LLM-based agents to interact with external functions and APIs.
"""

import asyncio
import inspect
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union, get_type_hints)

if TYPE_CHECKING:
    from kerb.core.enums import ToolResultFormat


# ============================================================================
# Tool Data Classes
# ============================================================================


class ToolStatus(Enum):
    """Status of tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


@dataclass
class ToolResult:
    """Result from tool execution.

    Attributes:
        output: Tool output
        status: Execution status
        error: Error message if failed
        metadata: Additional result metadata
    """

    output: Any
    status: ToolStatus = ToolStatus.SUCCESS
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output": self.output,
            "status": self.status.value,
            "error": self.error,
            "metadata": self.metadata,
        }

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolStatus.SUCCESS


@dataclass
class Tool:
    """Represents a tool/function that an agent can use.

    Attributes:
        name: Tool name
        description: What the tool does
        func: The actual function to execute
        parameters: Parameter descriptions
        returns: Return value description
        examples: Usage examples
    """

    name: str
    description: str
    func: Callable
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    returns: str = ""
    examples: List[str] = field(default_factory=list)

    def execute(self, *args, **kwargs) -> ToolResult:
        """Execute the tool.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            ToolResult with output or error
        """
        try:
            output = self.func(*args, **kwargs)
            return ToolResult(output=output, status=ToolStatus.SUCCESS)
        except TypeError as e:
            return ToolResult(
                output=None,
                status=ToolStatus.INVALID_INPUT,
                error=f"Invalid input: {str(e)}",
            )
        except Exception as e:
            return ToolResult(output=None, status=ToolStatus.ERROR, error=str(e))

    async def execute_async(self, *args, **kwargs) -> ToolResult:
        """Execute tool asynchronously.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            ToolResult
        """
        if asyncio.iscoroutinefunction(self.func):
            try:
                output = await self.func(*args, **kwargs)
                return ToolResult(output=output, status=ToolStatus.SUCCESS)
            except Exception as e:
                return ToolResult(output=None, status=ToolStatus.ERROR, error=str(e))
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.execute, *args, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "examples": self.examples,
        }

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format.

        Returns:
            Dictionary in OpenAI function format
        """
        properties = {}
        required = []

        for param_name, param_info in self.parameters.items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }

            if param_info.get("enum"):
                properties[param_name]["enum"] = param_info["enum"]

            if param_info.get("required", True):
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format.

        Returns:
            Dictionary in Anthropic tool format
        """
        input_schema = {"type": "object", "properties": {}, "required": []}

        for param_name, param_info in self.parameters.items():
            input_schema["properties"][param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }

            if param_info.get("required", True):
                input_schema["required"].append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema,
        }

    def __str__(self) -> str:
        """String representation."""
        return f"Tool({self.name}): {self.description}"

    def __repr__(self) -> str:
        """Repr."""
        return self.__str__()


# ============================================================================
# Tool Registry
# ============================================================================


class ToolRegistry:
    """Registry for managing tools.

    Provides centralized tool management, lookup, and organization.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(
        self, tool: Tool, category: Optional[str] = None, overwrite: bool = False
    ) -> None:
        """Register a tool.

        Args:
            tool: Tool to register
            category: Optional category for organization
            overwrite: Whether to overwrite existing tool

        Raises:
            ValueError: If tool already exists and overwrite=False
        """
        if tool.name in self._tools and not overwrite:
            raise ValueError(
                f"Tool '{tool.name}' already registered. Use overwrite=True to replace."
            )

        self._tools[tool.name] = tool

        if category:
            if category not in self._categories:
                self._categories[category] = []
            if tool.name not in self._categories[category]:
                self._categories[category].append(tool.name)

    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool if found, None otherwise
        """
        return self._tools.get(name)

    def get_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a category.

        Args:
            category: Category name

        Returns:
            List of tools in category
        """
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def list_categories(self) -> List[str]:
        """List all categories.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def search(self, query: str) -> List[Tool]:
        """Search tools by name or description.

        Args:
            query: Search query

        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        matches = []

        for tool in self._tools.values():
            if (
                query_lower in tool.name.lower()
                or query_lower in tool.description.lower()
            ):
                matches.append(tool)

        return matches

    def remove(self, name: str) -> bool:
        """Remove tool from registry.

        Args:
            name: Tool name

        Returns:
            True if removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]

            # Remove from categories
            for category_tools in self._categories.values():
                if name in category_tools:
                    category_tools.remove(name)

            return True
        return False

    def clear(self) -> None:
        """Clear all tools from registry."""
        self._tools.clear()
        self._categories.clear()

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """Export all tools in OpenAI function format.

        Returns:
            List of tool definitions
        """
        return [tool.to_openai_function() for tool in self._tools.values()]

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Export all tools in Anthropic format.

        Returns:
            List of tool definitions
        """
        return [tool.to_anthropic_tool() for tool in self._tools.values()]

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __iter__(self):
        """Iterate over tools."""
        return iter(self._tools.values())


# Global registry instance
# Users can import and use this explicitly: from kerb.agent.tools import global_tool_registry
_global_registry = ToolRegistry()
global_tool_registry = _global_registry  # Explicit export for users


# ============================================================================
# Tool Creation Functions
# ============================================================================


def create_tool(
    name: str,
    func: Callable,
    description: str = "",
    parameters: Dict[str, Dict[str, Any]] = None,
    returns: str = "",
    examples: List[str] = None,
    auto_parse: bool = True,
) -> Tool:
    """Create a tool from a function.

    Args:
        name: Tool name
        func: Function to wrap
        description: Tool description
        parameters: Parameter specifications
        returns: Return value description
        examples: Usage examples
        auto_parse: Auto-parse parameters from function signature

    Returns:
        Tool instance
    """
    if not description:
        description = func.__doc__ or f"Execute {name}"

    if auto_parse and parameters is None:
        parameters = _parse_function_parameters(func)

    return Tool(
        name=name,
        description=description,
        func=func,
        parameters=parameters or {},
        returns=returns,
        examples=examples or [],
    )


def _parse_function_parameters(func: Callable) -> Dict[str, Dict[str, Any]]:
    """Parse parameters from function signature.

    Args:
        func: Function to parse

    Returns:
        Parameter specifications
    """
    parameters = {}
    sig = inspect.signature(func)
    type_hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_info = {
            "description": f"Parameter {param_name}",
            "required": param.default == inspect.Parameter.empty,
        }

        # Get type hint
        if param_name in type_hints:
            param_type = type_hints[param_name]
            param_info["type"] = _python_type_to_json_type(param_type)
        else:
            param_info["type"] = "string"

        # Get default value
        if param.default != inspect.Parameter.empty:
            param_info["default"] = param.default

        parameters[param_name] = param_info

    return parameters


def _python_type_to_json_type(python_type: type) -> str:
    """Convert Python type to JSON schema type.

    Args:
        python_type: Python type

    Returns:
        JSON schema type string
    """
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    # Handle Optional types
    if hasattr(python_type, "__origin__"):
        origin = python_type.__origin__
        if origin is Union:
            # For Optional[X], return type of X
            args = python_type.__args__
            non_none_types = [t for t in args if t != type(None)]
            if non_none_types:
                return _python_type_to_json_type(non_none_types[0])
        return type_mapping.get(origin, "string")

    return type_mapping.get(python_type, "string")


# ============================================================================
# Tool Registration Functions
# ============================================================================


def register_tool(
    tool: Tool, category: Optional[str] = None, registry: Optional[ToolRegistry] = None
) -> None:
    """Register a tool in the registry.

    Args:
        tool: Tool to register
        category: Optional category
        registry: Registry to use. If None, uses the global registry.
                 Import global registry explicitly: `from kerb.agent.tools import global_tool_registry`

    Examples:
        >>> # Using global registry (implicit)
        >>> register_tool(my_tool)

        >>> # Using global registry (explicit)
        >>> from kerb.agent.tools import global_tool_registry
        >>> register_tool(my_tool, registry=global_tool_registry)

        >>> # Using custom registry
        >>> my_registry = ToolRegistry()
        >>> register_tool(my_tool, registry=my_registry)
    """
    target_registry = registry or _global_registry
    target_registry.register(tool, category)


def get_tool(name: str, registry: Optional[ToolRegistry] = None) -> Optional[Tool]:
    """Get tool from registry.

    Args:
        name: Tool name
        registry: Registry to use. If None, uses the global registry.
                 Import global registry explicitly: `from kerb.agent.tools import global_tool_registry`

    Returns:
        Tool if found

    Examples:
        >>> # Using global registry (implicit)
        >>> tool = get_tool("calculator")

        >>> # Using custom registry
        >>> tool = get_tool("calculator", registry=my_registry)
    """
    target_registry = registry or _global_registry
    return target_registry.get(name)


def tool_decorator(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
):
    """Decorator to register a function as a tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        category: Tool category
        registry: Registry to use. If None, uses the global registry.
                 Import global registry explicitly: `from kerb.agent.tools import global_tool_registry`

    Examples:
        >>> # Using global registry (implicit)
        >>> @tool_decorator(category="math")
        ... def add(a: int, b: int) -> int:
        ...     '''Add two numbers.'''
        ...     return a + b

        >>> # Using custom registry (explicit)
        >>> my_registry = ToolRegistry()
        >>> @tool_decorator(category="math", registry=my_registry)
        ... def subtract(a: int, b: int) -> int:
        ...     '''Subtract two numbers.'''
        ...     return a - b
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Execute {tool_name}"

        tool = create_tool(name=tool_name, func=func, description=tool_description)

        register_tool(tool, category, registry)

        # Return original function
        return func

    return decorator


# ============================================================================
# Tool Execution Functions
# ============================================================================


def execute_tool(
    tool: Union[Tool, str], *args, registry: Optional[ToolRegistry] = None, **kwargs
) -> ToolResult:
    """Execute a tool.

    Args:
        tool: Tool instance or name (string)
        *args: Positional arguments to pass to tool
        registry: Registry to lookup tool (if tool name provided). If None, uses global registry.
                 Import global registry explicitly: `from kerb.agent.tools import global_tool_registry`
        **kwargs: Keyword arguments to pass to tool

    Returns:
        ToolResult: Result of tool execution

    Examples:
        >>> # Execute by tool instance
        >>> result = execute_tool(my_tool, arg1, arg2)

        >>> # Execute by name (using global registry)
        >>> result = execute_tool("calculator", operation="add", a=5, b=3)

        >>> # Execute by name (using custom registry)
        >>> result = execute_tool("calculator", operation="add", a=5, b=3, registry=my_registry)
    """
    if isinstance(tool, str):
        target_registry = registry or _global_registry
        tool_obj = target_registry.get(tool)
        if not tool_obj:
            return ToolResult(
                output=None,
                status=ToolStatus.ERROR,
                error=f"Tool '{tool}' not found in registry",
            )
        tool = tool_obj

    # Validate tool call arguments before execution
    is_valid, error_msg = validate_tool_call(tool, args=args, kwargs=kwargs)
    if not is_valid:
        return ToolResult(output=None, status=ToolStatus.INVALID_INPUT, error=error_msg)

    return tool.execute(*args, **kwargs)


async def execute_tool_async(
    tool: Union[Tool, str], *args, registry: Optional[ToolRegistry] = None, **kwargs
) -> ToolResult:
    """Execute tool asynchronously.

    Args:
        tool: Tool instance or name
        *args: Positional arguments
        registry: Registry to lookup tool
        **kwargs: Keyword arguments

    Returns:
        ToolResult
    """
    if isinstance(tool, str):
        target_registry = registry or _global_registry
        tool_obj = target_registry.get(tool)
        if not tool_obj:
            return ToolResult(
                output=None, status=ToolStatus.ERROR, error=f"Tool '{tool}' not found"
            )
        tool = tool_obj

    # Validate tool call arguments before execution
    is_valid, error_msg = validate_tool_call(tool, args=args, kwargs=kwargs)
    if not is_valid:
        return ToolResult(output=None, status=ToolStatus.INVALID_INPUT, error=error_msg)

    return await tool.execute_async(*args, **kwargs)


def batch_execute_tools(
    tool_calls: List[Tuple[Union[Tool, str], tuple, dict]],
    registry: Optional[ToolRegistry] = None,
    continue_on_error: bool = True,
) -> List[ToolResult]:
    """Execute multiple tools in batch.

    Args:
        tool_calls: List of (tool, args, kwargs) tuples
        registry: Registry for tool lookup
        continue_on_error: Continue if a tool fails

    Returns:
        List of ToolResults
    """
    results = []

    for tool, args, kwargs in tool_calls:
        result = execute_tool(tool, *args, registry=registry, **kwargs)
        results.append(result)

        if not continue_on_error and not result.is_success():
            break

    return results


def parallel_execute_tools(
    tool_calls: List[Tuple[Union[Tool, str], tuple, dict]],
    registry: Optional[ToolRegistry] = None,
    max_workers: int = 5,
) -> List[ToolResult]:
    """Execute multiple tools in parallel.

    Args:
        tool_calls: List of (tool, args, kwargs) tuples
        registry: Registry for tool lookup
        max_workers: Maximum parallel workers

    Returns:
        List of ToolResults in order
    """
    results = [None] * len(tool_calls)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {}
        for i, (tool, args, kwargs) in enumerate(tool_calls):
            future = executor.submit(
                execute_tool, tool, *args, registry=registry, **kwargs
            )
            future_to_index[future] = i

        # Collect results
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    return results


# ============================================================================
# Tool Call Parsing and Validation
# ============================================================================


def validate_tool_call(
    tool: Tool, args: tuple = None, kwargs: dict = None
) -> Tuple[bool, Optional[str]]:
    """Validate tool call arguments.

    Args:
        tool: Tool to validate
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Tuple of (is_valid, error_message)
    """
    kwargs = kwargs or {}

    # Check required parameters
    for param_name, param_info in tool.parameters.items():
        if param_info.get("required", True):
            if param_name not in kwargs:
                return False, f"Missing required parameter: {param_name}"

    # Check parameter types (basic validation)
    for param_name, value in kwargs.items():
        if param_name not in tool.parameters:
            continue

        expected_type = tool.parameters[param_name].get("type")
        if expected_type:
            if not _validate_type(value, expected_type):
                return (
                    False,
                    f"Parameter '{param_name}' has invalid type. Expected {expected_type}",
                )

    return True, None


def _validate_type(value: Any, expected_type: str) -> bool:
    """Validate value against expected type.

    Args:
        value: Value to validate
        expected_type: Expected JSON schema type

    Returns:
        True if valid
    """
    type_checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, (list, tuple)),
        "object": lambda v: isinstance(v, dict),
        "null": lambda v: v is None,
    }

    check = type_checks.get(expected_type)
    return check(value) if check else True


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from text.

    Supports multiple formats:
    - Action: tool_name
      Action Input: {"arg": "value"}
    - tool_name(arg1, arg2)
    - {"tool": "tool_name", "input": {...}}

    Args:
        text: Text containing tool calls

    Returns:
        List of parsed tool calls
    """
    tool_calls = []

    # Try JSON format first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "tool" in data:
            tool_calls.append({"name": data["tool"], "input": data.get("input", {})})
            return tool_calls
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "tool" in item:
                    tool_calls.append(
                        {"name": item["tool"], "input": item.get("input", {})}
                    )
            if tool_calls:
                return tool_calls
    except json.JSONDecodeError:
        pass

    # Try Action/Action Input format
    action_pattern = r"Action:\s*(.+?)(?:\n|$)"
    input_pattern = r"Action Input:\s*(.+?)(?:\n\n|\n(?=[A-Z])|$)"

    action_matches = list(re.finditer(action_pattern, text, re.IGNORECASE))
    input_matches = list(re.finditer(input_pattern, text, re.IGNORECASE | re.DOTALL))

    for i, action_match in enumerate(action_matches):
        tool_name = action_match.group(1).strip()
        tool_input = {}

        if i < len(input_matches):
            input_text = input_matches[i].group(1).strip()
            try:
                tool_input = json.loads(input_text)
            except json.JSONDecodeError:
                tool_input = {"input": input_text}

        tool_calls.append({"name": tool_name, "input": tool_input})

    if tool_calls:
        return tool_calls

    # Try function call format: tool_name(arg1, arg2)
    func_pattern = r"(\w+)\((.*?)\)"
    for match in re.finditer(func_pattern, text):
        tool_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments
        args = []
        kwargs = {}

        if args_str:
            # Simple parsing (doesn't handle nested parentheses)
            parts = [p.strip() for p in args_str.split(",")]
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    kwargs[key.strip()] = _parse_value(value.strip())
                else:
                    args.append(_parse_value(part))

        tool_calls.append(
            {
                "name": tool_name,
                "input": {"args": args, "kwargs": kwargs} if args else kwargs,
            }
        )

    return tool_calls


def _parse_value(value_str: str) -> Any:
    """Parse a value from string.

    Args:
        value_str: String representation of value

    Returns:
        Parsed value
    """
    value_str = value_str.strip()

    # Try JSON
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        pass

    # Remove quotes if present
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]

    # Try number
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Boolean
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # Return as string
    return value_str


def format_tool_result(
    result: ToolResult, format: Union["ToolResultFormat", str] = "text"
) -> str:
    """Format tool result for agent consumption.

    Args:
        result: Tool result
        format: Output format (ToolResultFormat enum or string: "text", "json", "markdown", "html")

    Returns:
        Formatted result string

    Examples:
        >>> from kerb.core.enums import ToolResultFormat
        >>> formatted = format_tool_result(result, format=ToolResultFormat.JSON)
    """
    from kerb.core.enums import ToolResultFormat, validate_enum_or_string

    # Validate and normalize format
    format_val = validate_enum_or_string(format, ToolResultFormat, "format")
    if isinstance(format_val, ToolResultFormat):
        format_str = format_val.value
    else:
        format_str = format_val

    if format_str == "json":
        return json.dumps(result.to_dict(), indent=2)

    elif format_str == "markdown":
        lines = ["### Tool Result\n"]
        lines.append(f"**Status:** {result.status.value}\n")
        if result.error:
            lines.append(f"**Error:** {result.error}\n")
        lines.append(f"**Output:**\n```\n{result.output}\n```")
        return "\n".join(lines)

    elif format_str == "html":
        html = f"<div class='tool-result'>"
        html += f"<h3>Tool Result</h3>"
        html += f"<p><strong>Status:</strong> {result.status.value}</p>"
        if result.error:
            html += f"<p><strong>Error:</strong> {result.error}</p>"
        html += f"<pre>{result.output}</pre>"
        html += "</div>"
        return html

    else:  # text
        if result.is_success():
            return f"Result: {result.output}"
        else:
            return f"Error ({result.status.value}): {result.error}"


# ============================================================================
# Common Tool Implementations
# ============================================================================


def create_calculator_tool() -> Tool:
    """Create a basic calculator tool.

    Returns:
        Calculator tool
    """

    def calculator(expression: str) -> Union[float, str]:
        """Evaluate a mathematical expression.

        Args:
            expression: Math expression to evaluate

        Returns:
            Result of evaluation
        """
        try:
            # Safe eval with restricted namespace
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return float(result)
        except Exception as e:
            return f"Error: {str(e)}"

    return create_tool(
        name="calculator",
        func=calculator,
        description="Evaluate mathematical expressions",
        parameters={
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate",
                "required": True,
            }
        },
        examples=[
            "calculator('2 + 2')",
            "calculator('pow(2, 8)')",
            "calculator('(10 + 5) * 3')",
        ],
    )


def create_search_tool(search_func: Callable[[str], List[str]]) -> Tool:
    """Create a search tool.

    Args:
        search_func: Function that performs search

    Returns:
        Search tool
    """
    return create_tool(
        name="search",
        func=search_func,
        description="Search for information",
        parameters={
            "query": {"type": "string", "description": "Search query", "required": True}
        },
    )


def create_python_repl_tool() -> Tool:
    """Create a Python REPL tool.

    Returns:
        Python REPL tool
    """

    def python_repl(code: str) -> str:
        """Execute Python code.

        Args:
            code: Python code to execute

        Returns:
            Output from execution
        """
        try:
            # Capture output
            import io
            import sys

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            # Execute code
            exec(code)

            # Get output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            return output if output else "Code executed successfully"
        except Exception as e:
            return f"Error: {str(e)}"

    return create_tool(
        name="python_repl",
        func=python_repl,
        description="Execute Python code and return output",
        parameters={
            "code": {
                "type": "string",
                "description": "Python code to execute",
                "required": True,
            }
        },
        examples=[
            "python_repl('print(2 + 2)')",
            "python_repl('x = [1, 2, 3]\\nprint(sum(x))')",
        ],
    )


def create_web_scraper_tool(scraper_func: Optional[Callable] = None) -> Tool:
    """Create a web scraping tool.

    Args:
        scraper_func: Custom scraper function

    Returns:
        Web scraper tool
    """

    def default_scraper(url: str) -> str:
        """Scrape content from URL.

        Args:
            url: URL to scrape

        Returns:
            Page content
        """
        try:
            import urllib.request

            with urllib.request.urlopen(url) as response:
                return response.read().decode("utf-8")
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"

    func = scraper_func or default_scraper

    return create_tool(
        name="web_scraper",
        func=func,
        description="Scrape content from a web page",
        parameters={
            "url": {"type": "string", "description": "URL to scrape", "required": True}
        },
    )


# ============================================================================
# Tool Chain Utilities
# ============================================================================


def chain_tools(tools: List[Tool], initial_input: Any) -> ToolResult:
    """Chain multiple tools together.

    The output of each tool becomes the input to the next.

    Args:
        tools: List of tools to chain
        initial_input: Initial input to first tool

    Returns:
        Final tool result
    """
    current_input = initial_input

    for tool in tools:
        result = tool.execute(current_input)
        if not result.is_success():
            return result
        current_input = result.output

    return ToolResult(output=current_input, status=ToolStatus.SUCCESS)


def conditional_tool_execution(
    condition: bool, true_tool: Tool, false_tool: Tool, input_data: Any
) -> ToolResult:
    """Execute tool based on condition.

    Args:
        condition: Condition to check
        true_tool: Tool to execute if True
        false_tool: Tool to execute if False
        input_data: Input for tool

    Returns:
        Tool result
    """
    tool = true_tool if condition else false_tool
    return tool.execute(input_data)


def retry_tool_execution(
    tool: Tool, *args, max_retries: int = 3, **kwargs
) -> ToolResult:
    """Retry tool execution on failure.

    Args:
        tool: Tool to execute
        *args: Tool arguments
        max_retries: Maximum retry attempts
        **kwargs: Tool keyword arguments

    Returns:
        Tool result
    """
    last_result = None

    for attempt in range(max_retries):
        result = tool.execute(*args, **kwargs)
        if result.is_success():
            return result
        last_result = result

    return last_result


# ============================================================================
# Tool Discovery
# ============================================================================


def discover_tools_from_module(module) -> List[Tool]:
    """Discover tools from a module.

    Finds all functions in module and converts them to tools.

    Args:
        module: Python module to scan

    Returns:
        List of discovered tools
    """
    tools = []

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and not name.startswith("_"):
            tool = create_tool(
                name=name, func=obj, description=obj.__doc__ or f"Function {name}"
            )
            tools.append(tool)

    return tools


def discover_tools_from_class(cls) -> List[Tool]:
    """Discover tools from a class.

    Converts all public methods to tools.

    Args:
        cls: Class to scan

    Returns:
        List of discovered tools
    """
    tools = []
    instance = cls() if inspect.isclass(cls) else cls

    for name, obj in inspect.getmembers(instance):
        if (
            inspect.ismethod(obj)
            and not name.startswith("_")
            and name not in ["__init__", "__new__"]
        ):
            tool = create_tool(
                name=f"{cls.__name__}.{name}",
                func=obj,
                description=obj.__doc__ or f"Method {name}",
            )
            tools.append(tool)

    return tools
