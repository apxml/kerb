"""Reasoning chains and multi-step execution patterns.

This module provides abstractions for building and executing complex
reasoning chains with various execution strategies.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from kerb.core import ChainStrategy
from kerb.core.enums import validate_enum_or_string

# ============================================================================
# Step Data Classes
# ============================================================================


class StepStatus(Enum):
    """Status of step execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result from step execution.

    Attributes:
        output: Step output
        status: Execution status
        error: Error message if failed
        metadata: Additional metadata
    """

    output: Any
    status: StepStatus = StepStatus.COMPLETED
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Check if step was successful."""
        return self.status == StepStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output": self.output,
            "status": self.status.value,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class Step:
    """Represents a single step in a reasoning chain.

    Attributes:
        name: Step name
        func: Function to execute
        description: Step description
        depends_on: Names of steps this depends on
        condition: Optional condition function (returns bool)
        retry_count: Number of times to retry on failure
    """

    name: str
    func: Callable
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[Callable] = None
    retry_count: int = 0

    def execute(self, context: Dict[str, Any] = None) -> StepResult:
        """Execute the step.

        Args:
            context: Execution context with results from previous steps

        Returns:
            StepResult
        """
        context = context or {}

        # Check condition
        if self.condition and not self.condition(context):
            return StepResult(
                output=None,
                status=StepStatus.SKIPPED,
                metadata={"reason": "Condition not met"},
            )

        # Retry logic
        for attempt in range(self.retry_count + 1):
            try:
                output = self.func(context)
                return StepResult(output=output, status=StepStatus.COMPLETED)
            except Exception as e:
                if attempt == self.retry_count:
                    return StepResult(
                        output=None, status=StepStatus.FAILED, error=str(e)
                    )

        return StepResult(output=None, status=StepStatus.FAILED)

    async def execute_async(self, context: Dict[str, Any] = None) -> StepResult:
        """Execute step asynchronously.

        Args:
            context: Execution context

        Returns:
            StepResult
        """
        if asyncio.iscoroutinefunction(self.func):
            context = context or {}

            if self.condition and not self.condition(context):
                return StepResult(
                    output=None,
                    status=StepStatus.SKIPPED,
                    metadata={"reason": "Condition not met"},
                )

            for attempt in range(self.retry_count + 1):
                try:
                    output = await self.func(context)
                    return StepResult(output=output, status=StepStatus.COMPLETED)
                except Exception as e:
                    if attempt == self.retry_count:
                        return StepResult(
                            output=None, status=StepStatus.FAILED, error=str(e)
                        )
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.execute, context)

        return StepResult(output=None, status=StepStatus.FAILED)


# ============================================================================
# Base Chain Class
# ============================================================================


class Chain(ABC):
    """Base class for reasoning chains.

    A chain represents a sequence of steps with a specific execution strategy.
    """

    def __init__(
        self, name: str = "Chain", steps: List[Step] = None, description: str = ""
    ):
        """Initialize chain.

        Args:
            name: Chain name
            steps: List of steps
            description: Chain description
        """
        self.name = name
        self.steps = steps or []
        self.description = description
        self.context: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute the chain.

        Args:
            input_data: Initial input data

        Returns:
            Dictionary with results from all steps
        """
        pass

    def add_step(self, step: Step) -> None:
        """Add a step to the chain.

        Args:
            step: Step to add
        """
        self.steps.append(step)

    def get_step(self, name: str) -> Optional[Step]:
        """Get step by name.

        Args:
            name: Step name

        Returns:
            Step if found
        """
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def clear_context(self) -> None:
        """Clear execution context."""
        self.context.clear()


# ============================================================================
# Sequential Chain
# ============================================================================


class SequentialChain(Chain):
    """Execute steps sequentially, one after another.

    Each step receives the context with results from all previous steps.
    """

    def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute steps sequentially.

        Args:
            input_data: Initial input

        Returns:
            Dictionary with all step results
        """
        self.context = {"input": input_data, "results": {}}

        for step in self.steps:
            result = step.execute(self.context)
            self.context["results"][step.name] = result

            # Make step output available to next steps
            self.context[step.name] = result.output

            # Stop on failure if not configured to continue
            if not result.is_success():
                break

        return self.context


# ============================================================================
# Parallel Chain
# ============================================================================


class ParallelChain(Chain):
    """Execute steps in parallel where possible.

    Steps with dependencies are executed after their dependencies complete.
    Independent steps run concurrently.
    """

    def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute steps in parallel.

        Args:
            input_data: Initial input

        Returns:
            Dictionary with all step results
        """
        self.context = {"input": input_data, "results": {}}

        # Build dependency graph
        completed = set()
        pending = {step.name: step for step in self.steps}

        while pending:
            # Find steps that can run now
            runnable = []
            for name, step in pending.items():
                if all(dep in completed for dep in step.depends_on):
                    runnable.append((name, step))

            if not runnable:
                # Circular dependency or missing dependency
                break

            # Execute runnable steps (simplified parallel execution)
            for name, step in runnable:
                result = step.execute(self.context)
                self.context["results"][name] = result
                self.context[name] = result.output
                completed.add(name)
                del pending[name]

        return self.context

    async def execute_async(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute steps in parallel asynchronously.

        Args:
            input_data: Initial input

        Returns:
            Dictionary with all step results
        """
        self.context = {"input": input_data, "results": {}}

        completed = set()
        pending = {step.name: step for step in self.steps}

        while pending:
            # Find steps that can run now
            runnable = []
            for name, step in pending.items():
                if all(dep in completed for dep in step.depends_on):
                    runnable.append((name, step))

            if not runnable:
                break

            # Execute runnable steps in parallel
            tasks = [step.execute_async(self.context) for _, step in runnable]
            results = await asyncio.gather(*tasks)

            for (name, step), result in zip(runnable, results):
                self.context["results"][name] = result
                self.context[name] = result.output
                completed.add(name)
                del pending[name]

        return self.context


# ============================================================================
# Conditional Chain
# ============================================================================


class ConditionalChain(Chain):
    """Execute different branches based on conditions.

    Allows for if-then-else logic in chain execution.
    """

    def __init__(
        self,
        name: str = "ConditionalChain",
        condition: Optional[Callable] = None,
        true_steps: List[Step] = None,
        false_steps: List[Step] = None,
        description: str = "",
    ):
        """Initialize conditional chain.

        Args:
            name: Chain name
            condition: Condition function (returns bool)
            true_steps: Steps to execute if condition is True
            false_steps: Steps to execute if condition is False
            description: Chain description
        """
        super().__init__(name, [], description)
        self.condition = condition
        self.true_steps = true_steps or []
        self.false_steps = false_steps or []

    def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute conditional chain.

        Args:
            input_data: Initial input

        Returns:
            Dictionary with results
        """
        self.context = {"input": input_data, "results": {}}

        # Evaluate condition
        if self.condition:
            condition_met = self.condition(self.context)
        else:
            condition_met = bool(input_data)

        # Select branch
        steps_to_execute = self.true_steps if condition_met else self.false_steps

        # Execute selected branch
        for step in steps_to_execute:
            result = step.execute(self.context)
            self.context["results"][step.name] = result
            self.context[step.name] = result.output

            if not result.is_success():
                break

        self.context["condition_met"] = condition_met
        return self.context


# ============================================================================
# Loop Chain
# ============================================================================


class LoopChain(Chain):
    """Repeat steps until a condition is met.

    Useful for iterative refinement or until-success patterns.
    """

    def __init__(
        self,
        name: str = "LoopChain",
        steps: List[Step] = None,
        condition: Optional[Callable] = None,
        max_iterations: int = 10,
        description: str = "",
    ):
        """Initialize loop chain.

        Args:
            name: Chain name
            steps: Steps to repeat
            condition: Condition to check (loop continues while True)
            max_iterations: Maximum iterations
            description: Chain description
        """
        super().__init__(name, steps, description)
        self.condition = condition
        self.max_iterations = max_iterations

    def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute loop chain.

        Args:
            input_data: Initial input

        Returns:
            Dictionary with results
        """
        self.context = {"input": input_data, "results": {}, "iterations": []}

        for iteration in range(self.max_iterations):
            iteration_results = {}

            # Execute all steps in iteration
            for step in self.steps:
                result = step.execute(self.context)
                iteration_results[step.name] = result
                self.context[step.name] = result.output

                if not result.is_success():
                    break

            self.context["iterations"].append(iteration_results)

            # Check condition
            if self.condition:
                if not self.condition(self.context):
                    break
            else:
                # Default: check if last step output is truthy
                if self.steps:
                    last_result = iteration_results[self.steps[-1].name]
                    if not last_result.output:
                        break

        self.context["total_iterations"] = len(self.context["iterations"])
        return self.context


# ============================================================================
# Chain Building Functions
# ============================================================================


def create_chain(
    steps: List[Step],
    strategy: Union[ChainStrategy, str] = "sequential",
    name: str = "Chain",
    **kwargs,
) -> Chain:
    """Create a chain with specified strategy.

    Args:
        steps: List of steps
        strategy: Execution strategy (ChainStrategy enum or string: "sequential", "parallel", "conditional", "loop")
        name: Chain name
        **kwargs: Additional chain-specific arguments

    Returns:
        Chain instance

    Examples:
        >>> chain = create_chain(steps, strategy=ChainStrategy.SEQUENTIAL)
    """
    # Validate and normalize strategy
    strategy_val = validate_enum_or_string(strategy, ChainStrategy, "strategy")
    if isinstance(strategy_val, ChainStrategy):
        strategy_str = strategy_val.value
    else:
        strategy_str = strategy_val

    strategy_map = {
        "sequential": SequentialChain,
        "parallel": ParallelChain,
        "conditional": ConditionalChain,
        "loop": LoopChain,
    }

    chain_class = strategy_map.get(strategy_str, SequentialChain)

    if strategy_str == "conditional":
        return chain_class(
            name=name,
            condition=kwargs.get("condition"),
            true_steps=kwargs.get("true_steps", steps),
            false_steps=kwargs.get("false_steps", []),
        )
    elif strategy_str == "loop":
        return chain_class(
            name=name,
            steps=steps,
            condition=kwargs.get("condition"),
            max_iterations=kwargs.get("max_iterations", 10),
        )
    else:
        return chain_class(name=name, steps=steps)


def chain_from_functions(
    functions: List[Callable],
    strategy: Union[ChainStrategy, str] = "sequential",
    name: str = "Chain",
) -> Chain:
    """Create chain from list of functions.

    Args:
        functions: List of functions to chain
        strategy: Execution strategy (ChainStrategy enum or string: "sequential", "parallel", "conditional", "dynamic")
        name: Chain name

    Returns:
        Chain instance

    Examples:
        >>> chain = chain_from_functions([func1, func2], strategy=ChainStrategy.PARALLEL)
    """
    steps = []
    for i, func in enumerate(functions):
        step_name = func.__name__ if hasattr(func, "__name__") else f"step_{i}"
        step = create_step(
            name=step_name,
            func=func,
            description=func.__doc__ or f"Execute {step_name}",
        )
        steps.append(step)

    return create_chain(steps, strategy, name)


def combine_chains(
    chains: List[Chain],
    strategy: Union[ChainStrategy, str] = "sequential",
    name: str = "CombinedChain",
) -> Chain:
    """Combine multiple chains into one.

    Args:
        chains: List of chains to combine
        strategy: How to combine (ChainStrategy enum or string: "sequential" or "parallel")
        name: Combined chain name

    Returns:
        Combined chain

    Examples:
        >>> chain1 = create_chain([step1, step2])
        >>> chain2 = create_chain([step3, step4])
        >>> combined = combine_chains([chain1, chain2], strategy=ChainStrategy.PARALLEL)
    """
    # Flatten all steps from all chains
    all_steps = []
    for chain in chains:
        all_steps.extend(chain.steps)

    return create_chain(all_steps, strategy, name)


# ============================================================================
# Step Creation and Management
# ============================================================================


def create_step(
    name: str,
    func: Callable,
    description: str = "",
    depends_on: List[str] = None,
    condition: Optional[Callable] = None,
    retry_count: int = 0,
) -> Step:
    """Create a step.

    Args:
        name: Step name
        func: Function to execute
        description: Step description
        depends_on: Dependencies
        condition: Execution condition
        retry_count: Number of retries

    Returns:
        Step instance
    """
    return Step(
        name=name,
        func=func,
        description=description,
        depends_on=depends_on or [],
        condition=condition,
        retry_count=retry_count,
    )


def validate_step(step: Step, context: Dict[str, Any] = None) -> bool:
    """Validate that a step can be executed.

    Args:
        step: Step to validate
        context: Execution context

    Returns:
        True if step is valid
    """
    # Check that function is callable
    if not callable(step.func):
        return False

    # Check dependencies are in context
    if context and step.depends_on:
        for dep in step.depends_on:
            if dep not in context:
                return False

    return True


# ============================================================================
# Advanced Chain Patterns
# ============================================================================


class MapReduceChain(Chain):
    """Map-Reduce pattern for parallel processing and aggregation.

    Applies a map function to each input, then reduces results.
    """

    def __init__(
        self,
        name: str = "MapReduceChain",
        map_func: Optional[Callable] = None,
        reduce_func: Optional[Callable] = None,
        description: str = "",
    ):
        """Initialize map-reduce chain.

        Args:
            name: Chain name
            map_func: Function to map over inputs
            reduce_func: Function to reduce results
            description: Chain description
        """
        super().__init__(name, [], description)
        self.map_func = map_func
        self.reduce_func = reduce_func

    def execute(self, input_data: List[Any]) -> Dict[str, Any]:
        """Execute map-reduce.

        Args:
            input_data: List of inputs to process

        Returns:
            Dictionary with mapped and reduced results
        """
        self.context = {"input": input_data, "mapped": [], "reduced": None}

        # Map phase
        if self.map_func:
            for item in input_data:
                try:
                    result = self.map_func(item)
                    self.context["mapped"].append(result)
                except Exception as e:
                    self.context["mapped"].append({"error": str(e)})

        # Reduce phase
        if self.reduce_func and self.context["mapped"]:
            try:
                self.context["reduced"] = self.reduce_func(self.context["mapped"])
            except Exception as e:
                self.context["reduced"] = {"error": str(e)}

        return self.context


class PipelineChain(SequentialChain):
    """Pipeline pattern where each step transforms the output.

    Similar to Unix pipes: step1 | step2 | step3
    """

    def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute pipeline.

        Args:
            input_data: Initial input

        Returns:
            Dictionary with final output
        """
        self.context = {"input": input_data, "results": {}}
        current_output = input_data

        for step in self.steps:
            # Pass previous output as input
            temp_context = dict(self.context)
            temp_context["input"] = current_output

            result = step.execute(temp_context)
            self.context["results"][step.name] = result

            if result.is_success():
                current_output = result.output
                self.context[step.name] = result.output
            else:
                break

        self.context["output"] = current_output
        return self.context


class FanOutFanInChain(Chain):
    """Fan-out to multiple parallel steps, then fan-in to aggregate.

    Useful for processing with multiple strategies then combining results.
    """

    def __init__(
        self,
        name: str = "FanOutFanInChain",
        fan_out_steps: List[Step] = None,
        fan_in_func: Optional[Callable] = None,
        description: str = "",
    ):
        """Initialize fan-out fan-in chain.

        Args:
            name: Chain name
            fan_out_steps: Steps to execute in parallel
            fan_in_func: Function to aggregate results
            description: Chain description
        """
        super().__init__(name, fan_out_steps, description)
        self.fan_in_func = fan_in_func

    def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute fan-out fan-in.

        Args:
            input_data: Input data

        Returns:
            Dictionary with aggregated results
        """
        self.context = {"input": input_data, "results": {}, "outputs": []}

        # Fan-out: execute all steps
        for step in self.steps:
            result = step.execute(self.context)
            self.context["results"][step.name] = result
            if result.is_success():
                self.context["outputs"].append(result.output)

        # Fan-in: aggregate results
        if self.fan_in_func:
            try:
                self.context["aggregated"] = self.fan_in_func(self.context["outputs"])
            except Exception as e:
                self.context["aggregated"] = {"error": str(e)}

        return self.context


class RetryChain(Chain):
    """Chain with automatic retry on failure.

    Retries entire chain or individual steps based on configuration.
    """

    def __init__(
        self,
        name: str = "RetryChain",
        steps: List[Step] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        description: str = "",
    ):
        """Initialize retry chain.

        Args:
            name: Chain name
            steps: Steps to execute
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            description: Chain description
        """
        super().__init__(name, steps, description)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def execute(self, input_data: Any = None) -> Dict[str, Any]:
        """Execute with retry logic.

        Args:
            input_data: Initial input

        Returns:
            Dictionary with results
        """
        import time

        for attempt in range(self.max_retries + 1):
            self.context = {"input": input_data, "results": {}, "attempt": attempt + 1}

            all_success = True
            for step in self.steps:
                result = step.execute(self.context)
                self.context["results"][step.name] = result
                self.context[step.name] = result.output

                if not result.is_success():
                    all_success = False
                    break

            if all_success:
                self.context["retries"] = attempt
                return self.context

            # Wait before retry
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff

        self.context["retries"] = self.max_retries
        return self.context


# ============================================================================
# Chain Utilities
# ============================================================================


def visualize_chain(chain: Chain) -> str:
    """Create a text visualization of a chain.

    Args:
        chain: Chain to visualize

    Returns:
        String visualization
    """
    lines = [f"Chain: {chain.name}"]
    if chain.description:
        lines.append(f"Description: {chain.description}")
    lines.append(f"Steps: {len(chain.steps)}")
    lines.append("")

    for i, step in enumerate(chain.steps):
        lines.append(f"{i+1}. {step.name}")
        if step.description:
            lines.append(f"   {step.description}")
        if step.depends_on:
            lines.append(f"   Depends on: {', '.join(step.depends_on)}")
        lines.append("")

    return "\n".join(lines)


def analyze_chain(chain: Chain) -> Dict[str, Any]:
    """Analyze chain structure and dependencies.

    Args:
        chain: Chain to analyze

    Returns:
        Analysis results
    """
    analysis = {
        "name": chain.name,
        "type": type(chain).__name__,
        "num_steps": len(chain.steps),
        "step_names": [step.name for step in chain.steps],
        "has_dependencies": any(step.depends_on for step in chain.steps),
        "has_conditions": any(step.condition for step in chain.steps),
        "max_retries": max((step.retry_count for step in chain.steps), default=0),
    }

    # Build dependency graph
    dependencies = {}
    for step in chain.steps:
        dependencies[step.name] = step.depends_on
    analysis["dependencies"] = dependencies

    return analysis


def optimize_chain(chain: Chain) -> Chain:
    """Optimize chain execution order.

    For chains with dependencies, reorder steps for optimal execution.

    Args:
        chain: Chain to optimize

    Returns:
        Optimized chain
    """
    # Topological sort for optimal ordering
    if not any(step.depends_on for step in chain.steps):
        return chain  # No dependencies, already optimal

    # Build dependency graph
    graph = {step.name: step.depends_on for step in chain.steps}
    step_map = {step.name: step for step in chain.steps}

    # Topological sort
    sorted_names = []
    visited = set()

    def visit(name: str):
        if name in visited:
            return
        visited.add(name)
        for dep in graph.get(name, []):
            if dep in step_map:
                visit(dep)
        sorted_names.append(name)

    for step in chain.steps:
        visit(step.name)

    # Create new chain with sorted steps
    sorted_steps = [step_map[name] for name in sorted_names]

    # Create new chain of same type
    optimized = type(chain)(
        name=f"{chain.name}_optimized",
        steps=sorted_steps,
        description=chain.description,
    )

    return optimized


def chain_decorator(
    strategy: Union[ChainStrategy, str] = "sequential", name: Optional[str] = None
):
    """Decorator to create a chain from multiple functions.

    Args:
        strategy: Execution strategy (ChainStrategy enum or string)
        name: Chain name

    Examples:
        >>> @chain_decorator(strategy=ChainStrategy.SEQUENTIAL)
        ... class MyChain:
        ...     def step1(self, context):
        ...         return "result1"
        ...
        ...     def step2(self, context):
        ...         return "result2"
    """

    def decorator(cls):
        # Extract methods as steps
        import inspect

        steps = []
        for name_attr, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name_attr.startswith("_"):
                step = create_step(
                    name=name_attr, func=method, description=method.__doc__ or ""
                )
                steps.append(step)

        # Create chain
        chain_name = name or cls.__name__
        return create_chain(steps, strategy, chain_name)

    return decorator
