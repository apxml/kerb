"""Mock LLM providers for testing."""

import random
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from .types import MockBehavior, MockResponse


class MockLLM:
    """Mock LLM provider with configurable responses.

    This class provides a drop-in replacement for real LLM providers,
    useful for testing without making actual API calls.
    """

    def __init__(
        self,
        responses: Optional[Union[str, List[str], Dict[str, str]]] = None,
        behavior: MockBehavior = MockBehavior.FIXED,
        default_response: str = "Mock response",
        latency: float = 0.1,
        token_calculator: Optional[Callable[[str], int]] = None,
    ):
        """Initialize mock LLM.

        Args:
            responses: Response(s) to return
            behavior: Behavior mode for returning responses
            default_response: Default response when no match found
            latency: Simulated latency per response
            token_calculator: Function to calculate token counts
        """
        self.behavior = behavior
        self.default_response = default_response
        self.latency = latency
        self.token_calculator = token_calculator or self._simple_token_count
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []

        # Configure responses based on behavior
        if isinstance(responses, str):
            self.responses = [responses]
        elif isinstance(responses, list):
            self.responses = responses
        elif isinstance(responses, dict):
            self.pattern_responses = responses
            self.responses = []
        else:
            self.responses = [default_response]

        self.current_index = 0

    def generate(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs
    ) -> MockResponse:
        """Generate a mock response.

        Args:
            prompt: Input prompt (string or message list)
            **kwargs: Additional generation parameters (ignored)

        Returns:
            MockResponse object
        """
        self.call_count += 1

        # Extract text from prompt
        if isinstance(prompt, list):
            prompt_text = " ".join([msg.get("content", "") for msg in prompt])
        else:
            prompt_text = prompt

        # Record call
        self.call_history.append(
            {
                "prompt": prompt_text,
                "kwargs": kwargs,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Generate response based on behavior
        if self.behavior == MockBehavior.FIXED:
            content = self.responses[0] if self.responses else self.default_response
        elif self.behavior == MockBehavior.SEQUENTIAL:
            content = self.responses[self.current_index % len(self.responses)]
            self.current_index += 1
        elif self.behavior == MockBehavior.RANDOM:
            content = (
                random.choice(self.responses)
                if self.responses
                else self.default_response
            )
        elif self.behavior == MockBehavior.PATTERN:
            content = self._match_pattern(prompt_text)
        else:
            content = self.default_response

        # Simulate latency
        time.sleep(self.latency)

        return MockResponse(
            content=content,
            prompt_tokens=self.token_calculator(prompt_text),
            completion_tokens=self.token_calculator(content),
            latency=self.latency,
            metadata={"call_count": self.call_count},
        )

    def _match_pattern(self, prompt: str) -> str:
        """Match prompt against patterns and return response."""
        for pattern, response in self.pattern_responses.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                return response
        return self.default_response

    def _simple_token_count(self, text: str) -> int:
        """Simple token count estimation."""
        return len(text.split())

    def reset(self) -> None:
        """Reset call count and history."""
        self.call_count = 0
        self.call_history = []
        self.current_index = 0

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get the last call made to the mock."""
        return self.call_history[-1] if self.call_history else None

    def assert_called(self) -> None:
        """Assert that the mock was called at least once."""
        assert self.call_count > 0, "Mock LLM was not called"

    def assert_called_with(self, prompt_contains: str) -> None:
        """Assert that the mock was called with a prompt containing text."""
        for call in self.call_history:
            if prompt_contains in call["prompt"]:
                return
        raise AssertionError(
            f"Mock LLM was not called with prompt containing: {prompt_contains}"
        )


class MockStreamingLLM:
    """Mock streaming LLM for testing streaming responses."""

    def __init__(
        self, response: str, chunk_size: int = 10, delay_per_chunk: float = 0.01
    ):
        """Initialize mock streaming LLM.

        Args:
            response: Full response to stream
            chunk_size: Characters per chunk
            delay_per_chunk: Delay between chunks in seconds
        """
        self.response = response
        self.chunk_size = chunk_size
        self.delay_per_chunk = delay_per_chunk

    def generate_stream(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs
    ) -> Iterator[str]:
        """Generate streaming mock response.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (ignored)

        Yields:
            Response chunks
        """
        for i in range(0, len(self.response), self.chunk_size):
            chunk = self.response[i : i + self.chunk_size]
            time.sleep(self.delay_per_chunk)
            yield chunk


def create_mock_llm(
    responses: Union[str, List[str], Dict[str, str]],
    behavior: MockBehavior = MockBehavior.FIXED,
    **kwargs,
) -> MockLLM:
    """Helper to create a mock LLM instance.

    Args:
        responses: Response(s) to configure
        behavior: Behavior mode
        **kwargs: Additional MockLLM parameters

    Returns:
        Configured MockLLM instance
    """
    return MockLLM(responses=responses, behavior=behavior, **kwargs)
