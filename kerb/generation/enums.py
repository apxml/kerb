"""Enums and constants for LLM generation."""

from enum import Enum
from typing import Dict, Tuple


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


class ModelName(Enum):
    """Supported model names with their actual string identifiers."""

    # OpenAI
    GPT_5_2 = "gpt-5.2"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O3 = "o3"
    O3_MINI = "o3-mini"
    O1 = "o1"
    O1_MINI = "o1-mini"

    # Anthropic
    CLAUDE_45_OPUS = "claude-4.5-opus-20260212"
    CLAUDE_45_SONNET = "claude-4.5-sonnet-20260212"
    CLAUDE_45_HAIKU = "claude-4.5-haiku-20260212"
    CLAUDE_OPUS_4 = "claude-opus-4"
    CLAUDE_SONNET_4 = "claude-sonnet-4"
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_35_HAIKU = "claude-3-5-haiku-20241022"

    # Google
    GEMINI_3_PRO = "gemini-3-pro"
    GEMINI_3_FLASH = "gemini-3-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"


MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI - prices in $ per million tokens (input, output)
    "gpt-5.2": (2.00, 12.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-mini": (0.150, 0.600),
    "o3": (20.00, 80.00),
    "o3-mini": (1.10, 4.40),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    # Anthropic
    "claude-4.5-opus-20260212": (18.00, 90.00),
    "claude-4.5-sonnet-20260212": (4.00, 20.00),
    "claude-4.5-haiku-20260212": (1.00, 5.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    # Google
    "gemini-3-pro": (1.50, 12.00),
    "gemini-3-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.075, 0.30),
}


def get_pricing(model: ModelName) -> Tuple[float, float]:
    """Return (input_cost, output_cost) for a model, or raise KeyError if unknown."""
    key = model.value
    if key not in MODEL_PRICING:
        raise KeyError(f"No pricing defined for model {key}")
    return MODEL_PRICING[key]
