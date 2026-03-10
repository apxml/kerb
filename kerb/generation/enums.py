"""Enums and constants for LLM generation."""

from enum import Enum
from typing import Dict, Tuple


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


class ReasoningLevel(Enum):
    """Universal reasoning effort levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelName(Enum):
    """Supported model names with their actual string identifiers."""

    # OpenAI
    GPT_5_4 = "gpt-5.4"
    GPT_5_4_PRO = "gpt-5.4-pro"
    GPT_5_3 = "gpt-5.3"
    GPT_5_2 = "gpt-5.2"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O3_PRO = "o3-pro"
    O3 = "o3"
    O3_MINI = "o3-mini"
    O1 = "o1"
    O1_PRO = "o1-pro"
    O1_MINI = "o1-mini"
    O4_MINI = "o4-mini"

    # Anthropic
    CLAUDE_OPUS_4_6 = "claude-opus-4-6"
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"
    CLAUDE_45_OPUS = "claude-4.5-opus-20260212"
    CLAUDE_45_SONNET = "claude-4.5-sonnet-20260212"
    CLAUDE_45_HAIKU = "claude-4.5-haiku-20260212"
    CLAUDE_OPUS_4 = "claude-opus-4"
    CLAUDE_SONNET_4 = "claude-sonnet-4"
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_35_HAIKU = "claude-3-5-haiku-20241022"

    # Google
    GEMINI_3_1_PRO = "gemini-3.1-pro-preview"
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite-preview"
    GEMINI_3_PRO = "gemini-3-pro"
    GEMINI_3_FLASH = "gemini-3-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"


MODEL_PRICING: Dict[ModelName, Tuple[float, float]] = {
    # OpenAI - prices in $ per million tokens (input, output)
    ModelName.GPT_5_4: (2.50, 15.00),
    ModelName.GPT_5_4_PRO: (30.00, 180.00),
    ModelName.GPT_5_3: (1.75, 14.00),
    ModelName.GPT_5_2: (1.75, 14.00),
    ModelName.GPT_5: (1.25, 10.00),
    ModelName.GPT_5_MINI: (0.25, 2.00),
    ModelName.GPT_5_NANO: (0.05, 0.40),
    ModelName.GPT_4_1: (2.00, 8.00),
    ModelName.GPT_4_1_MINI: (0.40, 1.60),
    ModelName.GPT_4_1_NANO: (0.10, 0.40),
    ModelName.GPT_4O: (2.50, 10.00),
    ModelName.GPT_4O_MINI: (0.15, 0.60),
    ModelName.O1: (15.00, 60.00),
    ModelName.O1_PRO: (150.00, 600.00),
    ModelName.O1_MINI: (1.10, 4.40),
    ModelName.O3_PRO: (20.00, 80.00),
    ModelName.O3: (2.00, 8.00),
    ModelName.O3_MINI: (1.10, 4.40),
    ModelName.O4_MINI: (1.10, 4.40),
    # Anthropic
    ModelName.CLAUDE_OPUS_4_6: (5.00, 25.00),
    ModelName.CLAUDE_SONNET_4_6: (3.00, 15.00),
    ModelName.CLAUDE_HAIKU_4_5: (1.00, 5.00),
    ModelName.CLAUDE_45_OPUS: (18.00, 90.00),
    ModelName.CLAUDE_45_SONNET: (4.00, 20.00),
    ModelName.CLAUDE_45_HAIKU: (1.00, 5.00),
    ModelName.CLAUDE_OPUS_4: (15.00, 75.00),
    ModelName.CLAUDE_SONNET_4: (3.00, 15.00),
    ModelName.CLAUDE_35_SONNET: (3.00, 15.00),
    ModelName.CLAUDE_35_HAIKU: (0.80, 4.00),
    # Google
    ModelName.GEMINI_3_1_PRO: (2.00, 12.00),
    ModelName.GEMINI_3_1_FLASH_LITE: (0.25, 1.50),
    ModelName.GEMINI_3_PRO: (2.00, 12.00),
    ModelName.GEMINI_3_FLASH: (0.50, 3.00),
    ModelName.GEMINI_2_5_PRO: (1.25, 10.00),
    ModelName.GEMINI_2_5_FLASH: (0.30, 2.50),
    ModelName.GEMINI_2_5_FLASH_LITE: (0.10, 0.40),
}


def get_pricing(model: ModelName) -> Tuple[float, float]:
    """Return (input_cost, output_cost) for a model, or raise KeyError if unknown."""
    if model not in MODEL_PRICING:
        raise KeyError(f"No pricing defined for model {model.value}")
    return MODEL_PRICING[model]
