"""Tokenizer utility functions for token estimation and cost calculation.

This module provides utility functions for:
- Converting between tokens and characters
- Estimating API costs based on token usage
- Optimizing token usage in applications
"""

from typing import Optional

from .tokenizer import Tokenizer


def tokens_to_chars(
    token_count: int, tokenizer: Tokenizer = Tokenizer.CL100K_BASE
) -> int:
    """Estimate character count from token count.

    Args:
        token_count (int): Number of tokens
        tokenizer (Tokenizer): Tokenizer for estimation. Defaults to Tokenizer.CL100K_BASE.

    Returns:
        int: Estimated character count

    Examples:
        >>> tokens_to_chars(100, tokenizer=Tokenizer.CL100K_BASE)
        400
    """
    # Tokenizer-specific character-to-token ratios
    if tokenizer in [
        Tokenizer.CL100K_BASE,
        Tokenizer.P50K_BASE,
        Tokenizer.R50K_BASE,
        Tokenizer.CHAR_4,
    ]:
        chars_per_token = 4
    elif tokenizer in [Tokenizer.CHAR_5]:
        chars_per_token = 5
    elif tokenizer == Tokenizer.WORD:
        chars_per_token = 6  # Assuming average word length + space
    else:
        chars_per_token = 4  # Default

    return token_count * chars_per_token


def chars_to_tokens(
    char_count: int, tokenizer: Tokenizer = Tokenizer.CL100K_BASE
) -> int:
    """Estimate token count from character count.

    Args:
        char_count (int): Number of characters
        tokenizer (Tokenizer): Tokenizer for estimation. Defaults to Tokenizer.CL100K_BASE.

    Returns:
        int: Estimated token count

    Examples:
        >>> chars_to_tokens(400, tokenizer=Tokenizer.CL100K_BASE)
        100
    """
    # Tokenizer-specific character-to-token ratios
    if tokenizer in [
        Tokenizer.CL100K_BASE,
        Tokenizer.P50K_BASE,
        Tokenizer.R50K_BASE,
        Tokenizer.CHAR_4,
    ]:
        chars_per_token = 4
    elif tokenizer in [Tokenizer.CHAR_5]:
        chars_per_token = 5
    elif tokenizer == Tokenizer.WORD:
        chars_per_token = 6  # Assuming average word length + space
    else:
        chars_per_token = 4  # Default

    return char_count // chars_per_token


def estimate_cost(
    token_count: int, model: str = "gpt-4o", is_input: bool = True
) -> float:
    """Estimate API cost based on token usage.

    Args:
        token_count (int): Number of tokens
        model (str): Model name for pricing. Defaults to "gpt-4o".
        is_input (bool): Whether tokens are input (True) or output (False).
            Defaults to True.

    Returns:
        float: Estimated cost in USD

    Examples:
        >>> estimate_cost(1000, model="gpt-4o", is_input=True)
        0.005

        >>> estimate_cost(1000, model="gpt-4o-mini", is_input=False)
        0.0006

    Note:
        Pricing is approximate and may change. Check official pricing for accuracy.
    """
    # Approximate pricing per 1K tokens (as of 2026)
    # These should be updated to match current pricing
    pricing = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "text-embedding-ada-002": {"input": 0.0001, "output": 0.0001},
    }

    # Get pricing for model (default to gpt-4o if not found)
    model_pricing = pricing.get(model, pricing["gpt-4o"])
    price_per_1k = model_pricing["input"] if is_input else model_pricing["output"]

    # Calculate cost
    cost = (token_count / 1000) * price_per_1k
    return cost


def optimize_token_usage(
    text: str,
    max_tokens: Optional[int] = None,
    tokenizer: Tokenizer = Tokenizer.CL100K_BASE,
) -> dict:
    """Analyze and suggest optimizations for token usage.

    Args:
        text (str): Text to analyze
        max_tokens (Optional[int]): Maximum token limit. If provided, will check
            if text exceeds limit.
        tokenizer (Tokenizer): Tokenizer to use. Defaults to Tokenizer.CL100K_BASE.

    Returns:
        dict: Analysis results including:
            - token_count: Actual token count
            - char_count: Character count
            - tokens_per_char: Token to character ratio
            - exceeds_limit: Whether text exceeds max_tokens (if provided)
            - suggested_action: Recommended action based on analysis

    Examples:
        >>> result = optimize_token_usage("Hello world!", max_tokens=10)
        >>> result["token_count"]
        3
        >>> result["exceeds_limit"]
        False
    """
    from .tokenizer import count_tokens

    token_count = count_tokens(text, tokenizer)
    char_count = len(text)
    tokens_per_char = token_count / char_count if char_count > 0 else 0

    result = {
        "token_count": token_count,
        "char_count": char_count,
        "tokens_per_char": round(tokens_per_char, 4),
        "exceeds_limit": False,
        "suggested_action": "Token usage is optimal",
    }

    if max_tokens is not None:
        result["exceeds_limit"] = token_count > max_tokens

        if result["exceeds_limit"]:
            excess = token_count - max_tokens
            result["suggested_action"] = (
                f"Text exceeds limit by {excess} tokens. "
                f"Consider truncating or summarizing."
            )

    return result
