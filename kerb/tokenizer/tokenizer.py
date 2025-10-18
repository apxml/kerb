"""Core token counting utilities for LLM applications.

This module provides flexible token counting for various LLM models and tokenizers,
supporting OpenAI models (via tiktoken), HuggingFace models (via transformers),
and approximation methods for quick estimates.
"""

from typing import Optional, List, Union
from enum import Enum
import warnings


class Tokenizer(Enum):
    """Enumeration of supported tokenizers for token counting.
    
    Using explicit tokenizers instead of model names provides better control
    and consistency for LLM developers.
    
    Tiktoken Encodings (OpenAI):
        CL100K_BASE: GPT-4, GPT-3.5-turbo, text-embedding-ada-002
        P50K_BASE: Code models (Codex, text-davinci-002, text-davinci-003)
        R50K_BASE: GPT-3 models (davinci, curie, babbage, ada)
        
    Approximation Methods:
        CHAR_4: Fast approximation using 4 chars/token (good for GPT-like models)
        CHAR_5: Fast approximation using 5 chars/token (good for BERT-like models)
        WORD: Word-based approximation (1.3 tokens/word average)
    """
    # Tiktoken encodings
    CL100K_BASE = "cl100k_base"
    P50K_BASE = "p50k_base"
    R50K_BASE = "r50k_base"
    P50K_EDIT = "p50k_edit"
    
    # Approximation methods
    CHAR_4 = "approximate_char_4"
    CHAR_5 = "approximate_char_5"
    WORD = "approximate_word"
    
    @property
    def method(self) -> str:
        """Get the tokenization method for this tokenizer."""
        if self.value.startswith("approximate"):
            return "approximate"
        else:
            return "tiktoken"


def count_tokens(
    text: str,
    tokenizer: Union[Tokenizer, str] = Tokenizer.CL100K_BASE
) -> int:
    """Count tokens in text using the specified tokenizer.
    
    Args:
        text (str): Text to count tokens for
        tokenizer (Union[Tokenizer, str]): Tokenizer to use. Can be a Tokenizer enum value
            or a HuggingFace model name (e.g., "bert-base-uncased", "meta-llama/Llama-2-7b-hf").
            Defaults to Tokenizer.CL100K_BASE (used by GPT-4 and GPT-3.5-turbo).
            
    Returns:
        int: Token count
        
    Examples:
        >>> count_tokens("Hello world!", tokenizer=Tokenizer.CL100K_BASE)
        3
        
        >>> count_tokens("Hello world!", tokenizer=Tokenizer.P50K_BASE)
        3
        
        >>> count_tokens("Hello world!", tokenizer="bert-base-uncased")
        4
        
        >>> count_tokens("Hello world!", tokenizer=Tokenizer.CHAR_4)
        3
    """
    if not text:
        return 0
    
    # Handle string tokenizer (HuggingFace model name)
    if isinstance(tokenizer, str) and not isinstance(tokenizer, Tokenizer):
        return _count_tokens_transformers(text, tokenizer)
    
    # Handle Tokenizer enum
    if isinstance(tokenizer, Tokenizer):
        if tokenizer.method == "tiktoken":
            return _count_tokens_tiktoken(text, tokenizer.value)
        elif tokenizer.method == "approximate":
            return _count_tokens_approximate(text, tokenizer)
    
    raise ValueError(f"Invalid tokenizer: {tokenizer}")


def batch_count_tokens(
    texts: List[str],
    tokenizer: Union[Tokenizer, str] = Tokenizer.CL100K_BASE
) -> List[int]:
    """Count tokens for multiple texts.
    
    Args:
        texts (List[str]): List of texts to count tokens for
        tokenizer (Union[Tokenizer, str]): Tokenizer to use. Defaults to Tokenizer.CL100K_BASE.
        
    Returns:
        List[int]: List of token counts
        
    Examples:
        >>> texts = ["Hello world!", "How are you?", "Good morning!"]
        >>> batch_count_tokens(texts, tokenizer=Tokenizer.CL100K_BASE)
        [3, 4, 3]
    """
    return [count_tokens(text, tokenizer) for text in texts]


def count_tokens_for_messages(
    messages: List[dict],
    tokenizer: Union[Tokenizer, str] = Tokenizer.CL100K_BASE
) -> int:
    """Count tokens for a list of chat messages including format overhead.
    
    OpenAI chat models format messages with special tokens. This function
    accounts for the overhead of message formatting. Works best with tiktoken
    tokenizers (CL100K_BASE, P50K_BASE, etc.).
    
    Args:
        messages (List[dict]): List of message dicts with 'role' and 'content' keys.
            Example: [{"role": "user", "content": "Hello!"}]
        tokenizer (Union[Tokenizer, str]): Tokenizer to use. Defaults to Tokenizer.CL100K_BASE.
        
    Returns:
        int: Total token count including message formatting overhead
        
    Examples:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> count_tokens_for_messages(messages, tokenizer=Tokenizer.CL100K_BASE)
        28
    """
    if not messages:
        return 0
    
    # Try using tiktoken for accurate counting
    try:
        import tiktoken
        
        # Get encoding based on tokenizer
        if isinstance(tokenizer, Tokenizer):
            if tokenizer.method == "tiktoken":
                encoding_obj = tiktoken.get_encoding(tokenizer.value)
            else:
                # Fall back to cl100k_base for approximation tokenizers
                encoding_obj = tiktoken.get_encoding("cl100k_base")
        elif isinstance(tokenizer, str):
            # Try as HuggingFace model - fall back to cl100k_base
            try:
                encoding_obj = tiktoken.encoding_for_model(tokenizer)
            except KeyError:
                encoding_obj = tiktoken.get_encoding("cl100k_base")
        else:
            encoding_obj = tiktoken.get_encoding("cl100k_base")
        
        # Token overhead is consistent for cl100k_base encoding
        tokens_per_message = 3
        tokens_per_name = 1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding_obj.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
        
    except ImportError:
        warnings.warn(
            "tiktoken not installed. Using approximation for message token counting. "
            "Install with: pip install tiktoken"
        )
        # Approximate counting
        total = 0
        for message in messages:
            content = message.get("content", "")
            total += count_tokens(content, tokenizer)
            total += 4  # Overhead per message
        total += 3  # Reply priming tokens
        return total


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    tokenizer: Union[Tokenizer, str] = Tokenizer.CL100K_BASE,
    preserve_end: bool = False,
    ellipsis: str = "..."
) -> str:
    """Truncate text to fit within a token limit.
    
    Args:
        text (str): Text to truncate
        max_tokens (int): Maximum number of tokens
        tokenizer (Union[Tokenizer, str]): Tokenizer to use. Defaults to Tokenizer.CL100K_BASE.
        preserve_end (bool): If True, keep the end of text instead of beginning. 
            Defaults to False.
        ellipsis (str): String to indicate truncation. Defaults to "...".
        
    Returns:
        str: Truncated text
        
    Examples:
        >>> text = "This is a long text that needs to be truncated."
        >>> truncate_to_token_limit(text, max_tokens=5, tokenizer=Tokenizer.CL100K_BASE)
        'This is a long...'
    """
    if not text:
        return ""
    
    current_tokens = count_tokens(text, tokenizer)
    
    if current_tokens <= max_tokens:
        return text
    
    # Import from utils module for character conversion
    from .utils import tokens_to_chars
    
    # Try using tiktoken for accurate truncation
    if isinstance(tokenizer, Tokenizer) and tokenizer.method == "tiktoken":
        try:
            import tiktoken
            
            encoding_obj = tiktoken.get_encoding(tokenizer.value)
            
            # Encode the text
            tokens = encoding_obj.encode(text)
            
            # Account for ellipsis tokens
            ellipsis_tokens = len(encoding_obj.encode(ellipsis))
            available_tokens = max_tokens - ellipsis_tokens
            
            if available_tokens <= 0:
                return ellipsis
            
            if preserve_end:
                truncated_tokens = tokens[-available_tokens:]
                return ellipsis + encoding_obj.decode(truncated_tokens)
            else:
                truncated_tokens = tokens[:available_tokens]
                return encoding_obj.decode(truncated_tokens) + ellipsis
                
        except ImportError:
            pass  # Fall through to character-based truncation
    
    # Fallback to character-based truncation
    if isinstance(tokenizer, Tokenizer):
        char_limit = tokens_to_chars(max_tokens, tokenizer)
    else:
        char_limit = max_tokens * 4  # Default approximation
    
    if preserve_end:
        if len(text) > char_limit:
            return ellipsis + text[-(char_limit - len(ellipsis)):]
        return text
    else:
        if len(text) > char_limit:
            return text[:char_limit - len(ellipsis)] + ellipsis
        return text


# Private helper functions

def _count_tokens_tiktoken(text: str, encoding: str) -> int:
    """Count tokens using tiktoken library."""
    try:
        import tiktoken
        
        enc = tiktoken.get_encoding(encoding)
        return len(enc.encode(text))
        
    except ImportError:
        warnings.warn(
            f"tiktoken not installed. Using approximation. "
            "Install with: pip install tiktoken"
        )
        return _count_tokens_approximate(text, Tokenizer.CHAR_4)


def _count_tokens_transformers(text: str, model: str) -> int:
    """Count tokens using HuggingFace transformers library."""
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokens = tokenizer.encode(text)
        return len(tokens)
        
    except ImportError:
        warnings.warn(
            f"transformers not installed. Using approximation for {model}. "
            "Install with: pip install transformers"
        )
        return _count_tokens_approximate(text, Tokenizer.CHAR_5)
    except Exception as e:
        warnings.warn(
            f"Could not load tokenizer for {model}: {e}. Using approximation."
        )
        return _count_tokens_approximate(text, Tokenizer.CHAR_5)


def _count_tokens_approximate(text: str, tokenizer: Tokenizer) -> int:
    """Fast approximation of token count based on character count or word count.
    
    This provides a quick estimate without requiring external libraries.
    Accuracy varies by language and text type.
    """
    if not text:
        return 0
    
    if tokenizer == Tokenizer.CHAR_4:
        return len(text) // 4
    elif tokenizer == Tokenizer.CHAR_5:
        return len(text) // 5
    elif tokenizer == Tokenizer.WORD:
        words = text.split()
        # Assume ~1.3 tokens per word on average
        return int(len(words) * 1.3)
    else:
        # Default to CHAR_4
        return len(text) // 4
