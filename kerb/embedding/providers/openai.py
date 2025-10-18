"""OpenAI embedding provider."""

import os
from typing import Any, Dict, List, Optional

# Model cache for OpenAI clients
_client_cache: Dict[str, Any] = {}


def openai_embed(
    text: str,
    model_name: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    **kwargs,
) -> List[float]:
    """Generate embedding using OpenAI API.

    Requires: pip install openai

    Args:
        text (str): Text to embed
        model_name (str): OpenAI model name (default: "text-embedding-3-small")
        api_key (str, optional): OpenAI API key (or set OPENAI_API_KEY env var)
        **kwargs: Additional API parameters

    Returns:
        List[float]: Embedding vector

    Popular models:
        - "text-embedding-3-small" (1536 dim, cost-effective)
        - "text-embedding-3-large" (3072 dim, highest quality)
        - "text-embedding-ada-002" (1536 dim, legacy)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. " "Install with: pip install openai")

    # Get API key from parameter or environment variable
    effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not effective_api_key:
        raise ValueError(
            "OpenAI API key required. Pass api_key parameter or set OPENAI_API_KEY environment variable."
        )

    # Get or create client (cache by API key)
    client_key = f"openai_client_{effective_api_key[:10]}"
    if client_key not in _client_cache:
        _client_cache[client_key] = OpenAI(api_key=effective_api_key)

    client = _client_cache[client_key]

    response = client.embeddings.create(model=model_name, input=text, **kwargs)

    return response.data[0].embedding


def openai_embed_batch(
    texts: List[str],
    model_name: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    batch_size: int = 100,
    **kwargs,
) -> List[List[float]]:
    """Generate embeddings for multiple texts using OpenAI API.

    Processes texts in batches to stay within API limits.

    Args:
        texts (List[str]): Texts to embed
        model_name (str): OpenAI model name (default: "text-embedding-3-small")
        api_key (str, optional): OpenAI API key (or set OPENAI_API_KEY env var)
        batch_size (int): Number of texts per API call (max 2048 for OpenAI)
        **kwargs: Additional API parameters

    Returns:
        List[List[float]]: List of embedding vectors
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. " "Install with: pip install openai")

    # Get API key from parameter or environment variable
    effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not effective_api_key:
        raise ValueError(
            "OpenAI API key required. Pass api_key parameter or set OPENAI_API_KEY environment variable."
        )

    # Get or create client (cache by API key)
    client_key = f"openai_client_{effective_api_key[:10]}"
    if client_key not in _client_cache:
        _client_cache[client_key] = OpenAI(api_key=effective_api_key)

    client = _client_cache[client_key]

    all_embeddings = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model_name, input=batch, **kwargs)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


async def openai_embed_async(
    text: str,
    model_name: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    **kwargs,
) -> List[float]:
    """Generate embedding using OpenAI API asynchronously.

    Requires: pip install openai

    Args:
        text (str): Text to embed
        model_name (str): OpenAI model name (default: "text-embedding-3-small")
        api_key (str, optional): OpenAI API key (or set OPENAI_API_KEY env var)
        **kwargs: Additional API parameters

    Returns:
        List[float]: Embedding vector

    Examples:
        >>> import asyncio
        >>> embedding = asyncio.run(openai_embed_async("Hello world"))
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("openai not installed. " "Install with: pip install openai")

    # Get API key from parameter or environment variable
    effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not effective_api_key:
        raise ValueError(
            "OpenAI API key required. Pass api_key parameter or set OPENAI_API_KEY environment variable."
        )

    # Get or create async client (cache by API key)
    client_key = f"openai_async_client_{effective_api_key[:10]}"
    if client_key not in _client_cache:
        _client_cache[client_key] = AsyncOpenAI(api_key=effective_api_key)

    client = _client_cache[client_key]

    response = await client.embeddings.create(model=model_name, input=text, **kwargs)

    return response.data[0].embedding


async def openai_embed_batch_async(
    texts: List[str],
    model_name: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    batch_size: int = 100,
    max_concurrent: int = 5,
    **kwargs,
) -> List[List[float]]:
    """Generate embeddings for multiple texts using OpenAI API asynchronously.

    Processes texts in batches with concurrent requests for improved performance.

    Args:
        texts (List[str]): Texts to embed
        model_name (str): OpenAI model name (default: "text-embedding-3-small")
        api_key (str, optional): OpenAI API key (or set OPENAI_API_KEY env var)
        batch_size (int): Number of texts per API call (max 2048 for OpenAI)
        max_concurrent (int): Maximum concurrent API requests
        **kwargs: Additional API parameters

    Returns:
        List[List[float]]: List of embedding vectors

    Examples:
        >>> import asyncio
        >>> texts = ["Hello", "World", "AI"]
        >>> embeddings = asyncio.run(openai_embed_batch_async(texts))
    """
    try:
        import asyncio

        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("openai not installed. " "Install with: pip install openai")

    # Get API key from parameter or environment variable
    effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not effective_api_key:
        raise ValueError(
            "OpenAI API key required. Pass api_key parameter or set OPENAI_API_KEY environment variable."
        )

    # Get or create async client (cache by API key)
    client_key = f"openai_async_client_{effective_api_key[:10]}"
    if client_key not in _client_cache:
        _client_cache[client_key] = AsyncOpenAI(api_key=effective_api_key)

    client = _client_cache[client_key]

    # Create batches
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    # Process batches with concurrency control
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch(batch):
        async with semaphore:
            response = await client.embeddings.create(
                model=model_name, input=batch, **kwargs
            )
            return [item.embedding for item in response.data]

    # Process all batches concurrently
    batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])

    # Flatten results
    all_embeddings = []
    for batch_embeddings in batch_results:
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


class OpenAIEmbedder:
    """OpenAI embedding provider.

    Requires: pip install openai

    Args:
        model_name (str): OpenAI model name (default: "text-embedding-3-small")
        api_key (str, optional): OpenAI API key (or set OPENAI_API_KEY env var)

    Examples:
        embedder = OpenAIEmbedder(model_name="text-embedding-3-large")
        vec = embedder.embed("Hello world")
        vecs = embedder.embed_batch(["Hello", "World"])

        # Async usage
        import asyncio
        async def main():
            vec = await embedder.embed_async("Hello")
        asyncio.run(main())
    """

    def __init__(
        self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None
    ):
        """Initialize the OpenAI embedder.

        Args:
            model_name (str): OpenAI model name
            api_key (str, optional): OpenAI API key
        """
        self.model_name = model_name
        self.api_key = api_key

    def embed(self, text: str, **kwargs) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text (str): Text to embed
            **kwargs: Additional API parameters

        Returns:
            List[float]: Embedding vector
        """
        return openai_embed(text, self.model_name, self.api_key, **kwargs)

    def embed_batch(
        self, texts: List[str], batch_size: int = 100, **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts (List[str]): Texts to embed
            batch_size (int): Number of texts per API call
            **kwargs: Additional API parameters

        Returns:
            List[List[float]]: List of embedding vectors
        """
        return openai_embed_batch(
            texts, self.model_name, self.api_key, batch_size, **kwargs
        )

    async def embed_async(self, text: str, **kwargs) -> List[float]:
        """Generate embedding asynchronously.

        Args:
            text (str): Text to embed
            **kwargs: Additional API parameters

        Returns:
            List[float]: Embedding vector
        """
        return await openai_embed_async(text, self.model_name, self.api_key, **kwargs)

    async def embed_batch_async(
        self, texts: List[str], batch_size: int = 100, max_concurrent: int = 5, **kwargs
    ) -> List[List[float]]:
        """Generate embeddings asynchronously for multiple texts.

        Args:
            texts (List[str]): Texts to embed
            batch_size (int): Number of texts per API call
            max_concurrent (int): Maximum concurrent requests
            **kwargs: Additional API parameters

        Returns:
            List[List[float]]: List of embedding vectors
        """
        return await openai_embed_batch_async(
            texts, self.model_name, self.api_key, batch_size, max_concurrent, **kwargs
        )
