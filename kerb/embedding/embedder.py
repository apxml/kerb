"""Core embedding functions with multiple backend support."""

from enum import Enum
from typing import Generator, List, Optional, Tuple, Union


class EmbeddingModel(Enum):
    """Enum for embedding models.

    For custom models not listed here, use a plain string instead.
    """

    # Local embedding (hash-based, no dependencies)
    LOCAL = "local"

    # Sentence Transformers models (local ML)
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"  # 384 dim, fast
    ALL_MINILM_L12_V2 = "all-MiniLM-L12-v2"  # 384 dim, balanced
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"  # 768 dim, quality
    PARAPHRASE_MINILM_L6_V2 = "paraphrase-MiniLM-L6-v2"  # 384 dim
    PARAPHRASE_MPNET_BASE_V2 = "paraphrase-mpnet-base-v2"  # 768 dim

    # OpenAI models (remote API)
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"  # 1536 dim
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"  # 3072 dim
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"  # 1536 dim, legacy


class ModelBackend(Enum):
    """Enum for embedding backends."""

    LOCAL = "local"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"


def _get_model_backend(model: Union[str, EmbeddingModel]) -> ModelBackend:
    """Determine which backend to use for a given model.

    Args:
        model: Model name (string) or EmbeddingModel enum value

    Returns:
        ModelBackend: The backend type to use
    """
    # Convert enum to string if needed
    model_str = model.value if isinstance(model, EmbeddingModel) else model

    # Check for local model
    if model_str == "local":
        return ModelBackend.LOCAL

    # Check for OpenAI models (more robust pattern matching)
    openai_prefixes = ["text-embedding-", "text-similarity-", "text-search-"]
    if any(model_str.startswith(prefix) for prefix in openai_prefixes):
        return ModelBackend.OPENAI

    # Check for known OpenAI model names
    openai_models = {
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    }
    if model_str in openai_models:
        return ModelBackend.OPENAI

    # Default to Sentence Transformers for everything else
    return ModelBackend.SENTENCE_TRANSFORMERS


def embed(
    text: str,
    model: Union[str, EmbeddingModel] = EmbeddingModel.LOCAL,
    dimensions: int = 384,
    api_key: Optional[str] = None,
    **kwargs,
) -> List[float]:
    """Generate an embedding vector for text.

    Args:
        text (str): The text to embed
        model (str or EmbeddingModel): Model to use:
            - EmbeddingModel.LOCAL - Hash-based (default, no dependencies)
            - EmbeddingModel.ALL_MINILM_L6_V2 - Sentence Transformers (384 dim)
            - EmbeddingModel.ALL_MPNET_BASE_V2 - Sentence Transformers (768 dim)
            - EmbeddingModel.TEXT_EMBEDDING_3_SMALL - OpenAI (1536 dim)
            - EmbeddingModel.TEXT_EMBEDDING_3_LARGE - OpenAI (3072 dim)
            - Or use a string for custom models: "custom-model-name"
        dimensions (int): Dimension for local embeddings (default: 384)
        api_key (str, optional): OpenAI API key (or set OPENAI_API_KEY env var)
        **kwargs: Additional model-specific parameters

    Returns:
        List[float]: Embedding vector (normalized to unit length)

    Examples:
        # Using enum (recommended for known models)
        vec = embed("Hello, world!")
        vec = embed("Hello", model=EmbeddingModel.ALL_MINILM_L6_V2)
        vec = embed("Hello", model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL, api_key="sk-...")

        # Using string for custom models
        vec = embed("Hello", model="my-custom-sentence-transformer")
    """
    from .providers.local import local_embed, sentence_transformer_embed
    from .providers.openai import openai_embed

    # Convert enum to string if needed
    model_str = model.value if isinstance(model, EmbeddingModel) else model

    # Determine backend and route accordingly
    backend = _get_model_backend(model)

    if backend == ModelBackend.LOCAL:
        return local_embed(text, dimensions=dimensions)
    elif backend == ModelBackend.OPENAI:
        return openai_embed(text, model_name=model_str, api_key=api_key, **kwargs)
    else:  # SENTENCE_TRANSFORMERS
        return sentence_transformer_embed(text, model_name=model_str, **kwargs)


def embed_batch(
    texts: List[str],
    model: Union[str, EmbeddingModel] = EmbeddingModel.LOCAL,
    dimensions: int = 384,
    batch_size: int = 32,
    api_key: Optional[str] = None,
    **kwargs,
) -> List[List[float]]:
    """Generate embeddings for multiple texts efficiently.

    Args:
        texts (List[str]): List of texts to embed
        model (str or EmbeddingModel): Model to use (see embed() for options)
        dimensions (int): Dimension for local embeddings
        batch_size (int): Batch size for processing
        api_key (str, optional): OpenAI API key (or set OPENAI_API_KEY env var)
        **kwargs: Additional model-specific parameters

    Returns:
        List[List[float]]: List of embedding vectors

    Examples:
        # Using enum
        embeddings = embed_batch(["doc1", "doc2", "doc3"])
        embeddings = embed_batch(docs, model=EmbeddingModel.ALL_MINILM_L6_V2)
        embeddings = embed_batch(docs, model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL)

        # Using string for custom models
        embeddings = embed_batch(docs, model="custom-model")
    """
    from .providers.local import local_embed, sentence_transformer_embed_batch
    from .providers.openai import openai_embed_batch

    # Convert enum to string if needed
    model_str = model.value if isinstance(model, EmbeddingModel) else model

    # Determine backend and route accordingly
    backend = _get_model_backend(model)

    if backend == ModelBackend.LOCAL:
        return [local_embed(text, dimensions=dimensions) for text in texts]
    elif backend == ModelBackend.OPENAI:
        return openai_embed_batch(
            texts,
            model_name=model_str,
            api_key=api_key,
            batch_size=batch_size,
            **kwargs,
        )
    else:  # SENTENCE_TRANSFORMERS
        return sentence_transformer_embed_batch(
            texts, model_name=model_str, batch_size=batch_size, **kwargs
        )


async def embed_async(
    text: str,
    model: Union[str, EmbeddingModel] = EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
    api_key: Optional[str] = None,
    **kwargs,
) -> List[float]:
    """Generate embedding asynchronously (wrapper for API-based models).

    Args:
        text (str): Text to embed
        model: Embedding model to use
        api_key (str, optional): API key (for OpenAI models)
        **kwargs: Additional model parameters

    Returns:
        List[float]: Embedding vector

    Note:
        Currently only supports async for OpenAI models.
        Local models will run synchronously in a thread pool.

    Examples:
        >>> import asyncio
        >>> embedding = asyncio.run(embed_async("Hello world"))
    """
    from .providers.openai import openai_embed_async

    backend = _get_model_backend(model)
    model_str = model.value if isinstance(model, EmbeddingModel) else model

    if backend == ModelBackend.OPENAI:
        return await openai_embed_async(
            text, model_name=model_str, api_key=api_key, **kwargs
        )
    else:
        # For local models, run in thread pool
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, embed, text, model, **kwargs)


async def embed_batch_async(
    texts: List[str],
    model: Union[str, EmbeddingModel] = EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
    api_key: Optional[str] = None,
    batch_size: int = 100,
    max_concurrent: int = 5,
    **kwargs,
) -> List[List[float]]:
    """Generate embeddings for multiple texts asynchronously.

    Args:
        texts (List[str]): Texts to embed
        model: Embedding model to use
        api_key (str, optional): API key (for OpenAI models)
        batch_size (int): Number of texts per API call
        max_concurrent (int): Maximum concurrent requests (for API models)
        **kwargs: Additional model parameters

    Returns:
        List[List[float]]: List of embedding vectors

    Examples:
        >>> import asyncio
        >>> texts = ["Hello", "World", "AI"]
        >>> embeddings = asyncio.run(embed_batch_async(texts))
    """
    from .providers.openai import openai_embed_batch_async

    backend = _get_model_backend(model)
    model_str = model.value if isinstance(model, EmbeddingModel) else model

    if backend == ModelBackend.OPENAI:
        return await openai_embed_batch_async(
            texts,
            model_name=model_str,
            api_key=api_key,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            **kwargs,
        )
    else:
        # For local models, run in thread pool
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, embed_batch, texts, model, **kwargs)


def embed_batch_stream(
    texts: List[str],
    model: Union[str, EmbeddingModel] = EmbeddingModel.LOCAL,
    batch_size: int = 32,
    api_key: Optional[str] = None,
    **kwargs,
) -> Generator[Tuple[int, List[float]], None, None]:
    """Stream embeddings for large datasets (memory efficient).

    Yields embeddings one at a time instead of loading all into memory.
    Useful for processing very large datasets.

    Args:
        texts (List[str]): Texts to embed
        model: Embedding model to use
        batch_size (int): Number of texts to process per batch
        api_key (str, optional): API key (for API-based models)
        **kwargs: Additional model parameters

    Yields:
        Tuple[int, List[float]]: (index, embedding) pairs

    Examples:
        >>> texts = ["text1", "text2", ...]  # Large list
        >>> for idx, embedding in embed_batch_stream(texts, batch_size=100):
        ...     # Process embedding immediately
        ...     print(f"Processed {idx}")
    """
    from .providers.local import local_embed, sentence_transformer_embed_batch
    from .providers.openai import openai_embed_batch

    backend = _get_model_backend(model)
    model_str = model.value if isinstance(model, EmbeddingModel) else model

    # Process in batches
    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]

        # Get embeddings for this batch
        if backend == ModelBackend.OPENAI:
            batch_embeddings = openai_embed_batch(
                batch_texts,
                model_name=model_str,
                api_key=api_key,
                batch_size=batch_size,
                **kwargs,
            )
        elif backend == ModelBackend.SENTENCE_TRANSFORMERS:
            batch_embeddings = sentence_transformer_embed_batch(
                batch_texts, model_name=model_str, batch_size=batch_size, **kwargs
            )
        else:  # LOCAL
            batch_embeddings = [local_embed(text, **kwargs) for text in batch_texts]

        # Yield each embedding with its global index
        for i, embedding in enumerate(batch_embeddings):
            yield (batch_start + i, embedding)


async def embed_batch_stream_async(
    texts: List[str],
    model: Union[str, EmbeddingModel] = EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
    batch_size: int = 100,
    api_key: Optional[str] = None,
    max_concurrent: int = 5,
    **kwargs,
):
    """Stream embeddings asynchronously for large datasets.

    Args:
        texts (List[str]): Texts to embed
        model: Embedding model to use
        batch_size (int): Number of texts per API call
        api_key (str, optional): API key (for API-based models)
        max_concurrent (int): Maximum concurrent requests
        **kwargs: Additional model parameters

    Yields:
        Tuple[int, List[float]]: (index, embedding) pairs

    Examples:
        >>> async def process():
        ...     texts = ["text1", "text2", ...]
        ...     async for idx, embedding in embed_batch_stream_async(texts):
        ...         print(f"Processed {idx}")
        >>> asyncio.run(process())
    """
    import os

    backend = _get_model_backend(model)
    model_str = model.value if isinstance(model, EmbeddingModel) else model

    if backend == ModelBackend.OPENAI:
        try:
            import asyncio

            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai required for async streaming")

        from .providers.openai import _client_cache

        # Get API key
        effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not effective_api_key:
            raise ValueError("OpenAI API key required")

        # Get or create async client
        client_key = f"openai_async_client_{effective_api_key[:10]}"
        if client_key not in _client_cache:
            _client_cache[client_key] = AsyncOpenAI(api_key=effective_api_key)

        client = _client_cache[client_key]

        # Create batches
        batches = [
            (i, texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)
        ]

        # Process batches with concurrency control
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch_start, batch_texts):
            async with semaphore:
                response = await client.embeddings.create(
                    model=model_str, input=batch_texts, **kwargs
                )
                return [
                    (batch_start + i, item.embedding)
                    for i, item in enumerate(response.data)
                ]

        # Process all batches and yield results
        for batch_start, batch_texts in batches:
            results = await process_batch(batch_start, batch_texts)
            for idx, embedding in results:
                yield (idx, embedding)
    else:
        # For local models, use synchronous streaming in thread pool
        for idx, embedding in embed_batch_stream(texts, model, batch_size, **kwargs):
            yield (idx, embedding)
