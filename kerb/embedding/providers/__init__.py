"""Embedding providers.

Local Providers (run on your machine):
- Hash-based (no dependencies)
- Sentence Transformers (local ML models)

Remote Providers (API-based):
- OpenAI
"""

# Local providers (both run locally)
from .local import (  # Hash-based (no dependencies); Sentence Transformers (local ML)
    LocalEmbedder, SentenceTransformerEmbedder, local_embed,
    sentence_transformer_embed, sentence_transformer_embed_batch)
# Remote providers (API-based)
from .openai import (OpenAIEmbedder, openai_embed, openai_embed_async,
                     openai_embed_batch, openai_embed_batch_async)

__all__ = [
    # Local providers - Hash-based
    "LocalEmbedder",
    "local_embed",
    # Local providers - Sentence Transformers
    "SentenceTransformerEmbedder",
    "sentence_transformer_embed",
    "sentence_transformer_embed_batch",
    # Remote providers - OpenAI
    "OpenAIEmbedder",
    "openai_embed",
    "openai_embed_batch",
    "openai_embed_async",
    "openai_embed_batch_async",
]
