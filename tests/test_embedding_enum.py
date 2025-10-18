"""Test suite for EmbeddingModel enum functionality."""

import pytest

from kerb.embedding import EmbeddingModel, embed, embed_batch
from kerb.embedding.embedder import ModelBackend, _get_model_backend


class TestEmbeddingModelEnum:
    """Test EmbeddingModel enum values and functionality."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert EmbeddingModel.LOCAL.value == "local"
        assert EmbeddingModel.ALL_MINILM_L6_V2.value == "all-MiniLM-L6-v2"
        assert EmbeddingModel.ALL_MINILM_L12_V2.value == "all-MiniLM-L12-v2"
        assert EmbeddingModel.ALL_MPNET_BASE_V2.value == "all-mpnet-base-v2"
        assert EmbeddingModel.PARAPHRASE_MINILM_L6_V2.value == "paraphrase-MiniLM-L6-v2"
        assert (
            EmbeddingModel.PARAPHRASE_MPNET_BASE_V2.value == "paraphrase-mpnet-base-v2"
        )
        assert EmbeddingModel.TEXT_EMBEDDING_3_SMALL.value == "text-embedding-3-small"
        assert EmbeddingModel.TEXT_EMBEDDING_3_LARGE.value == "text-embedding-3-large"
        assert EmbeddingModel.TEXT_EMBEDDING_ADA_002.value == "text-embedding-ada-002"

    def test_enum_iteration(self):
        """Test iterating over enum values."""
        models = list(EmbeddingModel)
        assert len(models) == 9
        assert EmbeddingModel.LOCAL in models
        assert EmbeddingModel.TEXT_EMBEDDING_3_SMALL in models


class TestBackendDetection:
    """Test backend detection logic."""

    def test_local_backend_detection(self):
        """Test detection of local backend."""
        assert _get_model_backend("local") == ModelBackend.LOCAL
        assert _get_model_backend(EmbeddingModel.LOCAL) == ModelBackend.LOCAL

    def test_openai_backend_detection_prefix(self):
        """Test OpenAI model detection by prefix."""
        # Standard prefix
        assert _get_model_backend("text-embedding-3-small") == ModelBackend.OPENAI
        assert _get_model_backend("text-embedding-3-large") == ModelBackend.OPENAI
        assert _get_model_backend("text-embedding-ada-002") == ModelBackend.OPENAI

        # Future models (prefix-based detection)
        assert _get_model_backend("text-embedding-4-small") == ModelBackend.OPENAI
        assert _get_model_backend("text-embedding-5-ultra") == ModelBackend.OPENAI

        # Other OpenAI prefixes
        assert _get_model_backend("text-similarity-ada-001") == ModelBackend.OPENAI
        assert _get_model_backend("text-search-ada-doc-001") == ModelBackend.OPENAI

    def test_openai_backend_detection_enum(self):
        """Test OpenAI model detection from enum."""
        assert (
            _get_model_backend(EmbeddingModel.TEXT_EMBEDDING_3_SMALL)
            == ModelBackend.OPENAI
        )
        assert (
            _get_model_backend(EmbeddingModel.TEXT_EMBEDDING_3_LARGE)
            == ModelBackend.OPENAI
        )
        assert (
            _get_model_backend(EmbeddingModel.TEXT_EMBEDDING_ADA_002)
            == ModelBackend.OPENAI
        )

    def test_sentence_transformers_backend_detection(self):
        """Test Sentence Transformers backend (default)."""
        assert (
            _get_model_backend("all-MiniLM-L6-v2") == ModelBackend.SENTENCE_TRANSFORMERS
        )
        assert (
            _get_model_backend("all-mpnet-base-v2")
            == ModelBackend.SENTENCE_TRANSFORMERS
        )
        assert (
            _get_model_backend("custom-model-name")
            == ModelBackend.SENTENCE_TRANSFORMERS
        )
        assert (
            _get_model_backend("bert-base-uncased")
            == ModelBackend.SENTENCE_TRANSFORMERS
        )
        assert (
            _get_model_backend("path/to/custom/model")
            == ModelBackend.SENTENCE_TRANSFORMERS
        )

        # Test with enum
        assert (
            _get_model_backend(EmbeddingModel.ALL_MINILM_L6_V2)
            == ModelBackend.SENTENCE_TRANSFORMERS
        )
        assert (
            _get_model_backend(EmbeddingModel.ALL_MPNET_BASE_V2)
            == ModelBackend.SENTENCE_TRANSFORMERS
        )


class TestEmbedWithEnum:
    """Test embed() function with enum values."""

    def test_embed_with_local_enum(self):
        """Test embed with LOCAL enum."""
        vec = embed("Hello, world!", model=EmbeddingModel.LOCAL)
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(x, float) for x in vec)

    def test_embed_with_local_enum_custom_dimensions(self):
        """Test embed with LOCAL enum and custom dimensions."""
        vec = embed("Hello", model=EmbeddingModel.LOCAL, dimensions=512)
        assert len(vec) == 512

    def test_embed_default_model_is_local(self):
        """Test that default model is LOCAL."""
        vec1 = embed("Test")
        vec2 = embed("Test", model=EmbeddingModel.LOCAL)
        assert vec1 == vec2


class TestEmbedBatchWithEnum:
    """Test embed_batch() function with enum values."""

    def test_embed_batch_with_local_enum(self):
        """Test embed_batch with LOCAL enum."""
        texts = ["First", "Second", "Third"]
        vecs = embed_batch(texts, model=EmbeddingModel.LOCAL)

        assert len(vecs) == 3
        assert all(len(vec) == 384 for vec in vecs)
        assert all(isinstance(vec, list) for vec in vecs)

    def test_embed_batch_with_custom_dimensions(self):
        """Test embed_batch with custom dimensions."""
        texts = ["A", "B"]
        vecs = embed_batch(texts, model=EmbeddingModel.LOCAL, dimensions=256)

        assert all(len(vec) == 256 for vec in vecs)


class TestBackwardCompatibility:
    """Test that string model names work for custom models."""

    def test_custom_string_models_work(self):
        """Test that custom string models are accepted."""
        # Custom models can be passed as strings
        # They would only fail when trying to actually load the model
        # if the model doesn't exist
        pass


class TestEnumConsistency:
    """Test consistency of enum usage."""

    def test_default_model_is_local(self):
        """Test that default model uses LOCAL enum."""
        vec1 = embed("Consistency test")
        vec2 = embed("Consistency test", model=EmbeddingModel.LOCAL)
        assert vec1 == vec2

    def test_batch_default_model(self):
        """Test batch default model."""
        texts = ["First", "Second"]
        vecs1 = embed_batch(texts)
        vecs2 = embed_batch(texts, model=EmbeddingModel.LOCAL)
        assert vecs1 == vecs2

    def test_custom_dimensions_with_enum(self):
        """Test custom dimensions with enum."""
        text = "Test"
        vec = embed(text, model=EmbeddingModel.LOCAL, dimensions=128)
        assert len(vec) == 128


class TestEnumEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_with_enum(self):
        """Test empty text with enum."""
        vec = embed("", model=EmbeddingModel.LOCAL)
        assert len(vec) == 384
        assert all(x == 0.0 for x in vec)

    def test_empty_batch_with_enum(self):
        """Test empty batch with enum."""
        vecs = embed_batch([], model=EmbeddingModel.LOCAL)
        assert vecs == []


class TestModelBackendEnum:
    """Test ModelBackend enum."""

    def test_model_backend_values(self):
        """Test ModelBackend enum values."""
        assert ModelBackend.LOCAL.value == "local"
        assert ModelBackend.SENTENCE_TRANSFORMERS.value == "sentence_transformers"
        assert ModelBackend.OPENAI.value == "openai"

    def test_model_backend_iteration(self):
        """Test iterating over ModelBackend."""
        backends = list(ModelBackend)
        assert len(backends) == 3
        assert ModelBackend.LOCAL in backends
        assert ModelBackend.SENTENCE_TRANSFORMERS in backends
        assert ModelBackend.OPENAI in backends


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
