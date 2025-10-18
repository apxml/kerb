"""Test doubles (stubs) for testing."""

import random
import hashlib
from typing import List, Dict, Any, Optional, Union


class StubEmbedding:
    """Stub embedding model for testing."""
    
    def __init__(self, dimension: int = 768, deterministic: bool = True):
        """Initialize stub embedding.
        
        Args:
            dimension: Embedding dimension
            deterministic: Whether to generate deterministic embeddings
        """
        self.dimension = dimension
        self.deterministic = deterministic
    
    def embed(self, text: str) -> List[float]:
        """Generate stub embedding.
        
        Args:
            text: Input text
            
        Returns:
            Stub embedding vector
        """
        if self.deterministic:
            # Hash-based deterministic embedding
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            rng = random.Random(hash_val)
            return [rng.random() for _ in range(self.dimension)]
        else:
            return [random.random() for _ in range(self.dimension)]


class StubRetriever:
    """Stub retrieval system for testing."""
    
    def __init__(self, documents: Optional[Dict[str, str]] = None):
        """Initialize stub retriever.
        
        Args:
            documents: Dict mapping doc IDs to content
        """
        self.documents = documents or {}
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Stub retrieval.
        
        Args:
            query: Query string
            top_k: Number of results
            
        Returns:
            List of stub results
        """
        # Simple keyword matching
        results = []
        for doc_id, content in self.documents.items():
            if query.lower() in content.lower():
                results.append({
                    "id": doc_id,
                    "content": content,
                    "score": 0.9
                })
        
        return results[:top_k]


class StubVectorStore:
    """Stub vector store for testing."""
    
    def __init__(self):
        """Initialize stub vector store."""
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def add(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict] = None
    ) -> None:
        """Add vector to store."""
        self.vectors[id] = vector
        self.metadata[id] = metadata or {}
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Stub search."""
        # Return random results
        ids = list(self.vectors.keys())
        random.shuffle(ids)
        
        results = []
        for id in ids[:top_k]:
            results.append({
                "id": id,
                "vector": self.vectors[id],
                "metadata": self.metadata[id],
                "score": random.random()
            })
        
        return results


def create_test_double(
    type: str,
    **kwargs
) -> Union[StubEmbedding, StubRetriever, StubVectorStore]:
    """Factory for creating test doubles.
    
    Args:
        type: Type of test double (embedding, retriever, vector_store)
        **kwargs: Configuration parameters
        
    Returns:
        Test double instance
    """
    if type == "embedding":
        return StubEmbedding(**kwargs)
    elif type == "retriever":
        return StubRetriever(**kwargs)
    elif type == "vector_store":
        return StubVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown test double type: {type}")
