"""Data structures for retrieval operations.

This module defines the core data structures used throughout the retrieval subpackage.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, TYPE_CHECKING

from kerb.core.types import Document

if TYPE_CHECKING:
    from kerb.core.enums import FusionMethod


@dataclass
class SearchResult:
    """Represents a search result with relevance information."""
    document: Document
    score: float
    rank: int
    method: str = "unknown"  # e.g., "keyword", "semantic", "hybrid", "reranked"
    
    def __repr__(self) -> str:
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, method='{self.method}', doc_id='{self.document.id}')"


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search operations.
    
    Attributes:
        top_k: Number of top results to return
        keyword_weight: Weight for keyword scores (0-1)
        semantic_weight: Weight for semantic scores (0-1)
        fusion_method: Fusion method (FusionMethod enum or string)
    """
    top_k: int = 10
    keyword_weight: float = 0.5
    semantic_weight: float = 0.5
    fusion_method: Union['FusionMethod', str] = "weighted"


@dataclass
class FilterConfig:
    """Configuration for result filtering operations.
    
    Attributes:
        min_score: Minimum score threshold
        max_results: Maximum number of results
        metadata_filter: Filter by metadata fields
        dedup_threshold: Similarity threshold for deduplication (0-1)
    """
    min_score: Optional[float] = None
    max_results: Optional[int] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    dedup_threshold: float = 0.9
