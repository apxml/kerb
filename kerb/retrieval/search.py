"""Search methods for retrieval.

This module provides keyword, semantic, and hybrid search implementations.
"""

import math
from typing import List, Optional, Union, TYPE_CHECKING
from collections import Counter

from kerb.core.types import Document
from .structures import SearchResult, HybridSearchConfig

if TYPE_CHECKING:
    from kerb.core.enums import FusionMethod


def keyword_search(
    query: str,
    documents: List[Document],
    top_k: int = 10,
    field: str = "content"
) -> List[SearchResult]:
    """Perform keyword-based search using BM25-like scoring.
    
    Args:
        query: Search query
        documents: List of documents to search
        top_k: Number of top results to return
        field: Document field to search ("content" or metadata key)
        
    Returns:
        List[SearchResult]: Ranked search results
        
    Example:
        >>> docs = [Document(id="1", content="Python is great"), ...]
        >>> results = keyword_search("python programming", docs)
    """
    if not documents:
        return []
    
    query_terms = query.lower().split()
    
    # Calculate document frequencies
    doc_freq = Counter()
    for doc in documents:
        text = doc.content if field == "content" else doc.metadata.get(field, "")
        terms = set(text.lower().split())
        for term in terms:
            if term in query_terms:
                doc_freq[term] += 1
    
    # Score documents
    scores = []
    num_docs = len(documents)
    
    for doc in documents:
        text = doc.content if field == "content" else doc.metadata.get(field, "")
        text_terms = text.lower().split()
        term_freq = Counter(text_terms)
        
        score = 0.0
        for term in query_terms:
            if term in term_freq:
                # BM25-like scoring
                tf = term_freq[term]
                df = doc_freq[term]
                idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
                
                # BM25 parameters
                k1 = 1.5
                b = 0.75
                avg_doc_len = sum(len(d.content.split()) for d in documents) / num_docs
                doc_len = len(text_terms)
                
                norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                score += idf * norm_tf
        
        scores.append((doc, score))
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Create search results
    results = []
    for rank, (doc, score) in enumerate(scores[:top_k], 1):
        results.append(SearchResult(
            document=doc,
            score=score,
            rank=rank,
            method="keyword"
        ))
    
    return results


def semantic_search(
    query_embedding: List[float],
    documents: List[Document],
    document_embeddings: List[List[float]],
    top_k: int = 10,
    similarity_metric: str = "cosine"
) -> List[SearchResult]:
    """Perform semantic search using embeddings.
    
    Args:
        query_embedding: Embedding vector of the query
        documents: List of documents
        document_embeddings: Embedding vectors for documents (same order as documents)
        top_k: Number of top results to return
        similarity_metric: "cosine", "dot", or "euclidean"
        
    Returns:
        List[SearchResult]: Ranked search results
        
    Example:
        >>> from kerb.embedding import embed
        >>> query_emb = embed("python programming")
        >>> doc_embs = [embed(doc.content) for doc in docs]
        >>> results = semantic_search(query_emb, docs, doc_embs)
    """
    if not documents or len(documents) != len(document_embeddings):
        return []
    
    scores = []
    
    for doc, doc_emb in zip(documents, document_embeddings):
        if similarity_metric == "cosine":
            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_embedding, doc_emb))
            norm_q = math.sqrt(sum(a * a for a in query_embedding))
            norm_d = math.sqrt(sum(b * b for b in doc_emb))
            score = dot / (norm_q * norm_d) if norm_q and norm_d else 0.0
            
        elif similarity_metric == "dot":
            # Dot product
            score = sum(a * b for a, b in zip(query_embedding, doc_emb))
            
        elif similarity_metric == "euclidean":
            # Euclidean distance (inverted for ranking)
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(query_embedding, doc_emb)))
            score = 1.0 / (1.0 + dist)  # Convert to similarity
        else:
            score = 0.0
        
        scores.append((doc, score))
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Create search results
    results = []
    for rank, (doc, score) in enumerate(scores[:top_k], 1):
        results.append(SearchResult(
            document=doc,
            score=score,
            rank=rank,
            method="semantic"
        ))
    
    return results


def hybrid_search(
    query: str,
    query_embedding: List[float],
    documents: List[Document],
    document_embeddings: List[List[float]],
    top_k: int = 10,
    keyword_weight: float = 0.5,
    semantic_weight: float = 0.5,
    fusion_method: Union['FusionMethod', str] = "weighted",
    config: Optional[HybridSearchConfig] = None
) -> List[SearchResult]:
    """Perform hybrid search combining keyword and semantic search.
    
    Args:
        query: Search query text
        query_embedding: Embedding vector of the query
        documents: List of documents
        document_embeddings: Embedding vectors for documents
        top_k: Number of top results to return (ignored if config is provided)
        keyword_weight: Weight for keyword scores (ignored if config is provided)
        semantic_weight: Weight for semantic scores (ignored if config is provided)
        fusion_method: Fusion method (ignored if config is provided)
        config: HybridSearchConfig object with all parameters (recommended)
        
    Returns:
        List[SearchResult]: Ranked search results
        
    Examples:
        >>> # Using config object (recommended)
        >>> from kerb.retrieval import HybridSearchConfig
        >>> from kerb.core.enums import FusionMethod
        >>> config = HybridSearchConfig(
        ...     top_k=10,
        ...     keyword_weight=0.4,
        ...     semantic_weight=0.6,
        ...     fusion_method=FusionMethod.RRF
        ... )
        >>> results = hybrid_search(
        ...     query="python async",
        ...     query_embedding=embed("python async"),
        ...     documents=docs,
        ...     document_embeddings=doc_embs,
        ...     config=config
        ... )
        
        >>> # Using individual parameters (backward compatible)
        >>> results = hybrid_search(
        ...     query="python async",
        ...     query_embedding=embed("python async"),
        ...     documents=docs,
        ...     document_embeddings=doc_embs,
        ...     keyword_weight=0.4,
        ...     semantic_weight=0.6
        ... )
    """
    from kerb.core.enums import FusionMethod, validate_enum_or_string
    
    # Use config if provided, otherwise use individual parameters
    if config is not None:
        top_k = config.top_k
        keyword_weight = config.keyword_weight
        semantic_weight = config.semantic_weight
        fusion_method = config.fusion_method
    
    # Validate and normalize fusion_method
    method_val = validate_enum_or_string(fusion_method, FusionMethod, "fusion_method")
    if isinstance(method_val, FusionMethod):
        method_str = method_val.value
    else:
        method_str = method_val
    
    # Get keyword search results
    keyword_results = keyword_search(query, documents, top_k=len(documents))
    
    # Get semantic search results
    semantic_results = semantic_search(
        query_embedding, documents, document_embeddings, top_k=len(documents)
    )
    
    # Create score maps
    keyword_scores = {r.document.id: r.score for r in keyword_results}
    semantic_scores = {r.document.id: r.score for r in semantic_results}
    keyword_ranks = {r.document.id: r.rank for r in keyword_results}
    semantic_ranks = {r.document.id: r.rank for r in semantic_results}
    
    # Normalize scores to 0-1 range
    if keyword_scores:
        max_kw = max(keyword_scores.values())
        if max_kw > 0:
            keyword_scores = {k: v / max_kw for k, v in keyword_scores.items()}
    
    if semantic_scores:
        max_sem = max(semantic_scores.values())
        if max_sem > 0:
            semantic_scores = {k: v / max_sem for k, v in semantic_scores.items()}
    
    # Combine scores
    combined_scores = []
    
    for doc in documents:
        doc_id = doc.id
        kw_score = keyword_scores.get(doc_id, 0.0)
        sem_score = semantic_scores.get(doc_id, 0.0)
        
        if method_str == "weighted":
            # Weighted combination
            score = keyword_weight * kw_score + semantic_weight * sem_score
            
        elif method_str == "rrf":
            # Reciprocal Rank Fusion
            k = 60  # RRF constant
            kw_rank = keyword_ranks.get(doc_id, len(documents) + 1)
            sem_rank = semantic_ranks.get(doc_id, len(documents) + 1)
            score = (1.0 / (k + kw_rank)) + (1.0 / (k + sem_rank))
        
        elif method_str in ("dbsf", "normalized"):
            # Distribution-Based Score Fusion or Normalized
            score = (kw_score + sem_score) / 2.0
            
        elif method_str == "max":
            # Take maximum score
            score = max(kw_score, sem_score)
        else:
            score = kw_score + sem_score
        
        combined_scores.append((doc, score))
    
    # Sort by combined score
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Create search results
    results = []
    for rank, (doc, score) in enumerate(combined_scores[:top_k], 1):
        results.append(SearchResult(
            document=doc,
            score=score,
            rank=rank,
            method="hybrid"
        ))
    
    return results
