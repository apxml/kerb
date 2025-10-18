"""Vector operations for embeddings."""

import math
from typing import List


def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize a vector to unit length (L2 norm = 1).
    
    Args:
        vector (List[float]): Input vector
        
    Returns:
        List[float]: Normalized vector
    """
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


def vector_magnitude(vector: List[float]) -> float:
    """Calculate the magnitude (L2 norm) of a vector.
    
    Args:
        vector (List[float]): Input vector
        
    Returns:
        float: Vector magnitude
    """
    return math.sqrt(sum(x * x for x in vector))


def mean_pooling(vectors: List[List[float]]) -> List[float]:
    """Calculate the mean of multiple vectors (centroid).
    
    Useful for averaging embeddings of multiple texts.
    
    Args:
        vectors (List[List[float]]): List of vectors to average
        
    Returns:
        List[float]: Mean vector
        
    Examples:
        from kerb.embedding import embed_batch
        # Average embeddings of multiple sentences
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embed_batch(sentences)
        avg_embedding = mean_pooling(embeddings)
    """
    if not vectors:
        return []
    
    dim = len(vectors[0])
    result = [0.0] * dim
    
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError("All vectors must have same dimensions")
        for i, val in enumerate(vec):
            result[i] += val
    
    n = len(vectors)
    return [x / n for x in result]


def weighted_mean_pooling(vectors: List[List[float]], weights: List[float]) -> List[float]:
    """Calculate weighted mean of multiple vectors.
    
    Args:
        vectors (List[List[float]]): List of vectors
        weights (List[float]): Weight for each vector (will be normalized)
        
    Returns:
        List[float]: Weighted mean vector
        
    Examples:
        from kerb.embedding import embed_batch
        embeddings = embed_batch(["important", "less important"])
        weighted_avg = weighted_mean_pooling(embeddings, weights=[0.8, 0.2])
    """
    if not vectors or not weights:
        return []
    
    if len(vectors) != len(weights):
        raise ValueError("Number of vectors and weights must match")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")
    norm_weights = [w / total_weight for w in weights]
    
    dim = len(vectors[0])
    result = [0.0] * dim
    
    for vec, weight in zip(vectors, norm_weights):
        if len(vec) != dim:
            raise ValueError("All vectors must have same dimensions")
        for i, val in enumerate(vec):
            result[i] += val * weight
    
    return result


def max_pooling(vectors: List[List[float]]) -> List[float]:
    """Apply max pooling across multiple vectors (element-wise maximum).
    
    Args:
        vectors (List[List[float]]): List of vectors
        
    Returns:
        List[float]: Max-pooled vector
    """
    if not vectors:
        return []
    
    dim = len(vectors[0])
    result = list(vectors[0])
    
    for vec in vectors[1:]:
        if len(vec) != dim:
            raise ValueError("All vectors must have same dimensions")
        for i, val in enumerate(vec):
            result[i] = max(result[i], val)
    
    return result
