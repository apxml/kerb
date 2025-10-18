"""Ground truth comparison metrics for evaluation.

This module provides metrics for comparing generated text against reference texts:
- BLEU: N-gram overlap with brevity penalty
- ROUGE: Recall-oriented n-gram and subsequence matching
- METEOR: Precision, recall, and word order
- Exact Match: Binary exact string matching
- F1 Score: Token-level precision and recall
- Semantic Similarity: Embedding-based and lexical similarity
"""

import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from kerb.core.enums import SimilarityMethod


# ============================================================================
# Public Metrics
# ============================================================================


def calculate_bleu(
    candidate: str,
    reference: Union[str, List[str]],
    n: int = 4,
    weights: Optional[List[float]] = None,
) -> float:
    """Calculate BLEU score between candidate and reference text(s).

    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap with brevity penalty.

    Args:
        candidate: The generated text to evaluate
        reference: Reference text(s) (ground truth)
        n: Maximum n-gram length (default: 4 for BLEU-4)
        weights: Weights for each n-gram (default: equal weights)

    Returns:
        float: BLEU score between 0 and 1

    Example:
        >>> calculate_bleu("the cat sat", "the cat sat on mat")
        0.7598
    """
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference

    if not candidate or not references:
        return 0.0

    # Tokenize
    candidate_tokens = candidate.lower().split()
    reference_tokens_list = [ref.lower().split() for ref in references]

    if not candidate_tokens:
        return 0.0

    # Default weights
    if weights is None:
        weights = [1.0 / n] * n

    # Calculate n-gram precisions
    precisions = []
    for i in range(1, n + 1):
        candidate_ngrams = _get_ngrams(candidate_tokens, i)

        # Get maximum counts from all references
        max_ref_counts: Dict[Tuple, int] = {}
        for ref_tokens in reference_tokens_list:
            ref_ngrams = _get_ngrams(ref_tokens, i)
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)

        # Calculate clipped counts
        clipped_count = 0
        total_count = 0
        for ngram, count in candidate_ngrams.items():
            clipped_count += min(count, max_ref_counts.get(ngram, 0))
            total_count += count

        precision = clipped_count / total_count if total_count > 0 else 0.0
        precisions.append(precision)

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        geo_mean = 0.0
    else:
        log_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
        geo_mean = math.exp(log_sum)

    # Brevity penalty
    candidate_len = len(candidate_tokens)
    ref_len = min(len(ref) for ref in reference_tokens_list)

    if candidate_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / candidate_len) if candidate_len > 0 else 0.0

    return bp * geo_mean


def calculate_rouge(
    candidate: str,
    reference: Union[str, List[str]],
    rouge_type: str = "rouge-l",
    beta: float = 1.2,
) -> Dict[str, float]:
    """Calculate ROUGE scores between candidate and reference text(s).

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures recall of n-grams.

    Args:
        candidate: The generated text to evaluate
        reference: Reference text(s) (ground truth)
        rouge_type: Type of ROUGE - "rouge-1", "rouge-2", "rouge-l"
        beta: Beta parameter for F-measure (default: 1.2 favors recall)

    Returns:
        dict: Dictionary with 'precision', 'recall', 'fmeasure' scores

    Example:
        >>> calculate_rouge("the cat sat", "the cat sat on mat", "rouge-1")
        {'precision': 1.0, 'recall': 0.6, 'fmeasure': 0.75}
    """
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference

    if not candidate or not references:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    candidate_tokens = candidate.lower().split()

    if rouge_type == "rouge-1":
        return _rouge_n(candidate_tokens, references, 1, beta)
    elif rouge_type == "rouge-2":
        return _rouge_n(candidate_tokens, references, 2, beta)
    elif rouge_type == "rouge-l":
        return _rouge_l(candidate_tokens, references, beta)
    else:
        raise ValueError(f"Unknown ROUGE type: {rouge_type}")


def calculate_meteor(
    candidate: str,
    reference: Union[str, List[str]],
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    """Calculate METEOR score (simplified version without stemming/synonyms).

    METEOR considers precision, recall, and word order with harmonic mean.

    Args:
        candidate: The generated text to evaluate
        reference: Reference text(s) (ground truth)
        alpha: Weight for recall vs precision (default: 0.9)
        beta: Shape parameter for f-mean (default: 3.0)
        gamma: Penalty weight for fragmentation (default: 0.5)

    Returns:
        float: METEOR score between 0 and 1

    Example:
        >>> calculate_meteor("the cat sat", "the cat sat on mat")
        0.833
    """
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference

    if not candidate or not references:
        return 0.0

    candidate_tokens = candidate.lower().split()

    # Calculate against best reference
    best_score = 0.0
    for ref in references:
        ref_tokens = ref.lower().split()

        # Find matches
        matches = _find_matches(candidate_tokens, ref_tokens)
        num_matches = len(matches)

        if num_matches == 0:
            continue

        # Precision and recall
        precision = num_matches / len(candidate_tokens) if candidate_tokens else 0.0
        recall = num_matches / len(ref_tokens) if ref_tokens else 0.0

        # F-mean
        if precision + recall > 0:
            fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        else:
            fmean = 0.0

        # Fragmentation penalty
        chunks = _count_chunks(matches)
        fragmentation = chunks / num_matches if num_matches > 0 else 1.0
        penalty = gamma * (fragmentation**beta)

        score = fmean * (1 - penalty)
        best_score = max(best_score, score)

    return best_score


def calculate_exact_match(candidate: str, reference: Union[str, List[str]]) -> float:
    """Calculate exact match score (1.0 if exact match, 0.0 otherwise).

    Args:
        candidate: The generated text to evaluate
        reference: Reference text(s) (ground truth)

    Returns:
        float: 1.0 if exact match, 0.0 otherwise

    Example:
        >>> calculate_exact_match("Paris", "Paris")
        1.0
    """
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference

    candidate_normalized = candidate.strip().lower()

    for ref in references:
        if candidate_normalized == ref.strip().lower():
            return 1.0

    return 0.0


def calculate_f1_score(candidate: str, reference: Union[str, List[str]]) -> float:
    """Calculate token-level F1 score.

    Args:
        candidate: The generated text to evaluate
        reference: Reference text(s) (ground truth)

    Returns:
        float: F1 score between 0 and 1

    Example:
        >>> calculate_f1_score("the cat sat", "the cat sat on mat")
        0.857
    """
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference

    if not candidate or not references:
        return 0.0

    candidate_tokens = set(candidate.lower().split())

    best_f1 = 0.0
    for ref in references:
        ref_tokens = set(ref.lower().split())

        if not candidate_tokens or not ref_tokens:
            continue

        common = candidate_tokens & ref_tokens

        if not common:
            continue

        precision = len(common) / len(candidate_tokens)
        recall = len(common) / len(ref_tokens)

        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def calculate_semantic_similarity(
    text1: str, text2: str, method: Union["SimilarityMethod", str] = "embedding"
) -> float:
    """Calculate semantic similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        method: Similarity method (SimilarityMethod enum or string: "embedding", "cosine", "jaccard", "bleu", "rouge", "bertscore")

    Returns:
        float: Similarity score between 0 and 1

    Examples:
        >>> # Using enum (recommended)
        >>> from kerb.core.enums import SimilarityMethod
        >>> score = calculate_semantic_similarity(text1, text2, method=SimilarityMethod.EMBEDDING)

        >>> # Using string (for backward compatibility)
        >>> calculate_semantic_similarity("cat", "kitten", method="jaccard")
        0.0
        >>> calculate_semantic_similarity("the cat sat", "the cat sits", method="jaccard")
        0.5
    """
    from kerb.core.enums import SimilarityMethod, validate_enum_or_string

    if not text1 or not text2:
        return 0.0

    # Validate and normalize method
    method_val = validate_enum_or_string(method, SimilarityMethod, "method")
    if isinstance(method_val, SimilarityMethod):
        method_str = method_val.value
    else:
        method_str = method_val

    if method_str in ("embedding", "cosine"):
        # Try to use embedding module if available
        try:
            from ..embedding import cosine_similarity, embed

            emb1 = embed(text1)
            emb2 = embed(text2)
            # Convert cosine similarity from [-1, 1] to [0, 1]
            return (cosine_similarity(emb1, emb2) + 1) / 2
        except ImportError:
            # Fall back to Jaccard if embeddings not available
            method_str = "jaccard"

    if method_str == "jaccard":
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union) if union else 0.0

    elif method == "tfidf":
        # Simple TF-IDF cosine similarity
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()

        # Calculate TF
        tf1 = Counter(tokens1)
        tf2 = Counter(tokens2)

        # Get all unique terms
        all_terms = set(tf1.keys()) | set(tf2.keys())

        # Simple IDF (just use log of inverse frequency)
        total_docs = 2
        doc_freq = {
            term: (1 if term in tf1 else 0) + (1 if term in tf2 else 0)
            for term in all_terms
        }
        idf = {
            term: math.log(total_docs / freq) if freq > 0 else 0
            for term, freq in doc_freq.items()
        }

        # Calculate TF-IDF vectors
        vec1 = [tf1[term] * idf[term] for term in all_terms]
        vec2 = [tf2[term] * idf[term] for term in all_terms]

        # Cosine similarity
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(v * v for v in vec1))
        magnitude2 = math.sqrt(sum(v * v for v in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    else:
        raise ValueError(f"Unknown similarity method: {method}")


# ============================================================================
# Helper Functions
# ============================================================================


def _get_ngrams(tokens: List[str], n: int) -> Dict[Tuple, int]:
    """Get n-grams from tokens."""
    ngrams: Dict[Tuple, int] = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        ngrams[ngram] += 1
    return dict(ngrams)


def _rouge_n(
    candidate_tokens: List[str], references: List[str], n: int, beta: float
) -> Dict[str, float]:
    """Calculate ROUGE-N scores."""
    candidate_ngrams = _get_ngrams(candidate_tokens, n)

    # Get maximum counts from all references
    max_ref_counts: Dict[Tuple, int] = {}
    for ref in references:
        ref_tokens = ref.lower().split()
        ref_ngrams = _get_ngrams(ref_tokens, n)
        for ngram, count in ref_ngrams.items():
            max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)

    # Calculate matches
    matches = sum(
        min(count, max_ref_counts.get(ngram, 0))
        for ngram, count in candidate_ngrams.items()
    )

    candidate_count = sum(candidate_ngrams.values())
    reference_count = sum(max_ref_counts.values())

    if candidate_count == 0 or reference_count == 0:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    precision = matches / candidate_count
    recall = matches / reference_count

    if precision + recall > 0:
        fmeasure = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    else:
        fmeasure = 0.0

    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


def _rouge_l(
    candidate_tokens: List[str], references: List[str], beta: float
) -> Dict[str, float]:
    """Calculate ROUGE-L (longest common subsequence) scores."""
    best_lcs = 0
    best_ref_len = 0

    for ref in references:
        ref_tokens = ref.lower().split()
        lcs_length = _longest_common_subsequence(candidate_tokens, ref_tokens)
        if lcs_length > best_lcs:
            best_lcs = lcs_length
            best_ref_len = len(ref_tokens)

    if len(candidate_tokens) == 0 or best_ref_len == 0:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    precision = best_lcs / len(candidate_tokens)
    recall = best_lcs / best_ref_len

    if precision + recall > 0:
        fmeasure = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    else:
        fmeasure = 0.0

    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


def _longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """Calculate length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def _find_matches(tokens1: List[str], tokens2: List[str]) -> List[int]:
    """Find matching token indices."""
    matches = []
    used = set()

    for i, token in enumerate(tokens1):
        for j, ref_token in enumerate(tokens2):
            if j not in used and token == ref_token:
                matches.append(i)
                used.add(j)
                break

    return matches


def _count_chunks(matches: List[int]) -> int:
    """Count number of contiguous chunks in matches."""
    if not matches:
        return 0

    chunks = 1
    for i in range(1, len(matches)):
        if matches[i] != matches[i - 1] + 1:
            chunks += 1

    return chunks
