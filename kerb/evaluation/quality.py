"""Quality assessment functions for generated text.

This module provides functions to assess various quality aspects of generated text:
- Coherence: Logical flow and structure
- Fluency: Naturalness and readability
- Faithfulness: Alignment with source material
- Answer Relevance: Relevance to questions
- Hallucination Detection: Unfounded claims
"""

import re
import statistics
from collections import Counter
from typing import TYPE_CHECKING, List, Union

from .metrics import calculate_semantic_similarity
from .types import EvaluationResult

if TYPE_CHECKING:
    from kerb.core.enums import FaithfulnessMethod


# ============================================================================
# Quality Assessment Functions
# ============================================================================


def assess_coherence(text: str) -> EvaluationResult:
    """Assess the coherence and logical flow of text.

    Args:
        text: Text to assess

    Returns:
        EvaluationResult: Coherence score and details

    Example:
        >>> result = assess_coherence("First point. Second point follows. Conclusion makes sense.")
        >>> result.score > 0.7
        True
    """
    if not text:
        return EvaluationResult(metric="coherence", score=0.0)

    sentences = _split_sentences(text)

    if len(sentences) <= 1:
        # Single sentence is coherent by default
        return EvaluationResult(metric="coherence", score=1.0, details={"sentences": 1})

    # Heuristics for coherence
    score = 1.0
    issues = []

    # Check for transition words
    transition_words = {
        "however",
        "therefore",
        "furthermore",
        "moreover",
        "additionally",
        "consequently",
        "thus",
        "hence",
        "meanwhile",
        "similarly",
        "first",
        "second",
        "third",
        "finally",
        "next",
        "then",
    }

    has_transitions = any(
        any(word in sent.lower() for word in transition_words) for sent in sentences
    )

    if not has_transitions and len(sentences) > 3:
        score -= 0.15
        issues.append("Few transition words")

    # Check for repeated sentence structures (good for coherence)
    avg_sent_length = statistics.mean(len(s.split()) for s in sentences)
    sent_length_variance = (
        statistics.variance(len(s.split()) for s in sentences)
        if len(sentences) > 1
        else 0
    )

    if sent_length_variance > avg_sent_length * 2:
        score -= 0.1
        issues.append("High sentence length variance")

    # Check for pronoun usage (indicates reference to previous content)
    pronouns = {"it", "they", "this", "that", "these", "those", "he", "she"}
    pronoun_usage = sum(
        1
        for sent in sentences[1:]  # Skip first sentence
        for word in sent.lower().split()
        if word in pronouns
    )

    if len(sentences) > 2 and pronoun_usage == 0:
        score -= 0.1
        issues.append("No pronouns referencing previous content")

    return EvaluationResult(
        metric="coherence",
        score=max(0.0, score),
        details={
            "sentences": len(sentences),
            "issues": issues,
            "has_transitions": has_transitions,
        },
    )


def assess_fluency(text: str) -> EvaluationResult:
    """Assess the fluency and naturalness of text.

    Args:
        text: Text to assess

    Returns:
        EvaluationResult: Fluency score and details

    Example:
        >>> result = assess_fluency("This is a well-written sentence.")
        >>> result.score > 0.8
        True
    """
    if not text:
        return EvaluationResult(metric="fluency", score=0.0)

    score = 1.0
    issues = []

    # Check for repetitive words
    words = text.lower().split()
    if len(words) > 0:
        word_freq = Counter(words)
        # Exclude common words
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        significant_words = {
            w: c for w, c in word_freq.items() if w not in common_words and len(w) > 3
        }

        max_repetition = max(significant_words.values()) if significant_words else 0
        avg_length = len(words)

        if max_repetition > avg_length / 10:  # Word repeated more than 10% of text
            score -= 0.2
            issues.append("Excessive word repetition")

    # Check for incomplete sentences
    sentences = _split_sentences(text)
    incomplete = sum(1 for s in sentences if len(s.split()) < 3)

    if incomplete > len(sentences) / 3:
        score -= 0.15
        issues.append("Many incomplete sentences")

    # Check for grammar patterns (very basic)
    # Look for common errors
    if re.search(r"\b(a)\s+([aeiou])", text.lower()):  # "a apple" instead of "an apple"
        score -= 0.1
        issues.append("Article agreement errors")

    # Check for excessive punctuation
    punct_count = sum(1 for c in text if c in "!?.,:;")
    word_count = len(words)

    if word_count > 0 and punct_count / word_count > 0.3:
        score -= 0.1
        issues.append("Excessive punctuation")

    return EvaluationResult(
        metric="fluency",
        score=max(0.0, score),
        details={"issues": issues, "sentences": len(sentences)},
    )


def detect_hallucination(
    output: str, context: str, threshold: float = 0.3
) -> EvaluationResult:
    """Detect potential hallucinations (unfounded claims not supported by context).

    Args:
        output: Generated text to check
        context: Source context that should support the output
        threshold: Threshold for hallucination detection (lower = stricter)

    Returns:
        EvaluationResult: Hallucination score (0 = no hallucination, 1 = likely hallucination)

    Example:
        >>> result = detect_hallucination(
        ...     "Paris is the capital of Germany",
        ...     "Paris is the capital of France"
        ... )
        >>> result.score > 0.5
        True
    """
    if not output or not context:
        return EvaluationResult(metric="hallucination", score=0.0)

    # Extract key entities and facts from output
    output_sentences = _split_sentences(output)
    context_lower = context.lower()

    unsupported_sentences = 0
    total_sentences = len(output_sentences)

    details = []

    for sent in output_sentences:
        # Check if sentence content appears in context
        sent_words = set(sent.lower().split())
        # Remove common words
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
        }
        significant_words = sent_words - common_words

        if not significant_words:
            continue

        # Check how many significant words appear in context
        words_in_context = sum(1 for word in significant_words if word in context_lower)
        support_ratio = (
            words_in_context / len(significant_words) if significant_words else 1.0
        )

        if support_ratio < threshold:
            unsupported_sentences += 1
            details.append(
                f"Unsupported: '{sent[:50]}...' (support: {support_ratio:.2f})"
            )

    hallucination_score = (
        unsupported_sentences / total_sentences if total_sentences > 0 else 0.0
    )

    return EvaluationResult(
        metric="hallucination",
        score=hallucination_score,
        details={
            "unsupported_sentences": unsupported_sentences,
            "total_sentences": total_sentences,
            "examples": details[:3],  # Limit to 3 examples
        },
        passed=hallucination_score < 0.3,
    )


def assess_faithfulness(
    output: str, source: str, method: Union["FaithfulnessMethod", str] = "entailment"
) -> EvaluationResult:
    """Assess whether output is faithful to the source material.

    Args:
        output: Generated text
        source: Source material
        method: Assessment method (FaithfulnessMethod enum or string: "entailment", "nli", "fact_check", "llm")

    Returns:
        EvaluationResult: Faithfulness score (1 = fully faithful, 0 = not faithful)

    Examples:
        >>> from kerb.core.enums import FaithfulnessMethod
        >>> result = assess_faithfulness(output, source, method=FaithfulnessMethod.ENTAILMENT)
        >>> result.score > 0.7
        True
    """
    from kerb.core.enums import FaithfulnessMethod, validate_enum_or_string

    if not output or not source:
        return EvaluationResult(metric="faithfulness", score=0.0)

    # Validate and normalize method
    method_val = validate_enum_or_string(method, FaithfulnessMethod, "method")
    if isinstance(method_val, FaithfulnessMethod):
        method_str = method_val.value
    else:
        method_str = method_val

    if method_str == "overlap":
        # Token overlap method
        output_words = set(output.lower().split())
        source_words = set(source.lower().split())

        if not output_words:
            return EvaluationResult(metric="faithfulness", score=0.0)

        overlap = output_words & source_words
        faithfulness_score = len(overlap) / len(output_words)

        return EvaluationResult(
            metric="faithfulness",
            score=faithfulness_score,
            details={"method": "overlap", "overlap_tokens": len(overlap)},
        )

    elif method_str == "semantic":
        # Use semantic similarity
        similarity = calculate_semantic_similarity(output, source, method="tfidf")

        return EvaluationResult(
            metric="faithfulness", score=similarity, details={"method": "semantic"}
        )

    elif method_str in ("entailment", "nli"):
        # Simple entailment check (heuristic-based)
        # Check if all key claims in output are supported by source
        output_sents = _split_sentences(output)
        source_lower = source.lower()

        supported = 0
        for sent in output_sents:
            # Extract key terms
            sent_words = set(sent.lower().split())
            common_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
            }
            key_terms = sent_words - common_words

            # Check if most key terms are in source
            terms_in_source = sum(1 for term in key_terms if term in source_lower)
            if key_terms and terms_in_source / len(key_terms) > 0.5:
                supported += 1

        faithfulness_score = supported / len(output_sents) if output_sents else 0.0

        return EvaluationResult(
            metric="faithfulness",
            score=faithfulness_score,
            details={
                "method": "entailment",
                "supported_sentences": supported,
                "total_sentences": len(output_sents),
            },
        )

    else:
        raise ValueError(f"Unknown faithfulness method: {method}")


def assess_answer_relevance(
    answer: str, question: str, threshold: float = 0.3
) -> EvaluationResult:
    """Assess whether an answer is relevant to the question.

    Args:
        answer: The answer text
        question: The question text
        threshold: Minimum overlap threshold

    Returns:
        EvaluationResult: Relevance score

    Example:
        >>> result = assess_answer_relevance(
        ...     "Python is a programming language",
        ...     "What is Python?"
        ... )
        >>> result.score > 0.5
        True
    """
    if not answer or not question:
        return EvaluationResult(metric="answer_relevance", score=0.0)

    # Extract key terms from question
    question_words = set(question.lower().split())
    # Remove question words
    question_stopwords = {
        "what",
        "when",
        "where",
        "why",
        "how",
        "who",
        "which",
        "is",
        "are",
        "the",
        "a",
        "an",
    }
    key_terms = question_words - question_stopwords

    # Check presence in answer
    answer_lower = answer.lower()
    terms_in_answer = sum(1 for term in key_terms if term in answer_lower)

    if not key_terms:
        # If no key terms, use semantic similarity
        relevance = calculate_semantic_similarity(answer, question, method="jaccard")
    else:
        relevance = terms_in_answer / len(key_terms)

    # Boost score if answer is substantive
    answer_length = len(answer.split())
    if answer_length > 10:
        relevance = min(1.0, relevance * 1.1)

    return EvaluationResult(
        metric="answer_relevance",
        score=relevance,
        details={"key_terms": list(key_terms), "terms_found": terms_in_answer},
        passed=relevance >= threshold,
    )


# ============================================================================
# Helper Functions
# ============================================================================


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]
