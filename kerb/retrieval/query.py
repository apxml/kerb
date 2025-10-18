"""Query processing utilities for retrieval.

This module provides functions for query rewriting, expansion, and decomposition.
"""

import re
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from kerb.core.enums import ExpansionMethod, QueryStyle


def rewrite_query(
    query: str,
    style: Union["QueryStyle", str] = "clear",
    max_length: Optional[int] = None,
) -> str:
    """Rewrite a query for better retrieval.

    Args:
        query: The original query text
        style: Rewriting style (QueryStyle enum or string: "clear", "detailed", "concise", "keyword", "natural")
        max_length: Maximum length of rewritten query

    Returns:
        str: Rewritten query

    Examples:
        >>> from kerb.core.enums import QueryStyle
        >>> rewritten = rewrite_query("python async", style=QueryStyle.DETAILED)
    """
    from kerb.core.enums import QueryStyle, validate_enum_or_string

    query = query.strip()

    # Validate and normalize style
    style_val = validate_enum_or_string(style, QueryStyle, "style")
    if isinstance(style_val, QueryStyle):
        style_str = style_val.value
    else:
        style_str = style_val

    if style_str == "clear":
        # Remove filler words and simplify
        filler_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        }
        words = query.lower().split()
        words = [w for w in words if w not in filler_words]
        rewritten = " ".join(words)

    elif style_str == "detailed":
        # Add context and specificity
        if "?" not in query:
            rewritten = f"Detailed information about {query}"
        else:
            rewritten = f"Please provide comprehensive details: {query}"

    elif style_str == "keyword":
        # Extract key terms only
        words = query.lower().split()
        # Remove common words
        stop_words = {
            "how",
            "what",
            "when",
            "where",
            "why",
            "who",
            "which",
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
            "from",
            "is",
            "are",
            "was",
            "were",
        }
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        rewritten = " ".join(keywords)

    elif style_str == "concise":
        # Make more concise
        words = query.split()
        # Keep only first few important words
        rewritten = " ".join(words[:5])

    elif style_str == "natural":
        # Convert to natural question format
        if "?" not in query:
            if query.lower().startswith(("how", "what", "when", "where", "why", "who")):
                rewritten = query + "?"
            else:
                rewritten = f"What is {query}?"
        else:
            rewritten = query
    else:
        rewritten = query

    if max_length and len(rewritten) > max_length:
        rewritten = rewritten[:max_length].rsplit(" ", 1)[0]

    return rewritten


def expand_query(
    query: str,
    expansions: Optional[List[str]] = None,
    method: Union["ExpansionMethod", str] = "synonyms",
) -> List[str]:
    """Expand a query into multiple variations for broader retrieval.

    Args:
        query: The original query text
        expansions: Custom expansion terms to add
        method: Expansion method (ExpansionMethod enum or string: "synonyms", "related_terms", "llm", "embeddings")

    Returns:
        List[str]: List of query variations

    Examples:
        >>> from kerb.core.enums import ExpansionMethod
        >>> queries = expand_query("machine learning", method=ExpansionMethod.SYNONYMS)
    """
    from kerb.core.enums import ExpansionMethod, validate_enum_or_string

    variations = [query]

    if expansions:
        variations.extend(expansions)

    # Validate and normalize method
    method_val = validate_enum_or_string(method, ExpansionMethod, "method")
    if isinstance(method_val, ExpansionMethod):
        method_str = method_val.value
    else:
        method_str = method_val

    if method_str == "synonyms":
        # Simple synonym expansion (can be enhanced with a synonym dictionary)
        synonym_map = {
            "ml": ["machine learning", "ML"],
            "ai": ["artificial intelligence", "AI"],
            "llm": ["large language model", "LLM"],
            "api": ["API", "application programming interface"],
            "db": ["database", "DB"],
            "async": ["asynchronous", "async"],
            "auth": ["authentication", "auth"],
            "config": ["configuration", "config"],
        }

        query_lower = query.lower()
        for term, synonyms in synonym_map.items():
            if term in query_lower:
                for syn in synonyms:
                    expanded = re.sub(term, syn, query_lower, flags=re.IGNORECASE)
                    if expanded not in variations:
                        variations.append(expanded)

    elif method_str == "related_terms":
        # Add related terms
        related_map = {
            "python": ["python programming", "python code", "python development"],
            "database": ["database design", "database query", "data storage"],
            "api": ["REST API", "API endpoint", "API integration"],
            "error": ["exception", "bug", "issue"],
        }

        query_lower = query.lower()
        for term, related in related_map.items():
            if term in query_lower:
                variations.extend(related)

    elif method_str in ("llm", "embeddings"):
        # Placeholder for LLM-based or embedding-based expansion
        # In production, you'd call an LLM or use embedding similarity
        # For now, just add the original query
        pass

    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        if v.lower() not in seen:
            seen.add(v.lower())
            unique_variations.append(v)

    return unique_variations


def generate_sub_queries(query: str, max_queries: int = 3) -> List[str]:
    """Generate sub-queries from a complex query for step-by-step retrieval.

    Args:
        query: The original complex query
        max_queries: Maximum number of sub-queries to generate

    Returns:
        List[str]: List of sub-queries

    Example:
        >>> generate_sub_queries("How to implement authentication in a Python FastAPI app?")
        ["What is authentication?", "How to use FastAPI?", "Python authentication methods"]
    """
    sub_queries = []

    # Split on conjunctions
    if " and " in query.lower():
        parts = [p.strip() for p in re.split(r"\band\b", query, flags=re.IGNORECASE)]
        sub_queries.extend(parts[:max_queries])

    # Extract ai
    question_words = ["how", "what", "when", "where", "why", "who", "which"]
    words = query.lower().split()

    # Remove question words to get core concepts
    concepts = [w for w in words if w not in question_words and len(w) > 3]

    # Generate sub-queries from concepts
    if len(concepts) >= 2 and len(sub_queries) < max_queries:
        for i, concept in enumerate(concepts[: max_queries - len(sub_queries)]):
            sub_queries.append(f"What is {concept}?")

    if not sub_queries:
        sub_queries = [query]

    return sub_queries[:max_queries]
