"""
Core enumerations for the kerb library.

This module contains enum definitions for strategy parameters, methods, and modes
used across multiple packages to provide type safety, IDE autocomplete, and prevent typos.
"""

from enum import Enum


# ============================================================================
# Agent Package Enums
# ============================================================================

class ChainStrategy(Enum):
    """Strategy for executing chain steps."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    DYNAMIC = "dynamic"


class ToolResultFormat(Enum):
    """Format for tool result output."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


class TruncatePosition(Enum):
    """Position to truncate context from."""
    START = "start"
    MIDDLE = "middle"
    END = "end"
    SMART = "smart"


# ============================================================================
# Fine-Tuning Package Enums
# ============================================================================

class BalanceMethod(Enum):
    """Method for balancing datasets."""
    UNDERSAMPLE = "undersample"
    OVERSAMPLE = "oversample"
    SMOTE = "smote"
    NONE = "none"


# ============================================================================
# Context Package Enums
# ============================================================================

class CompressionStrategy(Enum):
    """Strategy for compressing context."""
    TOP_K = "top_k"
    SUMMARIZE = "summarize"
    FILTER = "filter"
    TRUNCATE = "truncate"


class ReorderStrategy(Enum):
    """Strategy for reordering context items."""
    CHRONOLOGICAL = "chronological"
    PRIORITY = "priority"
    RELEVANCE = "relevance"
    ALTERNATING = "alternating"


# ============================================================================
# Memory Package Enums
# ============================================================================

class PruneStrategy(Enum):
    """Strategy for pruning conversation buffer."""
    OLDEST = "oldest"
    NEWEST = "newest"
    LEAST_RELEVANT = "least_relevant"
    MOST_RELEVANT = "most_relevant"
    TOKEN_LIMIT = "token_limit"


class SummaryStrategy(Enum):
    """Strategy for summarizing conversations."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    COMBINED = "combined"


# ============================================================================
# Preprocessing Package Enums
# ============================================================================

class TruncateStrategy(Enum):
    """Strategy for truncating text."""
    START = "start"
    END = "end"
    MIDDLE = "middle"
    SMART = "smart"


class CaseMode(Enum):
    """Case normalization mode."""
    LOWER = "lower"
    UPPER = "upper"
    TITLE = "title"
    SENTENCE = "sentence"


# ============================================================================
# Prompt Package Enums
# ============================================================================

class SelectionStrategy(Enum):
    """Strategy for selecting few-shot examples."""
    RANDOM = "random"
    FIRST = "first"
    LAST = "last"
    SIMILARITY = "similarity"
    SEMANTIC = "semantic"  # Alias for similarity
    DIVERSE = "diverse"
    RECENT = "recent"
    FIXED = "fixed"


class VersionSelectionStrategy(Enum):
    """Strategy for selecting prompt versions."""
    RANDOM = "random"
    LATEST = "latest"
    BEST_PERFORMING = "best_performing"
    A_B_TEST = "a_b_test"


# ============================================================================
# Retrieval Package Enums
# ============================================================================

class ExpansionMethod(Enum):
    """Method for expanding queries."""
    SYNONYMS = "synonyms"
    RELATED_TERMS = "related_terms"
    LLM = "llm"
    EMBEDDINGS = "embeddings"


class FusionMethod(Enum):
    """Method for fusing hybrid search results."""
    WEIGHTED = "weighted"
    RRF = "rrf"  # Reciprocal Rank Fusion
    DBSF = "dbsf"  # Distribution-Based Score Fusion
    NORMALIZED = "normalized"


class RerankMethod(Enum):
    """Method for reranking search results."""
    RELEVANCE = "relevance"
    DIVERSITY = "diversity"
    MMR = "mmr"  # Maximal Marginal Relevance
    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"
    RECENCY = "recency"
    POPULARITY = "popularity"
    CUSTOM = "custom"


class QueryStyle(Enum):
    """Style for query rewriting."""
    CLEAR = "clear"
    DETAILED = "detailed"
    CONCISE = "concise"
    KEYWORD = "keyword"
    NATURAL = "natural"


# ============================================================================
# Evaluation Package Enums
# ============================================================================

class FaithfulnessMethod(Enum):
    """Method for assessing faithfulness."""
    ENTAILMENT = "entailment"
    NLI = "nli"  # Natural Language Inference
    FACT_CHECK = "fact_check"
    LLM = "llm"


class SimilarityMethod(Enum):
    """Method for calculating semantic similarity."""
    EMBEDDING = "embedding"
    COSINE = "cosine"
    JACCARD = "jaccard"
    BLEU = "bleu"
    ROUGE = "rouge"
    BERTSCORE = "bertscore"


# ============================================================================
# Chunking Package Enums
# ============================================================================

class ChunkingStrategy(Enum):
    """Strategy for chunking text."""
    SIMPLE = "simple"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    TOKEN = "token"
    SLIDING_WINDOW = "sliding_window"


# ============================================================================
# Parsing Package Enums
# ============================================================================

class ParseMode(Enum):
    """Mode for parsing JSON/code from text."""
    STRICT = "strict"
    LENIENT = "lenient"
    BEST_EFFORT = "best_effort"
    MARKDOWN_AWARE = "markdown_aware"


# ============================================================================
# Generation Package Enums
# ============================================================================

class StreamingMode(Enum):
    """Mode for streaming generation."""
    NONE = "none"
    TOKENS = "tokens"
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"


class StopCondition(Enum):
    """Condition for stopping generation."""
    MAX_TOKENS = "max_tokens"
    END_OF_TEXT = "end_of_text"
    STOP_SEQUENCE = "stop_sequence"
    CUSTOM = "custom"


# ============================================================================
# Cache Package Enums
# ============================================================================

class SizeUnit(Enum):
    """Unit for measuring cache size."""
    ENTRIES = "entries"
    BYTES = "bytes"
    KB = "kb"
    MB = "mb"
    GB = "gb"


class ExportFormat(Enum):
    """Format for exporting cache statistics."""
    DICT = "dict"
    JSON = "json"
    CSV = "csv"
    TABLE = "table"


# ============================================================================
# Multimodal and Fine-tuning Package Enums
# ============================================================================

class Device(Enum):
    """Compute device for model inference."""
    CPU = "cpu"
    CUDA = "cuda"
    CUDA_0 = "cuda:0"
    CUDA_1 = "cuda:1"
    CUDA_2 = "cuda:2"
    MPS = "mps"  # Apple Silicon


class MessageFormat(Enum):
    """Format for displaying messages."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    MARKDOWN = "markdown"


class ResultFormat(Enum):
    """Format for displaying results."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    MARKDOWN = "markdown"


# ============================================================================
# Utility function for enum validation
# ============================================================================

def validate_enum_or_string(value, enum_class, param_name: str):
    """
    Validate that a value is either a valid enum member or a string matching an enum value.
    
    Args:
        value: The value to validate (can be enum or string)
        enum_class: The enum class to validate against
        param_name: Name of the parameter (for error messages)
        
    Returns:
        The enum member if valid, or the original string
        
    Raises:
        ValueError: If the value is a string but doesn't match any enum value
        
    Examples:
        >>> validate_enum_or_string("sequential", ChainStrategy, "strategy")
        <ChainStrategy.SEQUENTIAL: 'sequential'>
        
        >>> validate_enum_or_string(ChainStrategy.PARALLEL, ChainStrategy, "strategy")
        <ChainStrategy.PARALLEL: 'parallel'>
        
        >>> validate_enum_or_string("custom_strategy", ChainStrategy, "strategy")
        'custom_strategy'  # Allows custom values
    """
    # If it's already an enum member, return it
    if isinstance(value, enum_class):
        return value
    
    # If it's a string, try to convert to enum
    if isinstance(value, str):
        try:
            # Try exact match first
            return enum_class(value)
        except ValueError:
            # Try case-insensitive match
            for member in enum_class:
                if member.value.lower() == value.lower():
                    return member
            
            # If no match, allow custom string but warn
            import warnings
            valid_values = [m.value for m in enum_class]
            warnings.warn(
                f"Unknown {param_name} value: '{value}'. "
                f"Valid values are: {valid_values}. "
                f"Using custom value '{value}'.",
                UserWarning,
                stacklevel=3
            )
            return value
    
    raise TypeError(f"{param_name} must be a {enum_class.__name__} or string, got {type(value)}")


__all__ = [
    # Agent enums
    "ChainStrategy",
    "ToolResultFormat",
    "TruncatePosition",
    
    # Fine-tuning enums
    "BalanceMethod",
    
    # Context enums
    "CompressionStrategy",
    "ReorderStrategy",
    
    # Memory enums
    "PruneStrategy",
    "SummaryStrategy",
    
    # Preprocessing enums
    "TruncateStrategy",
    "CaseMode",
    
    # Prompt enums
    "SelectionStrategy",
    "VersionSelectionStrategy",
    
    # Retrieval enums
    "ExpansionMethod",
    "FusionMethod",
    "RerankMethod",
    "QueryStyle",
    
    # Evaluation enums
    "FaithfulnessMethod",
    "SimilarityMethod",
    
    # Chunking enums
    "ChunkingStrategy",
    
    # Parsing enums
    "ParseMode",
    
    # Generation enums
    "StreamingMode",
    "StopCondition",
    
    # Cache enums
    "SizeUnit",
    "ExportFormat",
    
    # Multimodal/Fine-tuning enums
    "Device",
    "MessageFormat",
    "ResultFormat",
    
    # Utility
    "validate_enum_or_string",
]
