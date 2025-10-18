"""Document loading and processing utilities for LLM applications.

This module provides comprehensive tools for working with various document formats.

Common Usage:
    # Load documents (top-level)
    from kerb.document import load_document, Document
    
    # Specialized loaders (submodule)
    from kerb.document.loaders import load_text, load_markdown
    
    # Utilities (submodule)
    from kerb.document.utils import detect_format, load_directory
    
    # Text processing (submodules)
    from kerb.document.extractors import extract_text_from_html
    from kerb.document.cleaners import clean_text
    from kerb.document.preprocessors import preprocess_pdf_text
    from kerb.document.metadata import extract_metadata

Document Loading:
    load_document() - Load any supported document (auto-detects format)
    
Submodules:
    loaders - Format-specific document loaders (PDF, DOCX, HTML, etc.)
    utils - Utilities for format detection, batch loading, and merging
    extractors - Text extraction from various formats
    cleaners - Text cleaning and normalization
    preprocessors - Format-specific preprocessing
    metadata - Metadata and entity extraction
    
Data Classes:
    Document - Document with content and metadata (from kerb.core.types)
    DocumentFormat - Enum of supported formats (from kerb.core.types)
"""

# Import core types from central location
from kerb.core.types import Document, DocumentFormat

# Top-level imports: core functionality
from .loaders import (
    load_document,
    load_text,
    load_markdown,
    load_json,
    load_csv,
    load_xml,
    load_html,
    load_pdf,
    load_docx,
)

from .utils import (
    detect_format,
    is_supported_format,
    load_directory,
    load_from_url,
    load_from_url_async,
    merge_documents,
)

from .extractors import (
    extract_text_from_html,
    strip_markdown,
    split_into_sentences,
    split_into_paragraphs,
)

from .cleaners import (
    clean_text,
    remove_extra_newlines,
)

from .preprocessors import (
    preprocess_pdf_text,
    preprocess_html_text,
    preprocess_markdown,
)

from .metadata import (
    extract_metadata,
    extract_document_stats,
    extract_urls,
    extract_emails,
    extract_dates,
    extract_phone_numbers,
)

# Import submodules for user access
from . import loaders
from . import utils
from . import extractors
from . import cleaners
from . import preprocessors
from . import metadata

__all__ = [
    # Core types
    "Document",
    "DocumentFormat",
    
    # Core loading functions
    "load_document",
    "load_text",
    "load_markdown",
    "load_json",
    "load_csv",
    "load_xml",
    "load_html",
    "load_pdf",
    "load_docx",
    
    # Utilities
    "detect_format",
    "is_supported_format",
    "load_directory",
    "load_from_url",
    "load_from_url_async",
    "merge_documents",
    
    # Text extraction
    "extract_text_from_html",
    "strip_markdown",
    "split_into_sentences",
    "split_into_paragraphs",
    
    # Text cleaning
    "clean_text",
    "remove_extra_newlines",
    
    # Preprocessors
    "preprocess_pdf_text",
    "preprocess_html_text",
    "preprocess_markdown",
    
    # Metadata extraction
    "extract_metadata",
    "extract_document_stats",
    "extract_urls",
    "extract_emails",
    "extract_dates",
    "extract_phone_numbers",
    
    # Submodules
    "loaders",
    "utils",
    "extractors",
    "cleaners",
    "preprocessors",
    "metadata",
]
