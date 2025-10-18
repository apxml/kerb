"""Document utility functions.

This module provides utility functions for:
- Format detection
- Batch loading (directory, URL)
- Document merging
- Async URL loading
"""

import os
import re
from pathlib import Path
from typing import List, Optional

from kerb.core.types import Document, DocumentFormat


def detect_format(file_path: str) -> DocumentFormat:
    """Detect document format from file extension.

    Args:
        file_path (str): Path to the file

    Returns:
        DocumentFormat: Detected format enum

    Examples:
        >>> detect_format("document.pdf")
        DocumentFormat.PDF

        >>> detect_format("notes.md")
        DocumentFormat.MARKDOWN
    """
    ext = Path(file_path).suffix.lower().lstrip(".")

    format_map = {
        "pdf": DocumentFormat.PDF,
        "docx": DocumentFormat.DOCX,
        "doc": DocumentFormat.DOC,
        "html": DocumentFormat.HTML,
        "htm": DocumentFormat.HTML,
        "md": DocumentFormat.MARKDOWN,
        "markdown": DocumentFormat.MARKDOWN,
        "txt": DocumentFormat.TXT,
        "text": DocumentFormat.TXT,
        "csv": DocumentFormat.CSV,
        "json": DocumentFormat.JSON,
        "xml": DocumentFormat.XML,
        "rtf": DocumentFormat.RTF,
        "odt": DocumentFormat.ODT,
        "epub": DocumentFormat.EPUB,
    }

    return format_map.get(ext, DocumentFormat.UNKNOWN)


def is_supported_format(file_path: str) -> bool:
    """Check if file format is supported.

    Args:
        file_path (str): Path to the file

    Returns:
        bool: True if format is supported
    """
    return detect_format(file_path) != DocumentFormat.UNKNOWN


def load_directory(
    directory_path: str,
    pattern: str = "*",
    recursive: bool = False,
    max_files: Optional[int] = None,
) -> List[Document]:
    """Load all supported documents from a directory.

    Args:
        directory_path (str): Path to directory
        pattern (str): File pattern to match (e.g., "*.pdf", "*.txt")
        recursive (bool): Search subdirectories
        max_files (Optional[int]): Maximum number of files to load

    Returns:
        List[Document]: List of loaded documents

    Examples:
        >>> docs = load_directory("./documents", pattern="*.pdf")
        >>> print(f"Loaded {len(docs)} documents")

        >>> docs = load_directory("./data", recursive=True, max_files=100)
    """
    from .loaders import load_document

    path = Path(directory_path)

    if recursive:
        files = list(path.rglob(pattern))
    else:
        files = list(path.glob(pattern))

    # Filter to supported formats
    files = [f for f in files if f.is_file() and is_supported_format(str(f))]

    if max_files:
        files = files[:max_files]

    documents = []
    for file_path in files:
        try:
            doc = load_document(str(file_path))
            documents.append(doc)
        except Exception as e:
            # Skip files that fail to load
            print(f"Warning: Failed to load {file_path}: {e}")
            continue

    return documents


def load_from_url(
    url: str,
    timeout: int = 30,
    max_size_mb: float = 100,
    max_retries: int = 3,
    **kwargs,
) -> Document:
    """Load document from a URL.

    Requires: requests package

    Args:
        url (str): URL to fetch document from
        timeout (int): Request timeout in seconds. Defaults to 30.
        max_size_mb (float): Maximum file size in MB. Defaults to 100.
        max_retries (int): Maximum number of retry attempts. Defaults to 3.
        **kwargs: Additional arguments for requests.get()

    Returns:
        Document: Loaded document

    Raises:
        ValueError: If content exceeds max_size_mb
        requests.exceptions.Timeout: If request times out
        requests.exceptions.HTTPError: If HTTP error occurs

    Examples:
        >>> doc = load_from_url("https://example.com/document.pdf")
        >>> print(doc.content)

        >>> # Custom timeout and size limit
        >>> doc = load_from_url("https://example.com/large.pdf",
        ...                     timeout=60, max_size_mb=200)
    """
    from .extractors import extract_text_from_html

    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError:
        raise ImportError(
            "URL loading requires requests. " "Install with: pip install requests"
        )

    # Set up retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set default timeout if not in kwargs
    if "timeout" not in kwargs:
        kwargs["timeout"] = timeout

    # Stream the response to check size before downloading
    kwargs["stream"] = True
    response = session.get(url, **kwargs)
    response.raise_for_status()

    # Check content length
    content_length = response.headers.get("content-length")
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        if size_mb > max_size_mb:
            response.close()
            raise ValueError(
                f"Content size ({size_mb:.2f} MB) exceeds maximum "
                f"allowed size ({max_size_mb} MB)"
            )

    # Download content with size check
    max_size_bytes = max_size_mb * 1024 * 1024
    content_bytes = bytearray()

    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            content_bytes.extend(chunk)
            if len(content_bytes) > max_size_bytes:
                response.close()
                raise ValueError(
                    f"Downloaded content exceeds maximum allowed size "
                    f"({max_size_mb} MB)"
                )

    # Decode content
    content = content_bytes.decode(response.encoding or "utf-8", errors="replace")

    # Determine format from URL or content-type
    content_type = response.headers.get("content-type", "").lower()

    if "html" in content_type:
        content = extract_text_from_html(content)
        fmt = DocumentFormat.HTML
    elif "json" in content_type:
        fmt = DocumentFormat.JSON
    elif "xml" in content_type:
        fmt = DocumentFormat.XML
    else:
        fmt = detect_format(url)

    metadata = {
        "url": url,
        "content_type": content_type,
        "status_code": response.status_code,
        "size_bytes": len(content_bytes),
    }

    return Document(content=content, metadata=metadata, format=fmt, source=url)


async def load_from_url_async(
    url: str,
    timeout: int = 30,
    max_size_mb: float = 100,
    max_retries: int = 3,
    **kwargs,
) -> Document:
    """Load document from a URL asynchronously.

    Requires: aiohttp package

    Args:
        url (str): URL to fetch document from
        timeout (int): Request timeout in seconds. Defaults to 30.
        max_size_mb (float): Maximum file size in MB. Defaults to 100.
        max_retries (int): Maximum number of retry attempts. Defaults to 3.
        **kwargs: Additional arguments for aiohttp.ClientSession.get()

    Returns:
        Document: Loaded document

    Raises:
        ValueError: If content exceeds max_size_mb
        asyncio.TimeoutError: If request times out
        aiohttp.ClientError: If HTTP error occurs

    Examples:
        >>> import asyncio
        >>> doc = asyncio.run(load_from_url_async("https://example.com/document.pdf"))
        >>> print(doc.content)
    """
    from .extractors import extract_text_from_html

    try:
        import asyncio

        import aiohttp
    except ImportError:
        raise ImportError(
            "Async URL loading requires aiohttp. " "Install with: pip install aiohttp"
        )

    max_size_bytes = max_size_mb * 1024 * 1024

    # Create client session with timeout
    timeout_config = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(timeout=timeout_config) as session:
        # Retry logic
        for attempt in range(max_retries):
            try:
                async with session.get(url, **kwargs) as response:
                    response.raise_for_status()

                    # Check content length
                    content_length = response.headers.get("content-length")
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        if size_mb > max_size_mb:
                            raise ValueError(
                                f"Content size ({size_mb:.2f} MB) exceeds maximum "
                                f"allowed size ({max_size_mb} MB)"
                            )

                    # Download content with size check
                    content_bytes = bytearray()
                    async for chunk in response.content.iter_chunked(8192):
                        content_bytes.extend(chunk)
                        if len(content_bytes) > max_size_bytes:
                            raise ValueError(
                                f"Downloaded content exceeds maximum allowed size "
                                f"({max_size_mb} MB)"
                            )

                    # Decode content
                    encoding = response.charset or "utf-8"
                    content = content_bytes.decode(encoding, errors="replace")

                    # Determine format from URL or content-type
                    content_type = response.headers.get("content-type", "").lower()

                    if "html" in content_type:
                        content = extract_text_from_html(content)
                        fmt = DocumentFormat.HTML
                    elif "json" in content_type:
                        fmt = DocumentFormat.JSON
                    elif "xml" in content_type:
                        fmt = DocumentFormat.XML
                    else:
                        fmt = detect_format(url)

                    metadata = {
                        "url": url,
                        "content_type": content_type,
                        "status_code": response.status,
                        "size_bytes": len(content_bytes),
                    }

                    return Document(
                        content=content, metadata=metadata, format=fmt, source=url
                    )

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff
                await asyncio.sleep(2**attempt)

    # Should not reach here
    raise RuntimeError(f"Failed to load URL after {max_retries} attempts")


def merge_documents(
    documents: List[Document], separator: str = "\n\n---\n\n"
) -> Document:
    """Merge multiple documents into one.

    Args:
        documents (List[Document]): Documents to merge
        separator (str): Separator between documents

    Returns:
        Document: Merged document

    Examples:
        >>> doc1 = Document(content="First doc", metadata={"id": 1})
        >>> doc2 = Document(content="Second doc", metadata={"id": 2})
        >>> merged = merge_documents([doc1, doc2])
        >>> print(merged.content)
    """
    contents = [doc.content for doc in documents]
    merged_content = separator.join(contents)

    merged_metadata = {
        "num_documents": len(documents),
        "sources": [doc.source for doc in documents if doc.source],
        "formats": [doc.format.value for doc in documents],
    }

    return Document(content=merged_content, metadata=merged_metadata)
