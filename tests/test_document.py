"""Tests for the document module."""

import os
import tempfile
from pathlib import Path

import pytest

from kerb import document


class TestDocumentFormat:
    """Test DocumentFormat enum and detection."""

    def test_detect_format_pdf(self):
        assert document.detect_format("file.pdf") == document.DocumentFormat.PDF

    def test_detect_format_docx(self):
        assert document.detect_format("file.docx") == document.DocumentFormat.DOCX

    def test_detect_format_markdown(self):
        assert document.detect_format("file.md") == document.DocumentFormat.MARKDOWN
        assert (
            document.detect_format("file.markdown") == document.DocumentFormat.MARKDOWN
        )

    def test_detect_format_html(self):
        assert document.detect_format("file.html") == document.DocumentFormat.HTML
        assert document.detect_format("file.htm") == document.DocumentFormat.HTML

    def test_detect_format_txt(self):
        assert document.detect_format("file.txt") == document.DocumentFormat.TXT

    def test_detect_format_csv(self):
        assert document.detect_format("file.csv") == document.DocumentFormat.CSV

    def test_detect_format_json(self):
        assert document.detect_format("file.json") == document.DocumentFormat.JSON

    def test_detect_format_xml(self):
        assert document.detect_format("file.xml") == document.DocumentFormat.XML

    def test_detect_format_unknown(self):
        assert document.detect_format("file.xyz") == document.DocumentFormat.UNKNOWN

    def test_is_supported_format(self):
        assert document.is_supported_format("file.pdf") == True
        assert document.is_supported_format("file.txt") == True
        assert document.is_supported_format("file.xyz") == False


class TestTextLoading:
    """Test loading text-based formats."""

    def test_load_text(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello World\nThis is a test.")
            f.flush()

            try:
                doc = document.load_text(f.name)
                assert doc.content == "Hello World\nThis is a test."
                assert doc.metadata["encoding"] == "utf-8"
                assert doc.metadata["lines"] == 2
                assert len(doc) == len("Hello World\nThis is a test.")
            finally:
                os.unlink(f.name)

    def test_load_markdown(self):
        md_content = """---
title: Test
author: John
---

# Hello

This is a **test**."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(md_content)
            f.flush()

            try:
                doc = document.load_markdown(f.name, extract_frontmatter=True)
                assert "# Hello" in doc.content or "Hello" in doc.content
                assert "frontmatter" in doc.metadata
                assert doc.metadata["frontmatter"]["title"] == "Test"
                assert "Hello" in doc.metadata["headings"]
            finally:
                os.unlink(f.name)

    def test_load_json(self):
        json_content = '{"name": "test", "value": 42}'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            f.flush()

            try:
                doc = document.load_json(f.name)
                assert "json_data" in doc.metadata
                assert doc.metadata["json_data"]["name"] == "test"
                assert doc.metadata["json_data"]["value"] == 42
            finally:
                os.unlink(f.name)

    def test_load_csv(self):
        csv_content = """name,age,city
John,30,NYC
Jane,25,LA"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            try:
                doc = document.load_csv(f.name, parse_as_dict=True)
                assert doc.metadata["num_rows"] == 2
                assert doc.metadata["headers"] == ["name", "age", "city"]
                assert doc.metadata["rows"][0]["name"] == "John"
                assert doc.metadata["rows"][1]["name"] == "Jane"
            finally:
                os.unlink(f.name)

    def test_load_xml(self):
        xml_content = """<?xml version="1.0"?>
<root>
  <item>test</item>
</root>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            f.flush()

            try:
                doc = document.load_xml(f.name)
                assert "root" in doc.content
                assert doc.metadata["root_tag"] == "root"
            finally:
                os.unlink(f.name)


class TestHTMLProcessing:
    """Test HTML loading and processing."""

    def test_extract_text_from_html(self):
        html = "<html><body><p>Hello <strong>World</strong></p></body></html>"
        text = document.extract_text_from_html(html)
        assert "Hello" in text
        assert "World" in text
        assert "<" not in text

    def test_extract_text_removes_scripts(self):
        html = '<div>Text<script>alert("x")</script>More</div>'
        text = document.extract_text_from_html(html, remove_scripts=True)
        assert "alert" not in text
        assert "Text" in text
        assert "More" in text

    def test_load_html(self):
        html_content = (
            "<html><head><title>Test</title></head><body><p>Content</p></body></html>"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html_content)
            f.flush()

            try:
                doc = document.load_html(f.name, extract_text=True)
                assert "Content" in doc.content
                assert doc.metadata["title"] == "Test"
                assert "raw_html" in doc.metadata
            finally:
                os.unlink(f.name)


class TestTextCleaning:
    """Test text cleaning functions."""

    def test_clean_text_whitespace(self):
        text = "Hello    World   Test"
        cleaned = document.clean_text(text, normalize_whitespace=True)
        assert cleaned == "Hello World Test"

    def test_clean_text_urls(self):
        text = "Visit https://example.com for info"
        cleaned = document.clean_text(text, remove_urls=True)
        assert "https://example.com" not in cleaned
        assert "Visit" in cleaned

    def test_clean_text_emails(self):
        text = "Contact test@example.com today"
        cleaned = document.clean_text(text, remove_emails=True)
        assert "test@example.com" not in cleaned
        assert "Contact" in cleaned

    def test_clean_text_lowercase(self):
        text = "Hello WORLD"
        cleaned = document.clean_text(text, lowercase=True)
        assert cleaned == "hello world"

    def test_normalize_whitespace_removed(self):
        """Test that normalize_whitespace was removed - use standard library instead."""
        text = "Hello    World\n\n\n\nTest"
        # Use standard library instead
        normalized = " ".join(text.split())
        assert "    " not in normalized
        assert "\n\n\n" not in normalized

    def test_remove_extra_newlines(self):
        text = "Hello\n\n\n\nWorld"
        result = document.remove_extra_newlines(text, max_consecutive=2)
        assert result == "Hello\n\nWorld"

    def test_truncate_text(self):
        """Test truncation - now uses preprocessing module."""
        from kerb.preprocessing import truncate_text

        text = "Hello World" * 100
        truncated = truncate_text(text, max_length=50, suffix="...")
        assert len(truncated) == 50
        assert truncated.endswith("...")


class TestMarkdownProcessing:
    """Test Markdown processing."""

    def test_strip_markdown_headers(self):
        md = "# Title\nContent"
        stripped = document.strip_markdown(md)
        assert "Title" in stripped
        assert "#" not in stripped

    def test_strip_markdown_bold(self):
        md = "This is **bold** text"
        stripped = document.strip_markdown(md)
        assert "bold" in stripped
        assert "**" not in stripped

    def test_strip_markdown_links(self):
        md = "Visit [Google](https://google.com)"
        stripped = document.strip_markdown(md)
        assert "Google" in stripped
        assert "[" not in stripped
        assert "https" not in stripped

    def test_preprocess_markdown(self):
        md = "# Title\n\n**Bold** text"
        processed = document.preprocess_markdown(md, keep_structure=False)
        assert "**" not in processed
        assert "#" not in processed


class TestTextSplitting:
    """Test text splitting functions."""

    def test_split_into_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = document.split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0].strip() == "First sentence."

    def test_split_into_paragraphs(self):
        text = "Para 1\n\nPara 2\n\nPara 3"
        paragraphs = document.split_into_paragraphs(text)
        assert len(paragraphs) == 3
        assert paragraphs[0] == "Para 1"


class TestMetadataExtraction:
    """Test metadata extraction functions."""

    def test_extract_document_stats(self):
        text = "Hello world. This is a test."
        stats = document.extract_document_stats(text)
        assert stats["word_count"] == 6
        assert stats["char_count"] == len(text)
        assert stats["sentence_count"] >= 1

    def test_extract_urls(self):
        text = "Visit https://example.com and www.test.com"
        urls = document.extract_urls(text)
        assert "https://example.com" in urls
        assert "www.test.com" in urls

    def test_extract_emails(self):
        text = "Contact info@example.com or sales@test.org"
        emails = document.extract_emails(text)
        assert "info@example.com" in emails
        assert "sales@test.org" in emails

    def test_extract_dates(self):
        text = "Meeting on 2024-01-15 and 01/20/2024"
        dates = document.extract_dates(text)
        assert "2024-01-15" in dates
        assert "01/20/2024" in dates

    def test_extract_phone_numbers(self):
        text = "Call (555) 123-4567 or 555-987-6543"
        phones = document.extract_phone_numbers(text)
        assert "(555) 123-4567" in phones
        assert "555-987-6543" in phones

    def test_extract_metadata(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            f.flush()

            try:
                metadata = document.extract_metadata(f.name)
                assert "filename" in metadata
                assert "extension" in metadata
                assert "size" in metadata
                assert metadata["extension"] == "txt"
            finally:
                os.unlink(f.name)


class TestFormatSpecificPreprocessing:
    """Test format-specific preprocessing."""

    def test_preprocess_pdf_text(self):
        pdf_text = "This is a sen-\ntence broken."
        cleaned = document.preprocess_pdf_text(pdf_text)
        assert "sentence" in cleaned
        assert "sen-\n" not in cleaned

    def test_preprocess_html_text(self):
        html = "<div>Text <script>bad</script> content</div>"
        cleaned = document.preprocess_html_text(html)
        assert "Text" in cleaned
        assert "script" not in cleaned
        assert "<" not in cleaned


class TestDocumentObject:
    """Test Document dataclass."""

    def test_document_creation(self):
        doc = document.Document(
            content="Test content",
            metadata={"key": "value"},
            format=document.DocumentFormat.TXT,
            source="test.txt",
        )

        assert doc.content == "Test content"
        assert doc.metadata["key"] == "value"
        assert doc.format == document.DocumentFormat.TXT
        assert doc.source == "test.txt"

    def test_document_length(self):
        doc = document.Document(content="Hello World")
        assert len(doc) == 11

    def test_document_to_dict(self):
        doc = document.Document(content="Test", metadata={"key": "value"})

        doc_dict = doc.to_dict()
        assert doc_dict["content"] == "Test"
        assert doc_dict["metadata"] == {"key": "value"}
        assert "format" in doc_dict


class TestBatchOperations:
    """Test batch loading operations."""

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "file1.txt").write_text("Content 1")
            Path(tmpdir, "file2.txt").write_text("Content 2")
            Path(tmpdir, "file3.md").write_text("# Content 3")

            docs = document.load_directory(tmpdir)
            assert len(docs) == 3

            # Test pattern filter
            docs = document.load_directory(tmpdir, pattern="*.txt")
            assert len(docs) == 2

    def test_load_directory_recursive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            Path(tmpdir, "file1.txt").write_text("Content 1")
            subdir = Path(tmpdir, "subdir")
            subdir.mkdir()
            Path(subdir, "file2.txt").write_text("Content 2")

            docs = document.load_directory(tmpdir, recursive=True)
            assert len(docs) == 2

            docs = document.load_directory(tmpdir, recursive=False)
            assert len(docs) == 1

    def test_merge_documents(self):
        doc1 = document.Document(content="First", metadata={"id": 1})
        doc2 = document.Document(content="Second", metadata={"id": 2})
        doc3 = document.Document(content="Third", metadata={"id": 3})

        merged = document.merge_documents([doc1, doc2, doc3], separator=" | ")

        assert "First | Second | Third" == merged.content
        assert merged.metadata["num_documents"] == 3


class TestLoadDocument:
    """Test the main load_document function."""

    def test_load_document_text(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            f.flush()

            try:
                doc = document.load_document(f.name)
                assert doc.content == "Test content"
                assert doc.format == document.DocumentFormat.TXT
                assert doc.source == f.name
            finally:
                os.unlink(f.name)

    def test_load_document_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"test": true}')
            f.flush()

            try:
                doc = document.load_document(f.name)
                assert doc.format == document.DocumentFormat.JSON
                assert doc.metadata["json_data"]["test"] == True
            finally:
                os.unlink(f.name)

    def test_load_document_not_found(self):
        with pytest.raises(FileNotFoundError):
            document.load_document("nonexistent.txt")

    def test_load_document_unsupported(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"content")
            f.flush()

            try:
                with pytest.raises(ValueError, match="Unsupported format"):
                    document.load_document(f.name)
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
