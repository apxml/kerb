"""Metadata Extraction for Document Analysis

This example demonstrates extracting structured information from documents.

Main concepts:
- Document statistics extraction
- Entity extraction (URLs, emails, dates, phones)
- File metadata
- Information for filtering and routing

Use cases:
- Extracting contact information from documents
- Analyzing document characteristics for chunking
- Building document indices for retrieval
- Filtering documents by content type
- Enriching document metadata for RAG
"""

import tempfile
import os
from datetime import datetime
from kerb.document import (
    load_document,
    extract_metadata,
    extract_document_stats,
    extract_urls,
    extract_emails,
    extract_dates,
    extract_phone_numbers,
    Document,
)


def create_sample_documents(temp_dir: str):
    """Create documents with various metadata for extraction."""
    
    # Document with contact information
    contact_doc = """
    Company Information
    ===================
    
    For general inquiries, please contact us at info@techcorp.com
    or reach our support team at support@techcorp.com.
    
    Sales Department: sales@techcorp.com
    Phone: (555) 123-4567
    Fax: 555-987-6543
    
    Visit our website at https://www.techcorp.com for more information.
    Check our documentation at https://docs.techcorp.com
    
    Office Hours: Monday-Friday, 9:00 AM - 5:00 PM
    Last Updated: 2024-01-15
    """
    
    contact_file = os.path.join(temp_dir, "contact_info.txt")
    with open(contact_file, 'w') as f:
        f.write(contact_doc)
    
    # Research paper with references
    research_doc = """
    Recent Advances in Natural Language Processing
    
    This paper reviews developments from 2023-01-01 to 2024-12-31.
    Key findings were published on 2024-06-15.
    
    References:
    - https://arxiv.org/abs/1234.5678
    - https://papers.nips.cc/paper/9876
    - https://www.aclweb.org/anthology/2024.acl-1.123
    
    For correspondence: researcher@university.edu
    Alternative contact: lab@ai-research.org
    
    Conference dates: 2024-08-20 to 2024-08-23
    """
    
    research_file = os.path.join(temp_dir, "research_paper.txt")
    with open(research_file, 'w') as f:
        f.write(research_doc)
    
    # Long document for statistics
    long_doc = """
    Introduction
    
    This is the first paragraph of a longer document. It contains multiple sentences
    that demonstrate various writing patterns. The content is structured to show
    how document statistics can be useful for LLM applications.
    
    Main Content
    
    The second paragraph introduces the main topic. It provides context and
    background information. This section is longer than the introduction.
    
    Here we have multiple paragraphs that contain detailed information about
    the subject matter. Each paragraph contributes to the overall narrative.
    
    Conclusion
    
    The final section summarizes key points and provides actionable insights.
    Document statistics help determine optimal chunking strategies for embeddings.
    """ * 3  # Repeat to make it longer
    
    long_file = os.path.join(temp_dir, "long_document.txt")
    with open(long_file, 'w') as f:
        f.write(long_doc)
    
    return {
        'contact': contact_file,
        'research': research_file,
        'long': long_file
    }


def main():
    """Run metadata extraction examples."""
    
    print("="*80)
    print("METADATA EXTRACTION FOR DOCUMENT ANALYSIS")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        files = create_sample_documents(temp_dir)
        
        # Example 1: Extract file metadata
        print("\n1. FILE METADATA EXTRACTION")
        print("-" * 80)
        
        file_meta = extract_metadata(files['contact'])
        print(f"Filename: {file_meta['filename']}")
        print(f"Extension: {file_meta['extension']}")
        print(f"Size: {file_meta['size']} bytes")
        print(f"Created: {datetime.fromtimestamp(file_meta['created']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Modified: {datetime.fromtimestamp(file_meta['modified']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Example 2: Extract document statistics
        print("\n2. DOCUMENT STATISTICS")
        print("-" * 80)
        
        doc = load_document(files['long'])
        stats = extract_document_stats(doc.content)
        
        print(f"Character count: {stats['char_count']:,}")
        print(f"Word count: {stats['word_count']:,}")
        print(f"Sentence count: {stats['sentence_count']:,}")
        print(f"Paragraph count: {stats['paragraph_count']:,}")
        print(f"Line count: {stats['line_count']:,}")
        
        # Calculate derived statistics
        avg_words_per_sentence = stats['word_count'] / max(stats['sentence_count'], 1)
        avg_chars_per_word = stats['char_count'] / max(stats['word_count'], 1)
        
        print(f"\nDerived statistics:")
        print(f"Average words per sentence: {avg_words_per_sentence:.1f}")
        print(f"Average characters per word: {avg_chars_per_word:.1f}")
        
        # Use statistics for chunking decisions
        print(f"\nChunking recommendation:")
        if stats['word_count'] < 500:
            print("- Document is small, use as single chunk")
        elif stats['word_count'] < 2000:
            print("- Medium document, split by paragraphs")
        else:
            print("- Large document, use sliding window chunking")
        
        # Example 3: Extract contact information
        print("\n3. CONTACT INFORMATION EXTRACTION")
        print("-" * 80)
        
        contact_doc = load_document(files['contact'])
        
        emails = extract_emails(contact_doc.content)
        phones = extract_phone_numbers(contact_doc.content)
        urls = extract_urls(contact_doc.content)
        dates = extract_dates(contact_doc.content)
        
        print(f"Emails found: {len(emails)}")
        for email in emails:
            print(f"  - {email}")
        
        print(f"\nPhone numbers found: {len(phones)}")
        for phone in phones:
            print(f"  - {phone}")
        
        print(f"\nURLs found: {len(urls)}")
        for url in urls:
            print(f"  - {url}")
        
        print(f"\nDates found: {len(dates)}")
        for date in dates:
            print(f"  - {date}")
        
        # Example 4: Extract from research paper
        print("\n4. RESEARCH PAPER METADATA")
        print("-" * 80)
        
        research_doc = load_document(files['research'])
        
        research_urls = extract_urls(research_doc.content)
        research_emails = extract_emails(research_doc.content)
        research_dates = extract_dates(research_doc.content)
        
        # Categorize URLs
        arxiv_papers = [url for url in research_urls if 'arxiv.org' in url]
        other_refs = [url for url in research_urls if 'arxiv.org' not in url]
        
        print(f"ArXiv papers referenced: {len(arxiv_papers)}")
        for url in arxiv_papers:
            print(f"  - {url}")
        
        print(f"\nOther references: {len(other_refs)}")
        for url in other_refs:
            print(f"  - {url}")
        
        print(f"\nCorrespondence emails: {research_emails}")
        print(f"Important dates: {research_dates}")
        
        # Example 5: Enrich document with extracted metadata
        print("\n5. ENRICHING DOCUMENT METADATA")
        print("-" * 80)
        
        # Create enriched document for RAG system
        enriched_doc = Document(
            content=contact_doc.content,
            metadata={
                "source": files['contact'],
                "extracted_at": datetime.now().isoformat(),
                "statistics": extract_document_stats(contact_doc.content),
                "contacts": {
                    "emails": emails,
                    "phones": phones,
                    "urls": urls,
                },
                "dates": dates,
                "document_type": "contact_information",
                "has_contact_info": len(emails) > 0 or len(phones) > 0,
            }
        )
        
        print("Enriched document metadata keys:")
        for key in enriched_doc.metadata.keys():
            print(f"  - {key}")
        
        print(f"\nDocument type: {enriched_doc.metadata['document_type']}")
        print(f"Has contact info: {enriched_doc.metadata['has_contact_info']}")
        print(f"Total contacts: {len(enriched_doc.metadata['contacts']['emails']) + len(enriched_doc.metadata['contacts']['phones'])}")
        
        # Example 6: Filtering documents by metadata
        print("\n6. DOCUMENT FILTERING STRATEGIES")
        print("-" * 80)
        
        # Simulate multiple documents
        documents = [
            Document(
                content="Research paper about NLP",
                metadata={
                    "type": "research",
                    "word_count": 5000,
                    "has_references": True,
                    "date": "2024-01-15"
                }
            ),
            Document(
                content="Quick blog post",
                metadata={
                    "type": "blog",
                    "word_count": 300,
                    "has_references": False,
                    "date": "2024-02-01"
                }
            ),
            Document(
                content="Technical documentation",
                metadata={
                    "type": "documentation",
                    "word_count": 8000,
                    "has_references": True,
                    "date": "2024-03-10"
                }
            ),
        ]
        
        # Filter by type
        research_docs = [d for d in documents if d.metadata.get('type') == 'research']
        print(f"Research documents: {len(research_docs)}")
        
        # Filter by length (for different processing strategies)
        short_docs = [d for d in documents if d.metadata.get('word_count', 0) < 1000]
        long_docs = [d for d in documents if d.metadata.get('word_count', 0) >= 1000]
        print(f"Short documents: {len(short_docs)}")
        print(f"Long documents: {len(long_docs)}")
        
        # Filter by content features
        reference_docs = [d for d in documents if d.metadata.get('has_references')]
        print(f"Documents with references: {len(reference_docs)}")
        
    print("\n" + "="*80)
    print("KEY TAKEAWAYS FOR LLM DEVELOPERS")
    print("="*80)
    print("- Extract statistics to determine chunking strategy")
    print("- Use entity extraction for document classification")
    print("- Enrich metadata for better retrieval filtering")
    print("- File metadata helps with document tracking")
    print("- Contact extraction useful for CRM/document routing")
    print("- Date extraction helps with temporal filtering")
    print("- Metadata-based filtering improves RAG relevance")


if __name__ == "__main__":
    main()
