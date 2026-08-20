"""
test_chunking.py
-----------------
Comprehensive Test Suite for AI Module Document Chunking:
1. Standard text chunking with size and overlap constraints.
2. Sentence boundary preservation and abbreviation protection (Dr., e.g., i.e., vs., Fig., 3.14).
3. Word boundary integrity (no chopped words at chunk start/end or overlap boundaries).
4. Edge cases: empty text, whitespace-only text, None input, very short text.
5. Oversized single sentences / long unbroken text handling.
6. Pre-embedding chunk verification (verify_chunks, validate_and_filter_chunks).
7. Page-aware chunking (chunk_pages) across multi-page structures.
8. Multi-PDF document ingestion and chunk verification across diverse PDF samples.
9. Invalid argument validation (ValueError handling).
"""

import os
import sys
import pytest

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from parsers.pdf_extractor import extract_text_from_pdf, extract_pages_from_pdf
from parsers.chunker import (
    chunk_text,
    chunk_pages,
    verify_chunks,
    validate_and_filter_chunks,
    _split_into_sentences,
)

SAMPLE_PDF = os.path.join(BASE_DIR, "test_files", "sample.pdf")
AI_ARCH_PDF = os.path.join(BASE_DIR, "test_files", "ai_architecture.pdf")
COMPLEX_PDF = os.path.join(BASE_DIR, "test_files", "complex_formatting.pdf")
MANUAL_PDF = os.path.join(BASE_DIR, "test_files", "multipage_manual.pdf")
SHORT_PDF = os.path.join(BASE_DIR, "test_files", "short_document.pdf")
EMPTY_PDF = os.path.join(BASE_DIR, "test_files", "empty_or_scanned.pdf")


def test_basic_chunking():
    """Verify basic chunking on sample document produces bounded, non-empty chunks."""
    text = extract_text_from_pdf(SAMPLE_PDF)
    assert len(text) > 0, "Failed to extract text from sample.pdf"

    chunk_size, overlap = 500, 50
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 0, "No chunks were created"
    for i, chunk in enumerate(chunks):
        assert len(chunk.strip()) > 0, f"Chunk #{i} is empty"
        # Each chunk should reasonably respect chunk size + overlap slack
        assert len(chunk) <= int((chunk_size + overlap) * 1.3), f"Chunk #{i} exceeded bounds: {len(chunk)} chars"

    report = verify_chunks(chunks, max_chars=chunk_size)
    assert report["is_valid"] is True
    assert report["empty_chunks"] == 0
    assert report["total_chunks"] == len(chunks)


def test_sentence_boundaries_and_abbreviations():
    """Verify sentence tokenization protects abbreviations, decimals, URLs, and quotes."""
    text = (
        "Dr. Evelyn Vance and Prof. Cole published results in Vol. 4, No. 2 of the AI Journal. "
        "Their system (e.g., FAISS vs. Qdrant) achieved a 99.85% precision rate with $14.50 cost. "
        "Visit https://example.com/docs for details. Vance et al. noted: 'Boundary integrity is vital.' "
        "The subsequent phase will expand testing across all regions."
    )

    sentences = _split_into_sentences(text)

    # Verify abbreviations were not split
    assert not any(s == "Dr." for s in sentences), "Split on 'Dr.'"
    assert not any(s.startswith("Cole") for s in sentences), "'Dr. Evelyn Vance and Prof. Cole' split inappropriately"
    assert not any(s == "e.g." for s in sentences), "Split on 'e.g.'"
    assert not any(s == "vs." for s in sentences), "Split on 'vs.'"
    assert not any(s == "Vol." or s == "No." for s in sentences), "Split on journal volume/number abbreviations"
    assert not any(s == "al." for s in sentences), "Split on 'et al.'"

    # Test chunking on this text
    chunks = chunk_text(text, chunk_size=180, overlap=30)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) > 0


def test_word_boundary_preservation():
    """Verify that overlap and chunk splits do not chop words in half."""
    text = (
        "Microservices architecture enables high availability and fault tolerance in enterprise systems. "
        "Each service handles a distinct business capability such as document ingestion, embedding indexing, "
        "semantic similarity search, and answer generation using large language models."
    )

    chunks = chunk_text(text, chunk_size=120, overlap=30)
    assert len(chunks) > 1

    for i, chunk in enumerate(chunks):
        # Chunk should start with an alphanumeric or valid punctuation/bracket, not a chopped word fragment
        words = chunk.split()
        assert len(words) > 0, f"Chunk #{i} has no words"
        # First word should be a complete word from text
        first_word = words[0].strip(".,!?:;\"'()")
        assert first_word in text, f"Chunk #{i} first word '{first_word}' appears truncated"
        # Last word should be a complete word from text
        last_word = words[-1].strip(".,!?:;\"'()")
        assert last_word in text, f"Chunk #{i} last word '{last_word}' appears truncated"


def test_empty_and_short_text_handling():
    """Verify empty, whitespace-only, None, and very short texts are handled gracefully."""
    # Empty string
    assert chunk_text("") == []

    # Whitespace-only string
    assert chunk_text("   \n\t   ") == []

    # None input
    assert chunk_text(None) == []

    # Very short text (less than chunk_size)
    short_text = "System is operational."
    res = chunk_text(short_text, chunk_size=500, overlap=50)
    assert res == ["System is operational."]

    # Single word
    single_word = "Initialization"
    res_word = chunk_text(single_word, chunk_size=500, overlap=50)
    assert res_word == ["Initialization"]


def test_oversized_sentence_handling():
    """Verify sentences exceeding chunk_size are safely split on word boundaries."""
    # Create a 1500-character sentence without periods
    long_sentence = " ".join([f"token_{i}" for i in range(250)])
    chunk_size = 300
    overlap = 50

    chunks = chunk_text(long_sentence, chunk_size=chunk_size, overlap=overlap)
    assert len(chunks) > 1

    for i, chunk in enumerate(chunks):
        assert len(chunk) <= int(chunk_size * 1.3)
        # Verify tokens aren't chopped (e.g. token_12 -> tok)
        for token in chunk.split():
            assert token.startswith("token_")


def test_chunk_verification_and_filtering():
    """Test verify_chunks and validate_and_filter_chunks utilities."""
    valid_chunks = [
        "First document chunk containing valid text content.",
        "Second document chunk with additional context.",
        "Third document chunk completing the paragraph.",
    ]

    report = verify_chunks(valid_chunks, min_chars=10, max_chars=100)
    assert report["is_valid"] is True
    assert report["total_chunks"] == 3
    assert report["empty_chunks"] == 0
    assert report["min_chunk_length"] > 0
    assert report["avg_chunk_length"] > 0
    assert len(report["warnings"]) == 0

    # Chunks with invalid empty elements
    mixed_chunks = ["Valid chunk.", "", "   ", "Another valid chunk."]
    invalid_report = verify_chunks(mixed_chunks)
    assert invalid_report["is_valid"] is False
    assert invalid_report["empty_chunks"] == 2

    # Filtering should clean it up
    cleaned = validate_and_filter_chunks(mixed_chunks, min_chars=5)
    assert len(cleaned) == 2
    cleaned_report = verify_chunks(cleaned)
    assert cleaned_report["is_valid"] is True


def test_chunk_pages_multipage():
    """Verify chunk_pages preserves page numbering and chunk indexing."""
    pages = [
        {"page_number": 1, "text": "Page one text. Introduces the core concepts of the AI module."},
        {"page_number": 2, "text": "Page two text. Discusses vector indexing and FAISS storage."},
        {"page_number": 3, "text": "   "},  # Empty page
        {"page_number": 4, "text": "Page four text. Concludes with retrieval evaluation metrics."},
    ]

    records = chunk_pages(pages, chunk_size=200, overlap=20)
    assert len(records) >= 3

    page_nums = [r["page_number"] for r in records]
    assert 1 in page_nums
    assert 2 in page_nums
    assert 3 not in page_nums  # Empty page excluded
    assert 4 in page_nums

    # Verify chunk_index is sequential
    indices = [r["chunk_index"] for r in records]
    assert indices == list(range(len(records)))

    # Verify metadata fields
    for r in records:
        assert "char_count" in r
        assert "word_count" in r
        assert r["char_count"] == len(r["text"])


def test_chunking_with_multiple_pdf_documents():
    """Test chunking with all sample PDFs in test_files."""
    pdf_files = [
        SAMPLE_PDF,
        AI_ARCH_PDF,
        COMPLEX_PDF,
        MANUAL_PDF,
        SHORT_PDF,
    ]

    for pdf_path in pdf_files:
        assert os.path.exists(pdf_path), f"Test PDF not found: {pdf_path}"
        pages = extract_pages_from_pdf(pdf_path)
        assert len(pages) > 0, f"No pages extracted from {pdf_path}"

        records = chunk_pages(pages, chunk_size=500, overlap=50)
        assert len(records) > 0, f"No chunks created from {pdf_path}"

        chunks = [r["text"] for r in records]
        report = verify_chunks(chunks, max_chars=500)
        assert report["is_valid"] is True, f"Chunk verification failed for {pdf_path}: {report['warnings']}"
        assert report["empty_chunks"] == 0

    # Test empty or scanned PDF handling
    if os.path.exists(EMPTY_PDF):
        empty_text = extract_text_from_pdf(EMPTY_PDF)
        empty_chunks = chunk_text(empty_text)
        assert empty_chunks == [], "Empty PDF should result in empty chunk list"


def test_invalid_arguments():
    """Verify ValueError is raised on invalid chunk_size or overlap."""
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        chunk_text("Sample text", chunk_size=0, overlap=0)

    with pytest.raises(ValueError, match="overlap must be >= 0 and less than chunk_size"):
        chunk_text("Sample text", chunk_size=500, overlap=500)

    with pytest.raises(ValueError, match="overlap must be >= 0 and less than chunk_size"):
        chunk_text("Sample text", chunk_size=500, overlap=-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
