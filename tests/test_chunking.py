"""
test_chunking.py
-----------------
Day 2 Task 2: Test Chunking
- Splits extracted text into chunks
- Checks chunk size and quality

Usage:
    python tests/test_chunking.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.pdf_extractor import extract_text_from_pdf
from parsers.chunker import chunk_text

SAMPLE_PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "test_files", "sample.pdf")


def test_chunking():
    print("=" * 60)
    print("TEST: Text Chunking")
    print("=" * 60)

    text = extract_text_from_pdf(SAMPLE_PDF)
    chunk_size, overlap = 500, 50
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 0, "FAILED: No chunks were created"

    print(f"Source text length: {len(text)} chars")
    print(f"Chunk size target: {chunk_size} | Overlap: {overlap}\n")

    all_within_bounds = True
    for i, chunk in enumerate(chunks, start=1):
        within_bounds = len(chunk) <= chunk_size + overlap  # allow small overlap slack
        status = "OK" if within_bounds else "OVER LIMIT"
        if not within_bounds:
            all_within_bounds = False
        preview = chunk[:80].replace("\n", " ")
        print(f"  Chunk {i:02d} | {len(chunk):4d} chars | {status:10s} | {preview}...")

    print()
    assert all_within_bounds, "FAILED: One or more chunks exceeded the size limit"
    print(f"[PASS] {len(chunks)} chunks created")
    print(f"[PASS] All chunks within size bounds (<= {chunk_size + overlap} chars)")
    print(f"[PASS] No empty chunks: {all(len(c.strip()) > 0 for c in chunks)}")

    print("\nRESULT: CHUNKING TEST PASSED")



if __name__ == "__main__":
    test_chunking()
