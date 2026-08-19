"""
test_extraction.py
-------------------
Day 2 Task 1: Test PDF Extraction
- Extracts text from a sample PDF
- Verifies and prints the output

Usage:
    python tests/test_extraction.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.pdf_extractor import extract_text_from_pdf

SAMPLE_PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "test_files", "sample.pdf")


def test_pdf_extraction():
    print("=" * 60)
    print("TEST: PDF Text Extraction")
    print("=" * 60)
    print(f"Input file: {SAMPLE_PDF}\n")

    text = extract_text_from_pdf(SAMPLE_PDF)

    assert text, "FAILED: No text was extracted from the PDF"
    assert len(text) > 0, "FAILED: Extracted text is empty"

    print("[PASS] Text extracted successfully")
    print(f"[PASS] Character count: {len(text)}")
    print(f"[PASS] Word count: {len(text.split())}")
    print("\n--- Extracted Text Preview (first 300 chars) ---")
    print(text[:300])
    print("--- End Preview ---\n")

    print("RESULT: PDF EXTRACTION TEST PASSED")



if __name__ == "__main__":
    test_pdf_extraction()
