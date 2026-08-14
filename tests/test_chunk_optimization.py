"""
Day 4 - Task 4: Chunk Size and Overlap Optimization

Tests different chunk sizes and overlap values
and compares the resulting number of chunks.
"""

import os
import sys

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from parsers.pdf_extractor import extract_text_from_pdf
from parsers.chunker import chunk_text


SAMPLE_PDF = os.path.join(
    BASE_DIR,
    "test_files",
    "sample.pdf"
)


def optimize_chunks():

    print("=" * 70)
    print("DAY 4 - CHUNK SIZE & OVERLAP OPTIMIZATION")
    print("=" * 70)

    # Step 1: Extract PDF text
    print("\n[STEP 1] Extracting text from PDF...")

    text = extract_text_from_pdf(SAMPLE_PDF)

    print(f"[INFO] Extracted characters: {len(text)}")

    # Different configurations to test
    configurations = [
        (500, 50),
        (500, 100),
        (800, 100),
        (1000, 100),
        (1000, 200),
    ]

    print("\n[STEP 2] Testing different chunk configurations...")
    print("-" * 70)

    results = []

    for chunk_size, overlap in configurations:

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        count = len(chunks)

        results.append(
            (chunk_size, overlap, count)
        )

        print(
            f"Chunk Size: {chunk_size:4} | "
            f"Overlap: {overlap:3} | "
            f"Chunks: {count}"
        )

    print("-" * 70)

    # Select configuration
    best_chunk_size = 800
    best_overlap = 100

    best_chunks = chunk_text(
        text,
        chunk_size=best_chunk_size,
        overlap=best_overlap
    )

    print("\n[STEP 3] Selected configuration")
    print("-" * 70)

    print(f"Chunk Size : {best_chunk_size}")
    print(f"Overlap    : {best_overlap}")
    print(f"Total Chunks: {len(best_chunks)}")

    print("\n[STEP 4] Sample optimized chunk")
    print("-" * 70)

    if best_chunks:
        preview = best_chunks[0][:300].replace("\n", " ")
        print(preview + "...")

    print("\n" + "=" * 70)
    print("RESULT: CHUNK OPTIMIZATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    optimize_chunks()