"""
test_embeddings.py
-------------------
Day 2 Task 3: Test Embeddings
- Generates embeddings using sentence-transformers
- Verifies vectors are created correctly

Usage:
    python tests/test_embeddings.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from parsers.pdf_extractor import extract_text_from_pdf
from parsers.chunker import chunk_text
from embeddings.embedder import embed_chunks

SAMPLE_PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "test_files", "sample.pdf")


def test_embeddings():
    print("=" * 60)
    print("TEST: Embedding Generation")
    print("=" * 60)

    text = extract_text_from_pdf(SAMPLE_PDF)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"Chunks to embed: {len(chunks)}\n")

    vectors = embed_chunks(chunks)

    assert vectors.shape[0] == len(chunks), "FAILED: Vector count doesn't match chunk count"
    assert vectors.dtype == np.float32, "FAILED: Vectors should be float32"
    assert not np.isnan(vectors).any(), "FAILED: Vectors contain NaN values"

    print(f"[PASS] Vector count matches chunk count: {vectors.shape[0]}")
    print(f"[PASS] Embedding dimension: {vectors.shape[1]}")
    print(f"[PASS] Data type: {vectors.dtype}")
    print(f"[PASS] No NaN values present")

    print("\n--- Sample vector (first 8 dims of chunk 1) ---")
    print(vectors[0][:8])

    # Sanity check: identical chunk text should give identical vector
    repeat_vector = embed_chunks([chunks[0]])
    identical = np.allclose(vectors[0], repeat_vector[0])
    print(f"\n[PASS] Deterministic check (same input -> same vector): {identical}")
    assert identical, "FAILED: Same input produced different vectors"

    print("\nRESULT: EMBEDDINGS TEST PASSED")
    return vectors, chunks


if __name__ == "__main__":
    test_embeddings()
