"""
test_faiss_storage.py
----------------------
Day 2 Task 4: Test FAISS Storage
- Stores embeddings in a FAISS index
- Verifies vectors are saved successfully (via save -> reload -> search)

Usage:
    python tests/test_faiss_storage.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.pdf_extractor import extract_text_from_pdf
from parsers.chunker import chunk_text
from embeddings.embedder import embed_chunks
from vector_db.faiss_store import FaissStore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_PDF = os.path.join(BASE_DIR, "test_files", "sample.pdf")
INDEX_PREFIX = os.path.join(BASE_DIR, "vector_db", "test_index")


def test_faiss_storage():
    print("=" * 60)
    print("TEST: FAISS Vector Storage")
    print("=" * 60)

    text = extract_text_from_pdf(SAMPLE_PDF)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    vectors = embed_chunks(chunks)
    print(f"Chunks: {len(chunks)} | Vector dim: {vectors.shape[1]}\n")

    # 1. Create store and add vectors
    store = FaissStore(dim=vectors.shape[1])
    store.add(vectors, chunks, source="sample.pdf")
    assert store.index.ntotal == len(chunks), "FAILED: Not all vectors were added to the index"
    print(f"[PASS] Added {store.index.ntotal} vectors to FAISS index")

    # 2. Save to disk
    store.save(INDEX_PREFIX)
    assert os.path.exists(f"{INDEX_PREFIX}.index"), "FAILED: Index file was not saved"
    assert os.path.exists(f"{INDEX_PREFIX}.meta.json"), "FAILED: Metadata file was not saved"
    print(f"[PASS] Index persisted to disk: {INDEX_PREFIX}.index / .meta.json")

    # 3. Reload from disk into a fresh object (proves it's really saved, not just in memory)
    reloaded_store = FaissStore.load(INDEX_PREFIX)
    assert reloaded_store.index.ntotal == len(chunks), "FAILED: Reloaded index has wrong vector count"
    print(f"[PASS] Reloaded index from disk: {reloaded_store.index.ntotal} vectors confirmed")

    # 4. Run a search against the reloaded index
    query_vector = vectors[0]
    results = reloaded_store.search(query_vector, top_k=min(2, len(chunks)))
    assert len(results) > 0, "FAILED: Search returned no results"
    print(f"[PASS] Search on reloaded index returned {len(results)} result(s)\n")

    print("--- Search Results (query = chunk 1) ---")
    for i, r in enumerate(results, start=1):
        preview = r["text"][:80].replace("\n", " ")
        print(f"  #{i} score={r['score']:.4f} source={r['source']} text={preview}...")

    print("\nRESULT: FAISS STORAGE TEST PASSED")
    return reloaded_store


if __name__ == "__main__":
    test_faiss_storage()
