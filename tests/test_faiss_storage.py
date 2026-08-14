"""
test_faiss_storage.py
---------------------
Test FAISS vector storage.

This test verifies:
1. PDF text extraction
2. Text chunking
3. Embedding generation
4. FAISS vector storage
5. Saving the FAISS index
6. Reloading the FAISS index
7. Similarity search
"""

import sys
import os

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, BASE_DIR)


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from parsers.pdf_extractor import extract_text_from_pdf
from parsers.chunker import chunk_text
from embeddings.embedder import embed_chunks
from vector_db.faiss_store import FaissStore


# ---------------------------------------------------------
# Test files
# ---------------------------------------------------------

SAMPLE_PDF = os.path.join(
    BASE_DIR,
    "test_files",
    "sample.pdf"
)

INDEX_PREFIX = os.path.join(
    BASE_DIR,
    "vector_db",
    "test_index"
)


# ---------------------------------------------------------
# FAISS Storage Test
# ---------------------------------------------------------

def test_faiss_storage():

    print("=" * 60)
    print("TEST: FAISS VECTOR STORAGE")
    print("=" * 60)

    # -----------------------------------------------------
    # STEP 1: Extract text from PDF
    # -----------------------------------------------------

    text = extract_text_from_pdf(SAMPLE_PDF)

    assert text, (
        "FAILED: PDF text extraction returned empty text"
    )

    print("[PASS] PDF text extracted successfully")


    # -----------------------------------------------------
    # STEP 2: Create text chunks
    # -----------------------------------------------------

    chunks = chunk_text(
        text,
        chunk_size=500,
        overlap=50
    )

    assert len(chunks) > 0, (
        "FAILED: No text chunks were created"
    )

    print(
        f"[PASS] Created {len(chunks)} text chunk(s)"
    )


    # -----------------------------------------------------
    # STEP 3: Generate embeddings
    # -----------------------------------------------------

    vectors = embed_chunks(chunks)

    assert vectors is not None, (
        "FAILED: Embeddings were not generated"
    )

    assert len(vectors) == len(chunks), (
        "FAILED: Number of vectors does not match "
        "number of chunks"
    )

    print(
        f"[PASS] Generated embeddings "
        f"shape={vectors.shape}"
    )


    # -----------------------------------------------------
    # STEP 4: Create FAISS store
    # -----------------------------------------------------

    store = FaissStore(
        dim=vectors.shape[1]
    )

    store.add(
        vectors,
        chunks,
        source="sample.pdf"
    )

    assert store.index.ntotal == len(chunks), (
        "FAILED: Not all vectors were added "
        "to the FAISS index"
    )

    print(
        f"[PASS] Added {store.index.ntotal} "
        f"vectors to FAISS index"
    )


    # -----------------------------------------------------
    # STEP 5: Save FAISS index
    # -----------------------------------------------------

    store.save(INDEX_PREFIX)

    index_file = f"{INDEX_PREFIX}.index"
    metadata_file = f"{INDEX_PREFIX}.meta.json"

    assert os.path.exists(index_file), (
        "FAILED: FAISS index file was not saved"
    )

    assert os.path.exists(metadata_file), (
        "FAILED: Metadata file was not saved"
    )

    print("[PASS] FAISS index saved successfully")
    print(f"       Index: {index_file}")
    print(f"       Metadata: {metadata_file}")


    # -----------------------------------------------------
    # STEP 6: Reload FAISS index
    # -----------------------------------------------------

    reloaded_store = FaissStore.load(
        INDEX_PREFIX
    )

    assert (
        reloaded_store.index.ntotal
        == len(chunks)
    ), (
        "FAILED: Reloaded FAISS index has "
        "incorrect vector count"
    )

    print(
        f"[PASS] Reloaded FAISS index "
        f"with {reloaded_store.index.ntotal} vectors"
    )


    # -----------------------------------------------------
    # STEP 7: Similarity search
    # -----------------------------------------------------

    query_vector = vectors[0]

    results = reloaded_store.search(
        query_vector,
        top_k=min(2, len(chunks))
    )

    assert len(results) > 0, (
        "FAILED: Similarity search returned no results"
    )

    print(
        f"[PASS] Similarity search returned "
        f"{len(results)} result(s)"
    )


    # -----------------------------------------------------
    # STEP 8: Display search results
    # -----------------------------------------------------

    print()
    print("--- Search Results (query = chunk 1) ---")

    for i, result in enumerate(
        results,
        start=1
    ):

        preview = result["text"][:80].replace(
            "\n",
            " "
        )

        print(
            f"  #{i} "
            f"score={result['score']:.4f} "
            f"source={result['source']} "
            f"text={preview}"
        )


    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print("RESULT: FAISS STORAGE TEST PASSED")

    # IMPORTANT:
    # Do NOT use return here.
    # Pytest test functions must return None.


# ---------------------------------------------------------
# Run directly
# ---------------------------------------------------------

if __name__ == "__main__":
    test_faiss_storage()