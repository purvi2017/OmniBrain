"""
test_similarity_search.py
-------------------------
Day 4 Task:
- Load the existing FAISS index
- Generate embeddings for different queries
- Perform similarity search
- Display top matching chunks and scores
"""

import os
import sys

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from embeddings.embedder import embed_chunks
from vector_db.faiss_store import FaissStore


# Existing FAISS index created by your pipeline
INDEX_PREFIX = os.path.join(BASE_DIR, "vector_db", "index_data")


def run_similarity_search():
    print("=" * 70)
    print("DAY 4 - FAISS SIMILARITY SEARCH")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1: Load existing FAISS index
    # ---------------------------------------------------------
    print("\n[STEP 1] Loading FAISS index...")

    store = FaissStore.load(INDEX_PREFIX)

    print(f"[INFO] FAISS index loaded successfully")
    print(f"[INFO] Total vectors: {store.index.ntotal}")

    if store.index.ntotal == 0:
        print("[ERROR] FAISS index is empty.")
        return

    # ---------------------------------------------------------
    # STEP 2: Define different test queries
    # ---------------------------------------------------------
    queries = [
        "What is this document about?",
        "What information is contained in the PDF?",
        "Tell me about the sample document."
    ]

    # ---------------------------------------------------------
    # STEP 3: Run similarity search
    # ---------------------------------------------------------
    for query_number, query in enumerate(queries, start=1):

        print("\n" + "-" * 70)
        print(f"QUERY {query_number}: {query}")
        print("-" * 70)

        # Generate embedding for the query
        query_embedding = embed_chunks([query])

        print(f"[INFO] Query embedding shape: {query_embedding.shape}")

        # Search FAISS index
        results = store.search(
            query_embedding,
            top_k=min(3, store.index.ntotal)
        )

        # -----------------------------------------------------
        # STEP 4: Display results
        # -----------------------------------------------------
        if not results:
            print("[WARNING] No results found.")
            continue

        print(f"[INFO] Retrieved {len(results)} result(s)\n")

        for rank, result in enumerate(results, start=1):

            text = result.get("text", "")
            source = result.get("source", "")
            score = result.get("score", 0.0)

            # Keep terminal output readable
            preview = text[:150].replace("\n", " ")

            print(
                f"#{rank} "
                f"score={score:.4f} "
                f"source={source}"
            )

            print(f"   text={preview}...")

    print("\n" + "=" * 70)
    print("RESULT: SIMILARITY SEARCH COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_similarity_search()