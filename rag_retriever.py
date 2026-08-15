"""
rag_retriever.py
----------------
RAG retrieval pipeline for the AI Module.

Flow:
User Query
    -> Query Embedding
    -> FAISS Similarity Search
    -> Relevant Chunks
    -> Context

Uses:
- all-MiniLM-L6-v2 embeddings
- FAISS vector store
"""

from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from vector_db.faiss_store import FaissStore


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_PREFIX = "vector_db/index_data"

DEFAULT_TOP_K = 3

# Minimum similarity score required for a chunk
# to be considered relevant.
DEFAULT_MIN_SCORE = 0.30


# ---------------------------------------------------------
# RAG Retriever
# ---------------------------------------------------------

class RAGRetriever:
    """
    Converts a user query into an embedding and retrieves
    the most relevant chunks from the FAISS index.
    """

    def __init__(
        self,
        index_prefix: str = INDEX_PREFIX,
        model_name: str = MODEL_NAME,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        print(f"[INFO] Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print(f"[INFO] Loading FAISS index: {index_prefix}")

        self.store = FaissStore.load(index_prefix)

        self.min_score = min_score

        print(
            f"[INFO] RAG retriever ready "
            f"(vectors={self.store.index.ntotal}, "
            f"dimension={self.store.dim})"
        )

    # -----------------------------------------------------
    # Query Embedding
    # -----------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert a user query into a normalized embedding vector.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        vector = np.asarray(vector, dtype="float32")

        print(f"[INFO] Query embedding shape: {vector.shape}")

        return vector

    # -----------------------------------------------------
    # Similarity Search
    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a user query.
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        print(f"[INFO] Query: {query}")

        # Step 1: Query embedding
        query_vector = self.embed_query(query)

        # Step 2: FAISS search
        results = self.store.search(
            query_vector,
            top_k=top_k,
        )

        print(f"[INFO] FAISS returned {len(results)} result(s)")

        # Step 3: Apply relevance threshold
        relevant_results = [
            result
            for result in results
            if float(result.get("score", 0.0)) >= self.min_score
        ]

        print(
            f"[INFO] Relevant results after threshold: "
            f"{len(relevant_results)}"
        )

        return relevant_results

    # -----------------------------------------------------
    # Context Builder
    # -----------------------------------------------------

    def build_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """
        Retrieve chunks and combine them into LLM-ready context.
        """

        results = self.retrieve(
            query=query,
            top_k=top_k,
        )

        # No relevant context
        if not results:
            return {
                "query": query,
                "found": False,
                "chunks": [],
                "context": "",
                "message": (
                    "No relevant context was found "
                    "for this query."
                ),
            }

        context_parts = []

        for index, result in enumerate(results, start=1):
            text = result.get("text", "").strip()
            source = result.get("source", "unknown")
            score = float(result.get("score", 0.0))

            context_parts.append(
                f"[Chunk {index} | "
                f"source={source} | "
                f"score={score:.4f}]\n"
                f"{text}"
            )

        context = "\n\n".join(context_parts)

        return {
            "query": query,
            "found": True,
            "chunks": results,
            "context": context,
            "message": "Relevant context retrieved successfully.",
        }


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------

def main():
    """
    Manual test for the complete retrieval pipeline.
    """

    print("=" * 70)
    print("RAG RETRIEVAL PIPELINE TEST")
    print("=" * 70)

    retriever = RAGRetriever()

    queries = [
        "What is this document about?",
        "What information is contained in the PDF?",
        "Tell me about the sample document.",
    ]

    for number, query in enumerate(queries, start=1):

        print("\n" + "-" * 70)
        print(f"QUERY {number}: {query}")
        print("-" * 70)

        result = retriever.build_context(
            query=query,
            top_k=3,
        )

        if not result["found"]:
            print("[NO CONTEXT]")
            print(result["message"])
            continue

        print(f"[INFO] Retrieved {len(result['chunks'])} chunk(s)")

        print("\n--- RETRIEVED CONTEXT ---")

        print(result["context"])

    print("\n" + "=" * 70)
    print("RESULT: RAG RETRIEVAL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()