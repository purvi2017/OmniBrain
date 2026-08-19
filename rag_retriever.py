"""
rag_retriever.py
----------------
RAG retrieval pipeline for the AI Module (Day 8 Enhanced).

Flow:
User Query
    -> Query Embedding (all-MiniLM-L6-v2)
    -> FAISS Similarity Search (Cosine / Inner Product)
    -> Adaptive Threshold & Delta Filtering (min_score & max_score_drop)
    -> Deduplication & Confidence Scoring
    -> Structured Context Assembly
"""

from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from vector_db.faiss_store import FaissStore


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PREFIX = "vector_db/index_data"
DEFAULT_TOP_K = 3

# Calibrated similarity thresholds for all-MiniLM-L6-v2:
DEFAULT_MIN_SCORE = 0.35
DEFAULT_MAX_SCORE_DROP = 0.25  # Relative drop limit from top candidate


def get_confidence_label(score: float) -> str:
    """Return a human-readable confidence label based on similarity score."""
    if score >= 0.55:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------
# RAG Retriever
# ---------------------------------------------------------

class RAGRetriever:
    """
    Converts user query into embedding, retrieves matching chunks
    from FAISS, applies adaptive filtering and formats context.
    """

    def __init__(
        self,
        index_prefix: str = INDEX_PREFIX,
        model_name: str = MODEL_NAME,
        min_score: float = DEFAULT_MIN_SCORE,
        max_score_drop: float = DEFAULT_MAX_SCORE_DROP,
    ):
        print(f"[INFO] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        print(f"[INFO] Loading FAISS index: {index_prefix}")
        self.store = FaissStore.load(index_prefix)
        self.min_score = min_score
        self.max_score_drop = max_score_drop

        print(
            f"[INFO] RAG retriever ready "
            f"(vectors={self.store.index.ntotal}, "
            f"dimension={self.store.dim})"
        )

    # -----------------------------------------------------
    # Query Embedding
    # -----------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Convert a user query into a normalized embedding vector."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vector = np.asarray(vector, dtype="float32")
        return vector

    # -----------------------------------------------------
    # Similarity Search & Adaptive Filtering
    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        document_id: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score_drop: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and filter relevant chunks for a user query.
        """
        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        effective_min_score = min_score if min_score is not None else self.min_score
        effective_score_drop = max_score_drop if max_score_drop is not None else self.max_score_drop

        # Step 1: Embed query
        query_vector = self.embed_query(query)

        # Step 2: FAISS search
        raw_results = self.store.search(
            query_vector,
            top_k=top_k,
            document_id=document_id,
        )

        if not raw_results:
            return []

        # Step 3: Absolute threshold filtering
        filtered_results = [
            r for r in raw_results if float(r.get("score", 0.0)) >= effective_min_score
        ]

        if not filtered_results:
            return []

        # Step 4: Adaptive relative score drop filtering
        # Avoid pulling low-relevance noise chunks if top match is strong
        top_score = float(filtered_results[0].get("score", 0.0))
        adaptive_results = [
            r
            for r in filtered_results
            if (top_score - float(r.get("score", 0.0))) <= effective_score_drop
        ]

        # Annotate with confidence labels
        for r in adaptive_results:
            r["confidence"] = get_confidence_label(float(r.get("score", 0.0)))

        return adaptive_results

    # -----------------------------------------------------
    # Context Builder
    # -----------------------------------------------------

    def build_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        document_id: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve chunks and combine them into LLM-ready context.
        """
        results = self.retrieve(
            query=query,
            top_k=top_k,
            document_id=document_id,
            min_score=min_score,
        )

        if not results:
            return {
                "query": query,
                "found": False,
                "chunks": [],
                "context": "",
                "message": (
                    "No relevant context was found in the indexed documents "
                    "matching the relevance threshold."
                ),
            }

        context_parts = []
        for index, result in enumerate(results, start=1):
            text = result.get("text", "").strip()
            source = result.get("source") or result.get("filename") or "unknown"
            doc_id = result.get("document_id", "unknown")
            page = result.get("page", 1)
            score = float(result.get("score", 0.0))
            confidence = result.get("confidence", "MEDIUM")

            context_parts.append(
                f"[Chunk {index} | Document: {source} (ID: {doc_id}) | Page: {page} | "
                f"Score: {score:.4f} ({confidence} Confidence)]\n"
                f"{text}"
            )

        context = "\n\n".join(context_parts)

        return {
            "query": query,
            "found": True,
            "chunks": results,
            "context": context,
            "message": f"Successfully retrieved {len(results)} relevant chunk(s).",
        }


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------

def main():
    print("=" * 70)
    print("DAY 8 - ENHANCED RAG RETRIEVAL PIPELINE TEST")
    print("=" * 70)

    retriever = RAGRetriever()

    queries = [
        "What is this document about?",
        "What information is contained in the PDF?",
        "Tell me about the sample document.",
        "What is the atomic structure of an uncharged neutron?",  # Out of domain
    ]

    for number, query in enumerate(queries, start=1):
        print("\n" + "-" * 70)
        print(f"QUERY {number}: {query}")
        print("-" * 70)

        result = retriever.build_context(query=query, top_k=3)

        if not result["found"]:
            print("[NO RELEVANT CONTEXT]")
            print(result["message"])
            continue

        print(f"[INFO] Retrieved {len(result['chunks'])} chunk(s)")
        print("\n--- RETRIEVED CONTEXT ---")
        print(result["context"])

    print("\n" + "=" * 70)
    print("RESULT: RAG RETRIEVAL PIPELINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()