"""
faiss_store.py
--------------
Local vector storage and similarity search using FAISS.

FAISS itself only stores vectors + lets you search by index position —
it has no concept of "metadata". So alongside the FAISS index we keep
a parallel Python list (`self.metadata`) mapping index position ->
{"text": chunk_text, "source": pdf_filename, ...}, and persist both
together (index + metadata) to disk.

Usage:
    from vector_db.faiss_store import FaissStore

    store = FaissStore(dim=384)
    store.add(vectors, chunks, source="sample.pdf")
    store.save("vector_db/index_data")

    store2 = FaissStore.load("vector_db/index_data")
    results = store2.search(query_vector, top_k=3)
"""

import os
import json
from typing import List, Dict, Any, Optional

import numpy as np
import faiss


class FaissStore:
    def __init__(self, dim: int):
        """
        Args:
            dim: Dimensionality of the embedding vectors (e.g. 384 for
                 all-MiniLM-L6-v2).
        """
        self.dim = dim
        # IndexFlatIP = exact search using inner product (cosine sim if
        # vectors are normalized). Simple and accurate for small/medium
        # datasets — fine for Day 1/2 scale.
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, chunks: List[str], source: str = "unknown") -> None:
        """
        Add vectors + their originating text chunks to the store.

        Args:
            vectors: numpy array of shape (n, dim).
            chunks: list of the text strings each vector represents
                    (must be same length as vectors).
            source: identifier for where these chunks came from
                    (e.g. the PDF filename).
        """
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        if vectors.size == 0:
            return

        # Normalize so inner product = cosine similarity
        normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.index.add(normalized.astype("float32"))

        for chunk in chunks:
            self.metadata.append({"text": chunk, "source": source})

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the top_k most similar chunks to the query vector.

        Args:
            query_vector: numpy array of shape (dim,) or (1, dim).
            top_k: number of results to return.

        Returns:
            List of dicts: {"text": ..., "source": ..., "score": ...}
            sorted by descending similarity score.
        """
        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        query_vector = query_vector / np.linalg.norm(query_vector)

        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results

    def save(self, path_prefix: str) -> None:
        """
        Persist the FAISS index and metadata to disk.
        Creates: {path_prefix}.index and {path_prefix}.meta.json
        """
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        faiss.write_index(self.index, f"{path_prefix}.index")
        with open(f"{path_prefix}.meta.json", "w") as f:
            json.dump({"dim": self.dim, "metadata": self.metadata}, f)
        print(f"[INFO] Saved FAISS index ({self.index.ntotal} vectors) to {path_prefix}.index")

    @classmethod
    def load(cls, path_prefix: str) -> "FaissStore":
        """Load a previously saved FAISS index + metadata from disk."""
        with open(f"{path_prefix}.meta.json", "r") as f:
            data = json.load(f)

        store = cls(dim=data["dim"])
        store.index = faiss.read_index(f"{path_prefix}.index")
        store.metadata = data["metadata"]
        return store


def main():
    """Quick manual test with random vectors."""
    dim = 384
    store = FaissStore(dim=dim)

    dummy_vectors = np.random.rand(3, dim).astype("float32")
    dummy_chunks = ["chunk A text", "chunk B text", "chunk C text"]
    store.add(dummy_vectors, dummy_chunks, source="test.pdf")

    results = store.search(dummy_vectors[0], top_k=2)
    print("[INFO] Search results:")
    for r in results:
        print(f"  score={r['score']:.4f} source={r['source']} text={r['text']}")


if __name__ == "__main__":
    main()
