"""
qdrant_store.py
---------------
Alternative vector storage backend using Qdrant, for teams that want
a server-based vector DB with metadata filtering and persistence
built in (vs. FAISS which is in-process/local only).

Requires a running Qdrant instance. Easiest way to get one for local
dev is Docker:
    docker run -p 6333:6333 qdrant/qdrant

Install the client:
    pip install qdrant-client

Usage:
    from vector_db.qdrant_store import QdrantStore

    store = QdrantStore(collection_name="documents", dim=384)
    store.add(vectors, chunks, source="sample.pdf")
    results = store.search(query_vector, top_k=3)
"""

from typing import List, Dict, Any
import uuid

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantStore:
    def __init__(self, collection_name: str, dim: int, host: str = "localhost", port: int = 6333):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. Run: pip install qdrant-client"
            )

        self.collection_name = collection_name
        self.dim = dim
        self.client = QdrantClient(host=host, port=port)

        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, vectors: np.ndarray, chunks: List[str], source: str = "unknown") -> None:
        """Add vectors + associated text chunks to the Qdrant collection."""
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        if vectors.size == 0:
            return

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={"text": chunk, "source": source},
            )
            for vec, chunk in zip(vectors, chunks)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for the top_k most similar chunks to the query vector."""
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
        )
        return [
            {"text": h.payload.get("text"), "source": h.payload.get("source"), "score": h.score}
            for h in hits
        ]


def main():
    if not QDRANT_AVAILABLE:
        print("[INFO] qdrant-client not installed — install it and run a Qdrant "
              "server to test this module. See docstring for setup instructions.")
        return

    dim = 384
    store = QdrantStore(collection_name="test_collection", dim=dim)
    dummy_vectors = np.random.rand(3, dim).astype("float32")
    dummy_chunks = ["chunk A text", "chunk B text", "chunk C text"]
    store.add(dummy_vectors, dummy_chunks, source="test.pdf")

    results = store.search(dummy_vectors[0], top_k=2)
    print("[INFO] Search results:")
    for r in results:
        print(f"  score={r['score']:.4f} source={r['source']} text={r['text']}")


if __name__ == "__main__":
    main()
