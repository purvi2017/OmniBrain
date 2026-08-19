"""
faiss_store.py
--------------

FAISS-based vector storage and semantic similarity search.

Supports:
- Embedding vector storage
- Cosine similarity search
- Document metadata
- document_id
- filename
- source
- chunk_id
- Chunk text
- Saving and loading FAISS indexes
"""

import os
import json
from typing import List, Dict, Any, Optional

import numpy as np
import faiss


class FaissStore:
    """
    FAISS vector store for document chunks.

    Uses normalized vectors with IndexFlatIP so that
    inner product represents cosine similarity.
    """

    def __init__(self, dim: int):
        """
        Initialize the FAISS vector store.

        Args:
            dim: Dimension of the embedding vectors.
        """

        self.dim = dim

        # Exact similarity search using inner product.
        # Since vectors are normalized, this is cosine similarity.
        self.index = faiss.IndexFlatIP(dim)

        # Metadata is stored separately because FAISS
        # stores vectors but does not store document metadata.
        self.metadata: List[Dict[str, Any]] = []

    # ---------------------------------------------------------
    # ADD DOCUMENT CHUNKS
    # ---------------------------------------------------------

    def add(
        self,
        vectors: np.ndarray,
        chunks: List[str],
        source: str = "",
        document_id: str = "",
        filename: str = "",
    ) -> None:
        """
        Add document chunk embeddings and metadata.

        Args:
            vectors:
                NumPy array with shape (n, dim).

            chunks:
                List of text chunks.

            source:
                Original document path or source.

            document_id:
                Unique identifier for the document.

            filename:
                Original uploaded PDF filename.
        """

        # Validate number of vectors and chunks
        if len(vectors) != len(chunks):
            raise ValueError(
                "vectors and chunks must have the same length"
            )

        # Nothing to add
        if vectors.size == 0:
            return

        # Convert to float32
        vectors = np.asarray(
            vectors,
            dtype=np.float32
        )

        # Check dimensions
        if vectors.ndim != 2:
            raise ValueError(
                "vectors must be a 2-dimensional NumPy array"
            )

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension mismatch. "
                f"Expected {self.dim}, "
                f"received {vectors.shape[1]}"
            )

        # -----------------------------------------------------
        # Normalize vectors
        # -----------------------------------------------------

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        # Avoid division by zero
        norms[norms == 0] = 1.0

        normalized_vectors = (
            vectors / norms
        )

        # -----------------------------------------------------
        # Add vectors to FAISS
        # -----------------------------------------------------

        self.index.add(
            normalized_vectors.astype(
                np.float32
            )
        )

        # -----------------------------------------------------
        # Add metadata
        # -----------------------------------------------------

        start_chunk_id = len(self.metadata)

        for i, chunk in enumerate(chunks):

            metadata = {
                "text": chunk,
                "source": source,
                "document_id": document_id,
                "filename": filename,
                "chunk_id": start_chunk_id + i,
            }

            self.metadata.append(metadata)

        print(
            f"[INFO] Added {len(chunks)} vectors "
            f"to FAISS index"
        )

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for the most similar document chunks.

        Args:
            query_vector:
                Query embedding with shape (dim,) or (1, dim).

            top_k:
                Number of results to return.

            document_id:
                Optional document ID filter.

        Returns:
            List of dictionaries containing:

            {
                "text": "...",
                "source": "...",
                "document_id": "...",
                "filename": "...",
                "chunk_id": 0,
                "score": 0.85
            }
        """

        # Empty index
        if self.index.ntotal == 0:
            return []

        # -----------------------------------------------------
        # Prepare query vector
        # -----------------------------------------------------

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(
                1, -1
            )

        if query_vector.shape[1] != self.dim:
            raise ValueError(
                f"Query vector dimension mismatch. "
                f"Expected {self.dim}, "
                f"received {query_vector.shape[1]}"
            )

        # Normalize query
        norm = np.linalg.norm(
            query_vector,
            axis=1,
            keepdims=True
        )

        norm[norm == 0] = 1.0

        query_vector = (
            query_vector / norm
        )

        # -----------------------------------------------------
        # Search FAISS
        # -----------------------------------------------------

        search_k = min(
            max(top_k, 1),
            self.index.ntotal
        )

        scores, indices = self.index.search(
            query_vector.astype(np.float32),
            search_k
        )

        # -----------------------------------------------------
        # Build results
        # -----------------------------------------------------

        results = []

        for score, index_position in zip(
            scores[0],
            indices[0]
        ):

            if index_position == -1:
                continue

            if index_position >= len(
                self.metadata
            ):
                continue

            metadata = self.metadata[
                index_position
            ].copy()

            metadata["score"] = float(score)

            # Optional document filtering
            if (
                document_id
                and metadata.get("document_id")
                != document_id
            ):
                continue

            results.append(metadata)

        return results

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    def save(
        self,
        path_prefix: str
    ) -> None:
        """
        Save FAISS index and metadata.

        Creates:

            path_prefix.index
            path_prefix.meta.json
        """

        directory = os.path.dirname(
            path_prefix
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        # Save FAISS index
        index_path = (
            f"{path_prefix}.index"
        )

        faiss.write_index(
            self.index,
            index_path
        )

        # Save metadata
        metadata_path = (
            f"{path_prefix}.meta.json"
        )

        data = {
            "dim": self.dim,
            "metadata": self.metadata,
        }

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"[INFO] Saved FAISS index "
            f"({self.index.ntotal} vectors)"
        )

        print(
            f"[INFO] Index: {index_path}"
        )

        print(
            f"[INFO] Metadata: {metadata_path}"
        )

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        path_prefix: str
    ) -> "FaissStore":
        """
        Load a previously saved FAISS index
        and metadata.

        Args:
            path_prefix:
                Path prefix used during save().
        """

        metadata_path = (
            f"{path_prefix}.meta.json"
        )

        index_path = (
            f"{path_prefix}.index"
        )

        if not os.path.exists(
            metadata_path
        ):
            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{metadata_path}"
            )

        if not os.path.exists(
            index_path
        ):
            raise FileNotFoundError(
                f"FAISS index file not found: "
                f"{index_path}"
            )

        # Load metadata
        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        # Create store
        store = cls(
            dim=data["dim"]
        )

        # Load FAISS index
        store.index = faiss.read_index(
            index_path
        )

        # Load metadata
        store.metadata = data.get(
            "metadata",
            []
        )

        # Validate index and metadata
        if (
            store.index.ntotal
            != len(store.metadata)
        ):
            raise ValueError(
                "FAISS index and metadata "
                "length do not match. "
                f"Vectors: {store.index.ntotal}, "
                f"Metadata: {len(store.metadata)}"
            )

        print(
            f"[INFO] Loaded FAISS index "
            f"({store.index.ntotal} vectors)"
        )

        return store

    # ---------------------------------------------------------
    # DOCUMENT INFORMATION
    # ---------------------------------------------------------

    def get_document_ids(self) -> List[str]:
        """
        Return unique document IDs stored in the index.
        """

        document_ids = []

        for metadata in self.metadata:

            document_id = metadata.get(
                "document_id"
            )

            if (
                document_id
                and document_id not in document_ids
            ):
                document_ids.append(
                    document_id
                )

        return document_ids

    def get_document_chunks(
        self,
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Return all chunks belonging to
        a particular document.
        """

        return [
            metadata
            for metadata in self.metadata
            if metadata.get("document_id")
            == document_id
        ]

    def count(self) -> int:
        """
        Return total number of vectors.
        """

        return self.index.ntotal


# ---------------------------------------------------------
# MANUAL TEST
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("FAISS VECTOR STORE TEST")
    print("=" * 70)

    dimension = 384

    # Create store
    store = FaissStore(
        dim=dimension
    )

    # Dummy vectors
    vectors = np.random.rand(
        3,
        dimension
    ).astype(np.float32)

    # Dummy chunks
    chunks = [
        "This is the first document chunk.",
        "This is the second document chunk.",
        "This is the third document chunk.",
    ]

    # Add vectors
    store.add(
        vectors=vectors,
        chunks=chunks,
        source="test.pdf",
        document_id="DOC-001",
        filename="test.pdf",
    )

    print(
        f"\n[INFO] Total vectors: "
        f"{store.count()}"
    )

    # Search
    results = store.search(
        vectors[0],
        top_k=2
    )

    print("\n[INFO] Search results:")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n#{i}"
        )

        print(
            f"Score       : "
            f"{result['score']:.4f}"
        )

        print(
            f"Document ID : "
            f"{result['document_id']}"
        )

        print(
            f"Filename    : "
            f"{result['filename']}"
        )

        print(
            f"Chunk ID    : "
            f"{result['chunk_id']}"
        )

        print(
            f"Text        : "
            f"{result['text']}"
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    test_path = (
        "vector_db/test_dynamic_index"
    )

    store.save(
        test_path
    )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    loaded_store = FaissStore.load(
        test_path
    )

    print(
        f"\n[INFO] Reloaded vectors: "
        f"{loaded_store.count()}"
    )

    print(
        f"[INFO] Document IDs: "
        f"{loaded_store.get_document_ids()}"
    )

    print("\n" + "=" * 70)
    print(
        "RESULT: FAISS VECTOR STORE TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()