"""
embedder.py
-----------
Generates vector embeddings for text chunks using sentence-transformers.

Default model: all-MiniLM-L6-v2
- 384-dimensional embeddings
- Fast, small, good for prototyping/dev.
- Can be swapped for a larger model (e.g. all-mpnet-base-v2) later
  for better retrieval quality — see README for the trade-off notes.

Usage:
    from embeddings.embedder import embed_chunks
    vectors = embed_chunks(["chunk one text", "chunk two text"])
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

_model_cache = {}


def get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Load (and cache) a SentenceTransformer model so we don't reload
    it from disk on every call.
    """
    if model_name not in _model_cache:
        print(f"[INFO] Loading embedding model: {model_name} (first load may take a moment)...")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_chunks(chunks: List[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks.

    Args:
        chunks: List of text chunks (strings).
        model_name: Name of the sentence-transformers model to use.

    Returns:
        A numpy array of shape (num_chunks, embedding_dim), dtype float32.
    """
    if not chunks:
        return np.array([])

    model = get_model(model_name)
    embeddings = model.encode(
        chunks,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def main():
    """Quick manual test of embedding generation."""
    sample_chunks = [
        "This is the first sample chunk about document processing.",
        "This is the second sample chunk about vector databases.",
    ]
    vectors = embed_chunks(sample_chunks)
    print(f"[INFO] Generated embeddings shape: {vectors.shape}")
    print(f"[INFO] First vector (first 5 dims): {vectors[0][:5]}")


if __name__ == "__main__":
    main()
