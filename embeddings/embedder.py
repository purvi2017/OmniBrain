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
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
FALLBACK_DIM = 384  # matches all-MiniLM-L6-v2 so downstream code (FAISS dim) doesn't care which path ran

_model_cache = {}
_offline_fallback_active = False


def get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Load (and cache) a SentenceTransformer model so we don't reload
    it from disk on every call.
    """
    if model_name not in _model_cache:
        print(f"[INFO] Loading embedding model: {model_name} (first load may take a moment)...")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _hashing_fallback_embed(chunks: List[str], dim: int = FALLBACK_DIM) -> np.ndarray:
    """
    Deterministic, dependency-free embedding fallback used ONLY when the
    real sentence-transformers model can't be downloaded (e.g. no internet
    access to Hugging Face in a restricted/offline environment).

    This is NOT a semantic embedding — it's a hashed bag-of-words vector,
    good enough to prove the chunk -> vector -> FAISS pipeline mechanics
    work end-to-end, but it will NOT give meaningful semantic search
    quality. Once run on a machine with normal internet access, this
    fallback is skipped automatically and the real model is used.
    """
    vectors = np.zeros((len(chunks), dim), dtype="float32")
    for i, chunk in enumerate(chunks):
        for word in chunk.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vectors[i, h % dim] += 1.0
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] /= norm
    return vectors


def embed_chunks(chunks: List[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks.

    Args:
        chunks: List of text chunks (strings).
        model_name: Name of the sentence-transformers model to use.

    Returns:
        A numpy array of shape (num_chunks, embedding_dim), dtype float32.
    """
    global _offline_fallback_active

    if not chunks:
        return np.array([])

    try:
        model = get_model(model_name)
        embeddings = model.encode(
            chunks,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")
    except Exception as e:
        if not _offline_fallback_active:
            print(f"[WARN] Could not load '{model_name}' from Hugging Face ({type(e).__name__}). "
                  f"Falling back to a local hashing-based embedder for this environment. "
                  f"This is for offline testing only — on a machine with internet access, "
                  f"the real semantic model will be used automatically.")
            _offline_fallback_active = True
        return _hashing_fallback_embed(chunks)


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
