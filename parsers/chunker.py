"""
chunker.py
----------
Splits extracted document text into overlapping chunks suitable
for embedding generation.

Chunking strategy:
- Target chunk size: 500-1000 characters (default 800)
- Splits on paragraph/sentence boundaries where possible, so we
  don't cut a sentence in half.
- Includes a small overlap between chunks (default 100 chars) so
  context isn't lost at chunk boundaries — this improves retrieval
  quality for RAG-style pipelines.

Usage:
    from parsers.chunker import chunk_text
    chunks = chunk_text(text, chunk_size=800, overlap=100)
"""

import re
from typing import List


def _split_into_sentences(text: str) -> List[str]:
    """Naive sentence splitter (splits on '.', '!', '?' followed by whitespace)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into chunks of roughly `chunk_size` characters,
    respecting sentence boundaries, with `overlap` characters
    repeated between consecutive chunks.

    Args:
        text: The full text to split.
        chunk_size: Target max characters per chunk (500-1000 recommended).
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size")

    sentences = _split_into_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed chunk_size, close off the chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from the end of the previous chunk
            overlap_text = current_chunk[-overlap:] if overlap else ""
            current_chunk = (overlap_text + " " + sentence).strip()
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

        # If a single sentence itself is longer than chunk_size, hard-split it
        while len(current_chunk) > chunk_size:
            chunks.append(current_chunk[:chunk_size].strip())
            current_chunk = current_chunk[chunk_size - overlap:].strip()

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def chunk_pages(pages: List[dict], chunk_size: int = 800, overlap: int = 100) -> List[dict]:
    """
    Chunk page-separated text while tracking page numbers.

    Args:
        pages: List of dicts with {"page_number": int, "text": str}.
        chunk_size: Target characters per chunk.
        overlap: Character overlap.

    Returns:
        List of dicts: [{"text": str, "page_number": int, "chunk_index": int}]
    """
    if not pages:
        return []

    result = []
    chunk_index = 0

    for page in pages:
        page_num = page.get("page_number", 1)
        page_text = page.get("text", "")
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)

        for text in page_chunks:
            result.append(
                {
                    "text": text,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

    return result



def main():
    """Quick manual test of the chunker."""
    sample_text = (
        "This is a sample document. " * 50
    )
    chunks = chunk_text(sample_text, chunk_size=500, overlap=50)
    print(f"[INFO] Generated {len(chunks)} chunks from sample text.")
    for i, c in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ({len(c)} chars) ---")
        print(c)


if __name__ == "__main__":
    main()
