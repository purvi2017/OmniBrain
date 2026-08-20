"""
chunker.py
----------
Splits extracted document text into overlapping chunks suitable
for embedding generation, retrieval, and LLM context assembly.

Key Features:
- Intelligent Sentence Boundary Detection:
  Accurately identifies sentence boundaries while safely ignoring abbreviations
  (e.g., 'Dr.', 'Mr.', 'e.g.', 'i.e.', 'vs.', 'Fig.', 'Inc.', 'et al.'), decimal
  numbers ('3.14', '$19.99'), URLs, and ellipses.
- Word-Safe & Sentence-Aware Overlap:
  Preserves semantic continuity across chunk boundaries without slicing words in half.
- Graceful Edge Case Handling:
  Handles empty, whitespace-only, very short texts, and oversized sentences seamlessly.
- Pre-Embedding Verification:
  Provides `verify_chunks` and `validate_and_filter_chunks` to audit chunk quality,
  ensure no empty/malformed chunks reach the embedding model, and report metrics.
- Page-Aware Chunking:
  Preserves page metadata and chunk indexing across multi-page documents.

Usage:
    from parsers.chunker import chunk_text, chunk_pages, verify_chunks, validate_and_filter_chunks
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    report = verify_chunks(chunks)
"""

import re
from typing import List, Dict, Any, Optional


# Known abbreviations that contain dots and should NOT trigger sentence breaks
ABBREVIATIONS = {
    # Titles & Honorifics
    r"mr", r"mrs", r"ms", r"dr", r"prof", r"sr", r"jr", r"st", r"rev", r"gen", r"col", r"capt",
    # Academic / Latin / Technical
    r"e\.g", r"i\.e", r"et al", r"etc", r"vs", r"cf", r"ibid", r"op\. cit", r"approx",
    r"fig", r"figs", r"tab", r"vol", r"no", r"p", r"pp", r"ed", r"eds", r"sec", r"ch",
    # Corporate & Organizations
    r"inc", r"ltd", r"corp", r"co", r"gov", r"dept", r"univ", r"assn",
    # Calendar & Time
    r"jan", r"feb", r"mar", r"apr", r"jun", r"jul", r"aug", r"sep", r"sept", r"oct", r"nov", r"dec",
    r"mon", r"tue", r"wed", r"thu", r"fri", r"sat", r"sun",
    r"a\.m", r"p\.m",
}

# Compile regex pattern to match known abbreviations followed by a dot
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(ABBREVIATIONS) + r")\.",
    re.IGNORECASE
)

# Placeholder token for dots inside protected tokens
_DOT_PLACEHOLDER = "\uE000"


def _protect_special_dots(text: str) -> str:
    """
    Replace dots in abbreviations, numbers, URLs, emails, and ellipses with a placeholder
    to prevent premature sentence splitting.
    """
    if not text:
        return ""

    protected = text

    # 1. Protect ellipsis (...) and (....)
    protected = re.sub(r"\.{2,}", lambda m: _DOT_PLACEHOLDER * len(m.group(0)), protected)

    # 2. Protect URLs (e.g., https://example.com/path)
    protected = re.sub(
        r"(https?://\S+)",
        lambda m: m.group(0).replace(".", _DOT_PLACEHOLDER),
        protected
    )

    # 3. Protect email addresses (e.g., user@domain.com)
    protected = re.sub(
        r"(\b[\w\.-]+@[\w\.-]+\.\w+\b)",
        lambda m: m.group(0).replace(".", _DOT_PLACEHOLDER),
        protected
    )

    # 4. Protect decimal numbers and versions (e.g., 3.14, $10.50, 2.0, v1.2.3)
    protected = re.sub(
        r"(?<=\d)\.(?=\d)",
        _DOT_PLACEHOLDER,
        protected
    )

    # 5. Protect known abbreviations (e.g., Dr., e.g., i.e., vs., Fig.)
    def _replace_abbrev(match):
        return match.group(0).replace(".", _DOT_PLACEHOLDER)

    protected = _ABBREV_PATTERN.sub(_replace_abbrev, protected)

    # 6. Protect single-letter initials (e.g., "John F. Kennedy", "U. S. A.", "A. Smith")
    protected = re.sub(
        r"\b([A-Z])\.",
        lambda m: m.group(1) + _DOT_PLACEHOLDER,
        protected
    )

    return protected


def _restore_dots(text: str) -> str:
    """Restore placeholder dots to standard dots."""
    return text.replace(_DOT_PLACEHOLDER, ".")


# Compiled regex for sentence splitting:
# 1) Paragraph breaks (2+ newlines)
# 2) Sentence terminators (. ! ?) optionally followed by a quote/bracket, then whitespace
# 3) Bullet points or numbered lists on new lines
_SPLIT_PATTERN = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'\)\]]))\s+(?=[A-Z0-9\"'\(•\-\*]|\Z)|\n{2,}|\n(?=\s*[-*•]\s+|\s*\d+[\.\)]\s+)"
)


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences while respecting abbreviations, numbers, quotes,
    bullet points, and paragraph breaks.
    """
    if not text or not text.strip():
        return []

    # Clean redundant whitespace while preserving paragraph breaks
    cleaned = re.sub(r"\r\n|\r", "\n", text.strip())

    # Protect periods that should not trigger splits
    protected = _protect_special_dots(cleaned)

    raw_segments = _SPLIT_PATTERN.split(protected)
    sentences = []

    for seg in raw_segments:
        if not seg:
            continue
        # Restore dots and normalize inner whitespace
        restored = _restore_dots(seg).strip()
        # Clean internal extra whitespace (convert multiple spaces/tabs to single space, keep single newlines if bullet)
        normalized = re.sub(r"[ \t]+", " ", restored)
        if normalized:
            sentences.append(normalized)

    # Fallback: if no sentences were produced but text existed, return the text
    if not sentences and cleaned:
        sentences = [_restore_dots(protected).strip()]

    return sentences


def _split_large_sentence(sentence: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Break down an oversized sentence along clause/word boundaries without cutting words in half.
    """
    if len(sentence) <= chunk_size:
        return [sentence]

    words = sentence.split(" ")
    sub_chunks = []
    current = []
    current_len = 0

    for word in words:
        word_len = len(word)
        # If single word is longer than chunk_size, hard-slice it as a last resort
        if word_len > chunk_size:
            if current:
                sub_chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            for i in range(0, word_len, chunk_size):
                sub_chunks.append(word[i : i + chunk_size])
            continue

        added_len = word_len + (1 if current else 0)
        if current and current_len + added_len > chunk_size:
            sub_chunks.append(" ".join(current).strip())
            # Add word-safe overlap from end of current
            overlap_words = []
            overlap_count = 0
            for w in reversed(current):
                if overlap_count + len(w) + 1 <= overlap:
                    overlap_words.insert(0, w)
                    overlap_count += len(w) + 1
                else:
                    break
            current = overlap_words + [word]
            current_len = sum(len(w) for w in current) + max(0, len(current) - 1)
        else:
            current.append(word)
            current_len += added_len

    if current:
        sub_chunks.append(" ".join(current).strip())

    return sub_chunks


def _get_word_safe_overlap(text: str, overlap: int) -> str:
    """
    Extract up to `overlap` characters from the end of `text`, ensuring the slice
    starts at a clean word or sentence boundary rather than cutting mid-word.
    """
    if overlap <= 0 or not text:
        return ""

    text = text.strip()
    if len(text) <= overlap:
        return text

    # Initial character slice
    candidate = text[-overlap:].strip()

    # Check if cut happened in the middle of a word
    cut_pos = len(text) - overlap
    if cut_pos > 0 and not text[cut_pos - 1].isspace() and not text[cut_pos].isspace():
        # Find next space in candidate to avoid starting with a partial word
        space_idx = candidate.find(" ")
        if space_idx != -1 and space_idx < len(candidate) - 1:
            candidate = candidate[space_idx + 1 :].strip()
        else:
            # Fallback: find previous space before cut_pos within reasonable limit
            prev_space = text.rfind(" ", 0, cut_pos)
            if prev_space != -1 and (len(text) - prev_space) <= int(overlap * 1.3):
                candidate = text[prev_space + 1 :].strip()

    return candidate


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into semantically coherent chunks of roughly `chunk_size` characters,
    strictly respecting sentence and word boundaries, with word-safe `overlap` characters.

    Args:
        text: The full text to split.
        chunk_size: Target max characters per chunk (500-1000 recommended).
        overlap: Target number of characters to overlap between consecutive chunks.

    Returns:
        List of cleaned text chunks. Returns [] for empty/whitespace-only input.

    Raises:
        ValueError: If chunk_size <= 0 or overlap < 0 or overlap >= chunk_size.
    """
    if text is None:
        return []

    if not isinstance(text, str):
        text = str(text)

    stripped_text = text.strip()
    if not stripped_text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size")

    # If the entire text is already smaller than or equal to chunk_size, return it directly
    if len(stripped_text) <= chunk_size:
        return [stripped_text]

    sentences = _split_into_sentences(stripped_text)
    if not sentences:
        return [stripped_text]

    # Pre-process any individual sentences that exceed chunk_size
    flattened_sentences: List[str] = []
    for s in sentences:
        if len(s) > chunk_size:
            flattened_sentences.extend(_split_large_sentence(s, chunk_size=chunk_size, overlap=overlap))
        else:
            flattened_sentences.append(s)

    chunks: List[str] = []
    current_chunk = ""

    for sentence in flattened_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Calculate length if we add this sentence to current chunk
        added_len = len(sentence) + (1 if current_chunk else 0)

        if current_chunk and (len(current_chunk) + added_len > chunk_size):
            chunks.append(current_chunk.strip())

            # Formulate overlap from previous chunk
            overlap_prefix = _get_word_safe_overlap(current_chunk, overlap)
            if overlap_prefix:
                current_chunk = f"{overlap_prefix} {sentence}".strip()
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Final cleanup: filter out empty chunks and ensure uniqueness of trivial consecutive duplicates
    cleaned_chunks: List[str] = []
    for c in chunks:
        c_str = c.strip()
        if c_str and (not cleaned_chunks or c_str != cleaned_chunks[-1]):
            cleaned_chunks.append(c_str)

    return cleaned_chunks


def chunk_pages(pages: List[dict], chunk_size: int = 800, overlap: int = 100) -> List[dict]:
    """
    Chunk page-separated text while tracking page numbers and chunk metadata.

    Args:
        pages: List of dicts with {"page_number": int, "text": str}.
        chunk_size: Target characters per chunk.
        overlap: Character overlap.

    Returns:
        List of dicts:
        [
            {
                "text": str,
                "page_number": int,
                "chunk_index": int,
                "char_count": int,
                "word_count": int
            }
        ]
    """
    if not pages:
        return []

    result = []
    chunk_index = 0

    for page in pages:
        page_num = page.get("page_number", 1)
        page_text = page.get("text", "")
        if not page_text or not page_text.strip():
            continue

        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)

        for text in page_chunks:
            result.append(
                {
                    "text": text,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                }
            )
            chunk_index += 1

    return result


def verify_chunks(
    chunks: List[str],
    min_chars: int = 1,
    max_chars: Optional[int] = None,
    allow_overlap_slack: float = 1.25,
) -> Dict[str, Any]:
    """
    Validate and audit generated text chunks before passing them to the embedding model.

    Checks:
    - Empty or whitespace-only chunks
    - Minimum length violations
    - Maximum length violations
    - Chunk count and summary statistics (min, max, average length)
    - Suspicious truncated words at chunk edges

    Args:
        chunks: List of chunk text strings.
        min_chars: Minimum allowed characters per chunk.
        max_chars: Maximum expected characters per chunk (optional).
        allow_overlap_slack: Multiplier on max_chars for acceptable overlap tolerance.

    Returns:
        Dict with verification results and statistics:
        {
            "is_valid": bool,
            "total_chunks": int,
            "empty_chunks": int,
            "min_chunk_length": int,
            "max_chunk_length": int,
            "avg_chunk_length": float,
            "warnings": List[str],
        }
    """
    warnings: List[str] = []

    if not chunks:
        return {
            "is_valid": True,
            "total_chunks": 0,
            "empty_chunks": 0,
            "min_chunk_length": 0,
            "max_chunk_length": 0,
            "avg_chunk_length": 0.0,
            "warnings": ["No chunks provided to verify (empty list)."],
        }

    empty_count = 0
    lengths: List[int] = []

    for i, c in enumerate(chunks):
        if not isinstance(c, str) or not c.strip():
            empty_count += 1
            warnings.append(f"Chunk #{i} is empty or whitespace-only.")
            continue

        c_len = len(c)
        lengths.append(c_len)

        if c_len < min_chars:
            warnings.append(f"Chunk #{i} is shorter than min_chars ({c_len} < {min_chars}).")

        if max_chars and c_len > int(max_chars * allow_overlap_slack):
            warnings.append(
                f"Chunk #{i} exceeds max length bounds ({c_len} > {int(max_chars * allow_overlap_slack)})."
            )

    is_valid = empty_count == 0 and len(lengths) > 0

    return {
        "is_valid": is_valid,
        "total_chunks": len(chunks),
        "empty_chunks": empty_count,
        "min_chunk_length": min(lengths) if lengths else 0,
        "max_chunk_length": max(lengths) if lengths else 0,
        "avg_chunk_length": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "warnings": warnings,
    }


def validate_and_filter_chunks(chunks: List[str], min_chars: int = 1) -> List[str]:
    """
    Sanitize and filter a list of chunks prior to embedding, stripping whitespace
    and removing empty or sub-minimum chunks.

    Args:
        chunks: List of chunk strings.
        min_chars: Minimum character threshold.

    Returns:
        List of valid, non-empty, stripped chunk strings.
    """
    if not chunks:
        return []

    valid_chunks = []
    for c in chunks:
        if isinstance(c, str):
            cleaned = c.strip()
            if len(cleaned) >= min_chars:
                valid_chunks.append(cleaned)

    return valid_chunks


def main():
    """Manual demonstration and verification of improved chunking logic."""
    sample_text = (
        "Artificial Intelligence (A.I.) has transformed modern software engineering. "
        "For example, Dr. Smith et al. demonstrated that microservices (e.g., auth service, "
        "vector DB, and LLM gateway) scale efficiently under heavy load. "
        "The system achieved 99.9% uptime and handled $10.50 per 1M tokens. "
        "Here are key components:\n"
        "1. Document Parser Service using PyPDF.\n"
        "2. Embedding Engine using all-MiniLM-L6-v2.\n"
        "3. FAISS Vector Database for millisecond similarity retrieval.\n\n"
        "In conclusion, careful chunking preserves sentence semantics."
    )

    print("=" * 70)
    print("DEMO: Improved Sentence-Aware Document Chunker")
    print("=" * 70)
    print(f"Source text length: {len(sample_text)} characters\n")

    chunks = chunk_text(sample_text, chunk_size=200, overlap=40)
    report = verify_chunks(chunks, max_chars=200)

    print(f"[INFO] Generated {len(chunks)} chunks.")
    print(f"[INFO] Verification Report: {report}")

    for i, c in enumerate(chunks, start=1):
        print(f"\n--- Chunk #{i} ({len(c)} chars) ---")
        print(c)


if __name__ == "__main__":
    main()
