"""
rag_llm.py
----------
Day 8: Enhanced RAG + LLM Pipeline with Multi-Document Retrieval,
Adaptive Threshold Filtering, and Precision Source Attribution.

Flow:
    User Query
        ↓
    Query Embedding (all-MiniLM-L6-v2)
        ↓
    Multi-Doc FAISS Retrieval
        ↓
    Adaptive Threshold Filtering (min_score & max_score_drop)
        ↓
    Confidence Scoring & Deduplication
        ↓
    Context Assembly with Page & Doc Citations
        ↓
    Gemini LLM Generation (Strict Grounding Rules)
        ↓
    Answer + Rich Source Attributions

Usage:
    python rag_llm.py test_files/sample.pdf test_files/ai_architecture.pdf --query "How does the vector database communicate?"
    python rag_llm.py test_files/sample.pdf --top-k 3 --min-score 0.35

Environment:
    GEMINI_API_KEY must be set in your environment to call the Gemini API.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from parsers.pdf_extractor import extract_pages_from_pdf, extract_text_from_pdf
from parsers.chunker import chunk_text, chunk_pages, verify_chunks, validate_and_filter_chunks
from embeddings.embedder import embed_chunks
from vector_db.faiss_store import FaissStore


# ------------------------------------------------------------
# Configuration & Calibrated Thresholds
# ------------------------------------------------------------

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.35          # Calibrated for all-MiniLM-L6-v2 cosine similarity
DEFAULT_MAX_SCORE_DROP = 0.25     # Maximum score drop allowed from top-1 match
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100

NO_CONTEXT_MESSAGE = (
    "I could not find relevant information in the uploaded documents to answer this question."
)


def get_confidence_label(score: float) -> str:
    """Return a confidence categorization for a similarity score."""
    if score >= 0.55:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    return "LOW"


# ------------------------------------------------------------
# Gemini Client
# ------------------------------------------------------------

def create_gemini_client(api_key: Optional[str] = None):
    """Initialize and return a Gemini API client using google-genai."""
    key = api_key or os.getenv("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "Please set the GEMINI_API_KEY environment variable before running the LLM pipeline."
        )

    try:
        from google import genai
    except ImportError:
        raise RuntimeError(
            "google-genai is not installed.\n"
            "Install it via: pip install google-genai"
        )

    return genai.Client(api_key=key)


# ------------------------------------------------------------
# PDF Document Processing with Page-Level Metadata
# ------------------------------------------------------------

def process_pdf(
    pdf_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
):
    """
    Extract text page-by-page, create chunks, and generate vector embeddings.

    Args:
        pdf_path: Path to the input PDF file.
        chunk_size: Target character length per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        tuple: (chunks: List[str], vectors: np.ndarray, page_numbers: List[int])
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {pdf_path}")

    print(f"\n[INFO] Processing document: {path.name}")

    # 1. Extract pages
    try:
        pages = extract_pages_from_pdf(str(path))
    except Exception:
        # Fallback to single-block extraction if page extraction fails
        text = extract_text_from_pdf(str(path))
        pages = [{"page_number": 1, "text": text}]

    if not pages:
        raise ValueError(f"No text could be extracted from {path.name}")

    total_chars = sum(len(p.get("text", "")) for p in pages)
    print(f"[INFO] Extracted {len(pages)} page(s), total {total_chars} characters")

    # 2. Chunk pages
    page_chunk_records = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    if not page_chunk_records:
        raise ValueError(f"No chunks could be created from {path.name}")

    chunks = [r["text"] for r in page_chunk_records]
    page_numbers = [r["page_number"] for r in page_chunk_records]

    # Pre-embedding verification
    verification = verify_chunks(chunks, max_chars=chunk_size)
    if not verification["is_valid"]:
        raise ValueError(f"Chunk verification failed for {path.name}: {verification['warnings']}")

    print(
        f"[INFO] Created {len(chunks)} chunks (verified: min={verification['min_chunk_length']}, "
        f"max={verification['max_chunk_length']}, avg={verification['avg_chunk_length']} chars)"
    )

    # 3. Generate embeddings
    vectors = embed_chunks(chunks)
    print(
        f"[INFO] Generated embeddings: "
        f"{vectors.shape[0]} vectors x {vectors.shape[1]} dimensions"
    )

    return chunks, vectors, page_numbers


# ------------------------------------------------------------
# Multi-Document Store Ingestion
# ------------------------------------------------------------

def build_store(
    pdf_paths: List[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
):
    """
    Process multiple PDFs and index all chunks in a single FAISS vector store.

    Args:
        pdf_paths: List of PDF file paths.
        chunk_size: Chunk size in characters.
        overlap: Overlap size in characters.

    Returns:
        tuple: (store: FaissStore, document_metadata: List[Dict[str, Any]])
    """
    if not pdf_paths:
        raise ValueError("At least one PDF path must be provided.")

    all_vectors = []
    all_chunks = []
    all_pages = []
    document_metadata = []

    for pdf_path in pdf_paths:
        path = Path(pdf_path)
        chunks, vectors, pages = process_pdf(
            str(path), chunk_size=chunk_size, overlap=overlap
        )

        all_vectors.append(vectors)
        all_chunks.extend(chunks)
        all_pages.extend(pages)

        document_metadata.append(
            {
                "document_id": path.stem,
                "filename": path.name,
                "chunk_count": len(chunks),
                "page_count": max(pages) if pages else 1,
                "path": str(path.resolve()),
            }
        )

    combined_vectors = np.vstack(all_vectors)

    print(f"\n[INFO] Total documents indexed: {len(document_metadata)}")
    print(f"[INFO] Total chunks indexed: {len(all_chunks)}")
    print(f"[INFO] Vector dimension: {combined_vectors.shape[1]}")

    store = FaissStore(dim=combined_vectors.shape[1])

    offset = 0
    for doc in document_metadata:
        count = doc["chunk_count"]
        doc_vectors = combined_vectors[offset : offset + count]
        doc_chunks = all_chunks[offset : offset + count]
        doc_pages = all_pages[offset : offset + count]

        store.add(
            vectors=doc_vectors,
            chunks=doc_chunks,
            source=doc["filename"],
            document_id=doc["document_id"],
            filename=doc["filename"],
            pages=doc_pages,
        )
        offset += count

    return store, document_metadata


# ------------------------------------------------------------
# Adaptive Retrieval Layer & Threshold Tuning
# ------------------------------------------------------------

def retrieve_context(
    store: FaissStore,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    max_score_drop: float = DEFAULT_MAX_SCORE_DROP,
    document_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate embedding for query, retrieve chunks from FAISS,
    and apply adaptive threshold filtering to eliminate low-relevance noise.

    Args:
        store: FAISS vector store instance.
        query: User question string.
        top_k: Maximum number of chunks to retrieve.
        min_score: Minimum absolute cosine similarity threshold (0.0 to 1.0).
        max_score_drop: Max allowable score drop from the top candidate.
        document_id: Optional document ID to restrict search.

    Returns:
        List of filtered and ranked matching chunk dictionaries.
    """
    if not query or not query.strip():
        return []

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)
    print(f"Query: {query}")
    if document_id:
        print(f"[FILTER] Target Document ID: {document_id}")

    # Generate query embedding
    query_vector = embed_chunks([query])
    print(f"[INFO] Query embedding shape: {query_vector.shape}")

    # Similarity search
    raw_results = store.search(
        query_vector[0],
        top_k=top_k,
        document_id=document_id,
    )
    print(f"[INFO] FAISS returned {len(raw_results)} raw candidate result(s)")

    # 1. Apply absolute minimum similarity threshold
    threshold_results = [
        result
        for result in raw_results
        if float(result.get("score", 0.0)) >= min_score
    ]

    print(
        f"[INFO] Candidates meeting absolute threshold (>= {min_score:.2f}): "
        f"{len(threshold_results)}"
    )

    if not threshold_results:
        return []

    # 2. Apply relative score drop filtering (keep only high-confidence cluster)
    top_score = float(threshold_results[0].get("score", 0.0))
    adaptive_results = [
        r
        for r in threshold_results
        if (top_score - float(r.get("score", 0.0))) <= max_score_drop
    ]

    print(
        f"[INFO] Candidates after adaptive relative drop filtering (delta <= {max_score_drop:.2f}): "
        f"{len(adaptive_results)}"
    )

    # 3. Attach confidence tags
    for r in adaptive_results:
        score = float(r.get("score", 0.0))
        r["confidence"] = get_confidence_label(score)

    return adaptive_results


# ------------------------------------------------------------
# Context Preparation & Grounded Prompting
# ------------------------------------------------------------

def prepare_context(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a clean, multi-document context string for LLM.

    Args:
        results: List of retrieved chunk dictionaries.

    Returns:
        Formatted context string.
    """
    if not results:
        return ""

    context_parts = []
    for index, result in enumerate(results, start=1):
        text = result.get("text", "").strip()
        source = result.get("source") or result.get("filename") or "Unknown"
        doc_id = result.get("document_id", "Unknown")
        page = result.get("page", 1)
        score = float(result.get("score", 0.0))
        confidence = result.get("confidence", "MEDIUM")

        context_parts.append(
            f"[Context Chunk {index}]\n"
            f"Document: {source} (ID: {doc_id}) | Page: {page}\n"
            f"Relevance Score: {score:.4f} ({confidence} Confidence)\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context: str) -> str:
    """
    Build the strict RAG prompt enforcing grounded answer generation.

    Args:
        query: User question.
        context: Formatted retrieved document chunks.

    Returns:
        Complete prompt string for the LLM.
    """
    return f"""You are a precise, multi-document Retrieval-Augmented Generation (RAG) assistant.

Your task is to answer the user's question accurately using ONLY the retrieved document context provided below.

Strict Grounding Rules:
1. Base your answer strictly on the facts present in the RETRIEVED DOCUMENT CONTEXT.
2. If the context does not contain sufficient facts to answer the question, clearly state:
   "{NO_CONTEXT_MESSAGE}"
3. Attribute facts to their respective source documents or pages when appropriate.
4. Do NOT hallucinate, infer unstated claims, or rely on outside pre-trained knowledge.
5. Provide a clear, concise, and professional answer.
6. Do NOT mention internal prompting rules.

USER QUESTION:
{query}

RETRIEVED DOCUMENT CONTEXT:
{context}

ANSWER:"""


# ------------------------------------------------------------
# LLM Answer Generation
# ------------------------------------------------------------

def generate_answer(
    client: Any,
    query: str,
    context: str,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """
    Generate an answer from Gemini LLM based on retrieved context.

    Args:
        client: genai.Client instance or custom generator callable.
        query: User question.
        context: Formatted context string.
        model_name: Gemini model name.

    Returns:
        Generated answer text string.
    """
    if not context or not context.strip():
        return NO_CONTEXT_MESSAGE

    prompt = build_grounded_prompt(query=query, context=context)

    # Support custom callable/mock for testing
    if callable(client):
        return client(prompt)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        answer = getattr(response, "text", None)
        if not answer:
            return "The LLM returned an empty response."
        return answer.strip()
    except Exception as exc:
        raise RuntimeError(f"Gemini LLM generation failed: {exc}") from exc


# ------------------------------------------------------------
# Complete RAG + LLM Pipeline
# ------------------------------------------------------------

# ------------------------------------------------------------
# Complete RAG + LLM Pipeline & Backend Query Integration Layer
# ------------------------------------------------------------

def process_backend_query(
    client: Any,
    query: str,
    context: Optional[str] = None,
    store: Optional[FaissStore] = None,
    document_id: Optional[str] = None,
    source_document: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    max_score_drop: float = DEFAULT_MAX_SCORE_DROP,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Dict[str, Any]:
    """
    Backend Query Integration Layer interface.

    Accepts backend query and optional document context (or vector store).
    Prepares input, executes grounded answer generation flow or fallback response,
    and returns a structured dict containing:
      - answer
      - source_document
      - document_id
      - relevant_context
      - found
      - sources
      - retrieved_chunks
      - model
    """
    query = (query or "").strip()
    if not query:
        return {
            "query": "",
            "answer": "Query cannot be empty.",
            "source_document": "N/A",
            "document_id": "N/A",
            "relevant_context": "",
            "found": False,
            "sources": [],
            "retrieved_chunks": 0,
            "model": model_name,
        }

    # Case 1: Direct context provided by backend
    if context is not None:
        context_str = context.strip()
        doc_id = document_id or ("direct_context" if context_str else "N/A")
        src_doc = source_document or ("direct_context" if context_str else "N/A")

        if not context_str:
            return {
                "query": query,
                "answer": NO_CONTEXT_MESSAGE,
                "source_document": src_doc,
                "document_id": doc_id,
                "relevant_context": "",
                "found": False,
                "sources": [],
                "retrieved_chunks": 0,
                "model": model_name,
            }

        answer = generate_answer(
            client=client,
            query=query,
            context=context_str,
            model_name=model_name,
        )

        found = (answer != NO_CONTEXT_MESSAGE)
        sources = []
        if found:
            sources.append(
                {
                    "source": src_doc,
                    "filename": src_doc,
                    "document_id": doc_id,
                    "chunk_id": 0,
                    "page": 1,
                    "score": 1.0,
                    "confidence": "HIGH",
                    "text_preview": context_str[:150].replace("\n", " "),
                }
            )

        return {
            "query": query,
            "answer": answer,
            "source_document": src_doc,
            "document_id": doc_id,
            "relevant_context": context_str,
            "found": found,
            "sources": sources,
            "retrieved_chunks": 1 if found else 0,
            "model": model_name,
        }

    # Case 2: Store provided, retrieve context via FAISS search
    if store is not None:
        return answer_query(
            client=client,
            store=store,
            query=query,
            top_k=top_k,
            min_score=min_score,
            max_score_drop=max_score_drop,
            document_id=document_id,
            model_name=model_name,
        )

    # Case 3: Neither direct context nor store provided
    return {
        "query": query,
        "answer": NO_CONTEXT_MESSAGE,
        "source_document": "N/A",
        "document_id": "N/A",
        "relevant_context": "",
        "found": False,
        "sources": [],
        "retrieved_chunks": 0,
        "model": model_name,
    }


def answer_query(
    client: Any,
    store: FaissStore,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    max_score_drop: float = DEFAULT_MAX_SCORE_DROP,
    document_id: Optional[str] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Dict[str, Any]:
    """
    Execute end-to-end multi-document RAG flow:
    Query -> Embed -> Multi-Doc FAISS Search -> Adaptive Filter -> LLM -> Structured Response.

    Returns:
        Dict containing answer, source_document, document_id, relevant_context, and source attributions.
    """
    query = (query or "").strip()
    doc_id_fallback = document_id or "N/A"
    src_doc_fallback = f"{document_id}.pdf" if document_id else "N/A"

    if not query:
        return {
            "query": "",
            "answer": "Query cannot be empty.",
            "source_document": src_doc_fallback,
            "document_id": doc_id_fallback,
            "relevant_context": "",
            "found": False,
            "sources": [],
            "retrieved_chunks": 0,
            "model": model_name,
        }

    # Step 1: Retrieval + Adaptive Thresholding
    results = retrieve_context(
        store=store,
        query=query,
        top_k=top_k,
        min_score=min_score,
        max_score_drop=max_score_drop,
        document_id=document_id,
    )

    # Case: No relevant context found
    if not results:
        return {
            "query": query,
            "answer": NO_CONTEXT_MESSAGE,
            "source_document": src_doc_fallback,
            "document_id": doc_id_fallback,
            "relevant_context": "",
            "found": False,
            "sources": [],
            "retrieved_chunks": 0,
            "model": model_name,
        }

    # Step 2: Prepare context
    context = prepare_context(results)

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT PASSED TO LLM")
    print("=" * 70)
    print(context)

    # Step 3: Generate LLM answer
    print("\n" + "=" * 70)
    print(f"GENERATING ANSWER WITH LLM ({model_name})...")
    print("=" * 70)

    answer = generate_answer(
        client=client,
        query=query,
        context=context,
        model_name=model_name,
    )

    # Step 4: Structured source attribution
    sources = []
    for result in results:
        sources.append(
            {
                "source": result.get("source", "unknown"),
                "filename": result.get("filename", result.get("source", "unknown")),
                "document_id": result.get("document_id", ""),
                "chunk_id": result.get("chunk_id", -1),
                "page": result.get("page", 1),
                "score": round(float(result.get("score", 0.0)), 4),
                "confidence": result.get("confidence", "MEDIUM"),
                "text_preview": result.get("text", "")[:150].replace("\n", " "),
            }
        )

    # Determine primary source document and document ID
    primary_source_doc = sources[0]["filename"] if sources else "unknown"
    primary_doc_id = sources[0]["document_id"] if sources else "unknown"

    return {
        "query": query,
        "answer": answer,
        "source_document": primary_source_doc,
        "document_id": primary_doc_id,
        "relevant_context": context,
        "found": True,
        "sources": sources,
        "retrieved_chunks": len(results),
        "model": model_name,
    }


# ------------------------------------------------------------
# Object-Oriented Pipeline Interface
# ------------------------------------------------------------

class RAGLLMPipeline:
    """
    Unified Multi-Document RAG + LLM Pipeline class.
    """

    def __init__(
        self,
        store: Optional[FaissStore] = None,
        client: Optional[Any] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        max_score_drop: float = DEFAULT_MAX_SCORE_DROP,
    ):
        self.store = store
        self.client = client
        self.model_name = model_name
        self.top_k = top_k
        self.min_score = min_score
        self.max_score_drop = max_score_drop

    def load_documents(
        self,
        pdf_paths: List[str],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ):
        """Index multiple PDF documents into internal FAISS store."""
        self.store, metadata = build_store(
            pdf_paths=pdf_paths,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return metadata

    def load_existing_index(self, index_prefix: str):
        """Load a previously persisted FAISS index."""
        self.store = FaissStore.load(index_prefix)
        return self.store

    def init_client(self, api_key: Optional[str] = None):
        """Initialize Gemini client."""
        self.client = create_gemini_client(api_key=api_key)
        return self.client

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        max_score_drop: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute RAG query across indexed documents."""
        if self.store is None:
            raise RuntimeError("FAISS store is not loaded. Call load_documents() first.")
        if self.client is None:
            self.init_client()

        return answer_query(
            client=self.client,
            store=self.store,
            query=query,
            top_k=top_k or self.top_k,
            min_score=min_score if min_score is not None else self.min_score,
            max_score_drop=max_score_drop if max_score_drop is not None else self.max_score_drop,
            document_id=document_id,
            model_name=self.model_name,
        )


# ------------------------------------------------------------
# Formatted Output Display
# ------------------------------------------------------------

def print_response(query: str, response: Dict[str, Any]):
    """Display structured RAG response with citation details."""
    print("\n" + "=" * 70)
    print("FINAL RAG RESPONSE")
    print("=" * 70)

    print(f"\n[QUESTION]\n{query}")
    print(f"\n[GENERATED ANSWER]\n{response['answer']}")

    print("\n[SOURCE CITATIONS]")
    if not response.get("sources"):
        print("  No relevant source documents found.")
    else:
        for i, src in enumerate(response["sources"], start=1):
            print(f"  [{i}] File       : {src['source']}")
            print(f"      Document ID: {src.get('document_id', 'N/A')}")
            print(f"      Page       : {src.get('page', 1)}")
            print(f"      Similarity : {src['score']:.4f} ({src.get('confidence', 'N/A')})")
            print(f"      Excerpt    : {src['text_preview']}...")

    print(f"\n[METADATA]")
    print(f"  Retrieved Chunks : {response.get('retrieved_chunks', 0)}")
    print(f"  Context Found    : {response.get('found', False)}")
    print(f"  Model Used       : {response.get('model', 'N/A')}")

    print("\n" + "=" * 70)
    print("RESULT: RAG + LLM PIPELINE COMPLETED")
    print("=" * 70)


# ------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Day 8: Enhanced Multi-Document RAG + LLM Answer Generation"
    )

    parser.add_argument(
        "pdfs",
        nargs="+",
        help="Path(s) to PDF documents to index and query",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Question to ask about the uploaded document(s)",
    )
    parser.add_argument(
        "--document-id",
        type=str,
        default=None,
        help="Optional Document ID filter to constrain query to one document",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top matching chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--min-score",
        "-s",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Minimum cosine similarity score threshold (default: {DEFAULT_MIN_SCORE})",
    )
    parser.add_argument(
        "--max-drop",
        type=float,
        default=DEFAULT_MAX_SCORE_DROP,
        help=f"Maximum allowed score drop from top candidate (default: {DEFAULT_MAX_SCORE_DROP})",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini LLM model name (default: {DEFAULT_GEMINI_MODEL})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk character size (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Chunk character overlap (default: {DEFAULT_OVERLAP})",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DAY 8: MULTI-DOCUMENT RAG + LLM PIPELINE")
    print("=" * 70)

    try:
        # Step 1: Client
        print("\n[STEP 1] Initializing Gemini LLM Client...")
        client = create_gemini_client()
        print(f"[PASS] Gemini client ready (Model: {args.model})")

        # Step 2: Index documents
        print("\n[STEP 2] Ingesting and Indexing PDF Document(s)...")
        store, metadata = build_store(
            pdf_paths=args.pdfs,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        print(f"[PASS] {len(metadata)} document(s) indexed in FAISS store")

        # Step 3: Query
        query = args.query
        if not query:
            query = input("\nEnter your question: ").strip()

        if not query:
            print("[ERROR] Query cannot be empty.")
            sys.exit(1)

        # Step 4: Execute query
        response = answer_query(
            client=client,
            store=store,
            query=query,
            top_k=args.top_k,
            min_score=args.min_score,
            max_score_drop=args.max_drop,
            document_id=args.document_id,
            model_name=args.model,
        )

        # Step 5: Display
        print_response(query=query, response=response)

    except KeyboardInterrupt:
        print("\n\n[INFO] Operation stopped by user.")
        sys.exit(0)
    except Exception as exc:
        print("\n[ERROR] Pipeline failed:")
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()