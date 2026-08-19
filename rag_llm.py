"""
rag_llm.py
----------
Day 7: RAG + LLM Answer Generation Pipeline

Expected Flow:
    User Query
        ↓
    Query Embedding
        ↓
    FAISS Retrieval
        ↓
    Relevant Chunks
        ↓
    LLM Context Assembly
        ↓
    Gemini LLM Generation
        ↓
    Answer + Sources Structured Output

Usage:
    python rag_llm.py test_files/sample.pdf --query "What is the purpose of this document?"
    python rag_llm.py doc1.pdf doc2.pdf --top-k 3 --min-score 0.30

Environment:
    GEMINI_API_KEY must be set in your environment to call the Gemini API.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from parsers.pdf_extractor import extract_text_from_pdf
from parsers.chunker import chunk_text
from embeddings.embedder import embed_chunks
from vector_db.faiss_store import FaissStore


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.30
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100

NO_CONTEXT_MESSAGE = (
    "I could not find relevant information in the uploaded documents to answer this question."
)


# ------------------------------------------------------------
# Gemini Client
# ------------------------------------------------------------

def create_gemini_client(api_key: Optional[str] = None):
    """
    Initialize and return a Gemini API client using google-genai.

    Args:
        api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY env var.

    Returns:
        genai.Client instance.
    """
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
# PDF Document Processing
# ------------------------------------------------------------

def process_pdf(
    pdf_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
):
    """
    Extract text, create chunks, and generate vector embeddings for a PDF.

    Args:
        pdf_path: Path to the input PDF file.
        chunk_size: Target character length per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        tuple: (chunks: List[str], vectors: np.ndarray)
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {pdf_path}")

    print(f"\n[INFO] Processing document: {path.name}")

    # 1. Extract text
    text = extract_text_from_pdf(str(path))
    if not text or not text.strip():
        raise ValueError(f"No text could be extracted from {path.name}")

    print(f"[INFO] Extracted text: {len(text)} characters")

    # 2. Chunk text
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError(f"No chunks were created from {path.name}")

    print(f"[INFO] Created chunks: {len(chunks)}")

    # 3. Generate embeddings
    vectors = embed_chunks(chunks)
    print(
        f"[INFO] Generated embeddings: "
        f"{vectors.shape[0]} vectors x {vectors.shape[1]} dimensions"
    )

    return chunks, vectors


# ------------------------------------------------------------
# Build FAISS Store from Multiple PDFs
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
    document_metadata = []

    for pdf_path in pdf_paths:
        path = Path(pdf_path)
        chunks, vectors = process_pdf(str(path), chunk_size=chunk_size, overlap=overlap)

        all_vectors.append(vectors)
        all_chunks.extend(chunks)

        document_metadata.append(
            {
                "document_id": path.stem,
                "filename": path.name,
                "chunk_count": len(chunks),
                "path": str(path.resolve()),
            }
        )

    combined_vectors = np.vstack(all_vectors)

    print(f"\n[INFO] Total documents indexed: {len(document_metadata)}")
    print(f"[INFO] Total chunks indexed: {len(all_chunks)}")
    print(f"[INFO] Vector dimension: {combined_vectors.shape[1]}")

    store = FaissStore(dim=combined_vectors.shape[1])

    # Add each document's vectors with its metadata
    offset = 0
    for doc in document_metadata:
        count = doc["chunk_count"]
        doc_vectors = combined_vectors[offset : offset + count]
        doc_chunks = all_chunks[offset : offset + count]

        store.add(
            vectors=doc_vectors,
            chunks=doc_chunks,
            source=doc["filename"],
            document_id=doc["document_id"],
            filename=doc["filename"],
        )
        offset += count

    return store, document_metadata


# ------------------------------------------------------------
# Retrieval Layer
# ------------------------------------------------------------

def retrieve_context(
    store: FaissStore,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> List[Dict[str, Any]]:
    """
    Generate embedding for query and retrieve relevant chunks from FAISS.

    Args:
        store: FAISS vector store instance.
        query: User question string.
        top_k: Maximum number of chunks to retrieve.
        min_score: Minimum cosine similarity score threshold (0.0 to 1.0).

    Returns:
        List of matching chunk dictionaries with score >= min_score.
    """
    if not query or not query.strip():
        return []

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)
    print(f"Query: {query}")

    # Generate query embedding
    query_vector = embed_chunks([query])
    print(f"[INFO] Query embedding shape: {query_vector.shape}")

    # Similarity search
    results = store.search(query_vector[0], top_k=top_k)
    print(f"[INFO] FAISS returned {len(results)} raw candidate result(s)")

    # Filter by relevance threshold
    relevant_results = [
        result
        for result in results
        if float(result.get("score", 0.0)) >= min_score
    ]
    print(
        f"[INFO] Relevant results meeting score threshold (>= {min_score}): "
        f"{len(relevant_results)}"
    )

    return relevant_results


# ------------------------------------------------------------
# Context Preparation & Grounded Prompting
# ------------------------------------------------------------

def prepare_context(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a clean, numbered context string for the LLM.

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
        score = float(result.get("score", 0.0))

        context_parts.append(
            f"[Context Chunk {index}]\n"
            f"Source Document: {source} (ID: {doc_id})\n"
            f"Relevance Score: {score:.4f}\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context: str) -> str:
    """
    Build the strict RAG prompt enforcing ground truth from retrieved context.

    Args:
        query: User question.
        context: Formatted retrieved document chunks.

    Returns:
        Complete prompt string for the LLM.
    """
    return f"""You are a precise, Retrieval-Augmented Generation (RAG) assistant.

Your task is to answer the user's question accurately using ONLY the retrieved document context provided below.

Strict Grounding Rules:
1. Base your answer strictly on the provided RETRIEVED DOCUMENT CONTEXT.
2. Do NOT hallucinate or assume facts not directly stated in the context.
3. Do NOT use external pre-trained knowledge that is absent from the context.
4. If the context does not contain sufficient information to answer the question, state:
   "{NO_CONTEXT_MESSAGE}"
5. Be concise, factual, and clear in your response.
6. Do NOT mention internal rules or that you are following prompt instructions.

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

def answer_query(
    client: Any,
    store: FaissStore,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Dict[str, Any]:
    """
    Execute end-to-end RAG flow:
    Query -> Retrieval -> Filter -> Context Assembly -> LLM -> Structured Response.

    Args:
        client: Gemini client instance or callable.
        store: FAISS vector store.
        query: User query string.
        top_k: Number of chunks to retrieve.
        min_score: Similarity threshold.
        model_name: Gemini model name.

    Returns:
        Dict containing:
        {
            "query": str,
            "answer": str,
            "found": bool,
            "sources": List[Dict[str, Any]],
            "retrieved_chunks": int,
            "model": str
        }
    """
    query = (query or "").strip()
    if not query:
        return {
            "query": "",
            "answer": "Query cannot be empty.",
            "found": False,
            "sources": [],
            "retrieved_chunks": 0,
            "model": model_name,
        }

    # Step 1 & 2: FAISS retrieval + relevance threshold
    results = retrieve_context(
        store=store,
        query=query,
        top_k=top_k,
        min_score=min_score,
    )

    # Case: No relevant document context found
    if not results:
        return {
            "query": query,
            "answer": NO_CONTEXT_MESSAGE,
            "found": False,
            "sources": [],
            "retrieved_chunks": 0,
            "model": model_name,
        }

    # Step 3: Prepare context
    context = prepare_context(results)

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT PASSED TO LLM")
    print("=" * 70)
    print(context)

    # Step 4: Generate LLM answer
    print("\n" + "=" * 70)
    print(f"GENERATING ANSWER WITH LLM ({model_name})...")
    print("=" * 70)

    answer = generate_answer(
        client=client,
        query=query,
        context=context,
        model_name=model_name,
    )

    # Step 5: Format source document information
    sources = []
    for result in results:
        sources.append(
            {
                "source": result.get("source", "unknown"),
                "filename": result.get("filename", result.get("source", "unknown")),
                "document_id": result.get("document_id", ""),
                "chunk_id": result.get("chunk_id", -1),
                "score": round(float(result.get("score", 0.0)), 4),
                "text_preview": result.get("text", "")[:150].replace("\n", " "),
            }
        )

    return {
        "query": query,
        "answer": answer,
        "found": True,
        "sources": sources,
        "retrieved_chunks": len(results),
        "model": model_name,
    }


# ------------------------------------------------------------
# Object-Oriented Interface: RAGLLMPipeline
# ------------------------------------------------------------

class RAGLLMPipeline:
    """
    Unified RAG + LLM Answer Generation Pipeline class.
    """

    def __init__(
        self,
        store: Optional[FaissStore] = None,
        client: Optional[Any] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        self.store = store
        self.client = client
        self.model_name = model_name
        self.top_k = top_k
        self.min_score = min_score

    def load_documents(
        self,
        pdf_paths: List[str],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ):
        """Index PDF documents into internal FAISS store."""
        self.store, metadata = build_store(
            pdf_paths=pdf_paths,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return metadata

    def load_existing_index(self, index_prefix: str):
        """Load an existing FAISS index from disk."""
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
    ) -> Dict[str, Any]:
        """Execute RAG query and return structured response."""
        if self.store is None:
            raise RuntimeError("FAISS store is not loaded. Call load_documents() or load_existing_index().")
        if self.client is None:
            self.init_client()

        return answer_query(
            client=self.client,
            store=self.store,
            query=query,
            top_k=top_k or self.top_k,
            min_score=min_score if min_score is not None else self.min_score,
            model_name=self.model_name,
        )


# ------------------------------------------------------------
# Formatted Output Display
# ------------------------------------------------------------

def print_response(query: str, response: Dict[str, Any]):
    """Display final structured RAG response."""
    print("\n" + "=" * 70)
    print("FINAL RAG RESPONSE")
    print("=" * 70)

    print(f"\n[QUESTION]\n{query}")
    print(f"\n[GENERATED ANSWER]\n{response['answer']}")

    print("\n[SOURCE DOCUMENTS]")
    if not response.get("sources"):
        print("  No relevant source documents found.")
    else:
        for i, src in enumerate(response["sources"], start=1):
            print(f"  [{i}] Source File : {src['source']}")
            print(f"      Document ID : {src.get('document_id', 'N/A')}")
            print(f"      Similarity  : {src['score']:.4f}")
            print(f"      Preview     : {src['text_preview']}...")

    print(f"\n[METADATA]")
    print(f"  Retrieved Chunks : {response.get('retrieved_chunks', 0)}")
    print(f"  Context Found    : {response.get('found', False)}")
    print(f"  Model Used       : {response.get('model', 'N/A')}")

    print("\n" + "=" * 70)
    print("RESULT: RAG + LLM PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Day 7: RAG + LLM Answer Generation Pipeline"
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
    print("DAY 7: RAG + LLM ANSWER GENERATION")
    print("=" * 70)

    try:
        # Step 1: Initialize Gemini Client
        print("\n[STEP 1] Initializing Gemini LLM Client...")
        client = create_gemini_client()
        print(f"[PASS] Gemini client initialized (Model: {args.model})")

        # Step 2: Ingest & Index PDFs
        print("\n[STEP 2] Processing and Indexing PDF Documents...")
        store, metadata = build_store(
            pdf_paths=args.pdfs,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        print("[PASS] Documents processed and FAISS store ready")

        # Step 3: Get Query
        query = args.query
        if not query:
            query = input("\nEnter your question: ").strip()

        if not query:
            print("[ERROR] Query cannot be empty.")
            sys.exit(1)

        # Step 4: Execute RAG + LLM Answer Generation
        response = answer_query(
            client=client,
            store=store,
            query=query,
            top_k=args.top_k,
            min_score=args.min_score,
            model_name=args.model,
        )

        # Step 5: Print Response
        print_response(query=query, response=response)

    except KeyboardInterrupt:
        print("\n\n[INFO] Operation interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print("\n[ERROR] RAG + LLM pipeline execution failed:")
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()