"""
rag_response_generator.py
--------------------------
Day 16: High-Level AI Response Generation Pipeline for Backend Integration.

Connects query requests with document ingestion, context retrieval, and grounded
LLM answer generation. Formats the final AI response into standardized backend-ready
payload structures.

Key Components:
    ResponseGenerator: Object-oriented interface for indexing documents, retrieving
                       context, and producing backend-ready AI responses.
    generate_ai_response(): Function interface for single-call execution.

Usage:
    from rag_response_generator import ResponseGenerator, generate_ai_response

    generator = ResponseGenerator(client=llm_client)
    generator.load_documents(["test_files/market_analysis_2026.pdf"])
    response = generator.generate_response(query="What was CyberPulse revenue in Q3 2026?")
"""

import os
import argparse
from typing import Any, Dict, List, Optional, Union

from rag_llm import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_MIN_SCORE,
    DEFAULT_MAX_SCORE_DROP,
    NO_CONTEXT_MESSAGE,
    build_store,
    create_gemini_client,
    process_backend_query,
    answer_query,
    retrieve_context,
    prepare_context,
    generate_answer,
)
from vector_db.faiss_store import FaissStore
from schemas import BackendQueryInput, BackendQueryOutput, SourceAttribution


class ResponseGenerator:
    """
    Unified AI Response Generation Engine.

    Encapsulates document context retrieval, grounded LLM answer generation,
    and backend payload formatting into a single cohesive service.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        store: Optional[FaissStore] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        max_score_drop: float = DEFAULT_MAX_SCORE_DROP,
    ):
        """
        Initialize ResponseGenerator with optional LLM client and FAISS vector store.
        """
        self.client = client
        self.store = store
        self.model_name = model_name
        self.top_k = top_k
        self.min_score = min_score
        self.max_score_drop = max_score_drop
        self.ingested_documents: List[Dict] = []

    def init_llm(self, api_key: Optional[str] = None):
        """Initialize Gemini client if not already provided."""
        self.client = create_gemini_client(api_key=api_key)
        return self.client

    def load_documents(
        self,
        pdf_paths: List[str],
        chunk_size: int = 800,
        overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Process PDF documents, create text chunks, generate embeddings,
        and populate the internal FAISS store.
        """
        store, metadata = build_store(
            pdf_paths=pdf_paths,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        self.store = store
        self.ingested_documents = metadata
        return metadata

    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        document_id: Optional[str] = None,
        source_document: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        max_score_drop: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end AI response generation flow:
        1. Process query & retrieve relevant document context (or use direct context).
        2. Generate grounded LLM answer.
        3. Format response in backend-ready structure containing:
           - answer
           - source_document
           - document_id
           - relevant_context
           - found
           - sources
           - retrieved_chunks
           - model
        """
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_min_score = min_score if min_score is not None else self.min_score
        effective_max_drop = max_score_drop if max_score_drop is not None else self.max_score_drop

        return process_backend_query(
            client=self.client,
            query=query,
            context=context,
            store=self.store,
            document_id=document_id,
            source_document=source_document,
            top_k=effective_top_k,
            min_score=effective_min_score,
            max_score_drop=effective_max_drop,
            model_name=self.model_name,
        )

    def generate_typed_response(
        self,
        query_input: Union[BackendQueryInput, Dict[str, Any]],
    ) -> BackendQueryOutput:
        """
        Generate AI response accepting BackendQueryInput Pydantic model
        and returning BackendQueryOutput Pydantic model.
        """
        if isinstance(query_input, dict):
            inp = BackendQueryInput(**query_input)
        else:
            inp = query_input

        raw_res = self.generate_response(
            query=inp.query,
            context=inp.context,
            document_id=inp.document_id,
            source_document=inp.source_document,
            top_k=inp.top_k,
            min_score=inp.min_score,
            max_score_drop=inp.max_score_drop,
        )

        sources = [SourceAttribution.from_dict(s) for s in raw_res.get("sources", [])]

        return BackendQueryOutput(
            query=raw_res["query"],
            answer=raw_res["answer"],
            source_document=raw_res["source_document"],
            document_id=raw_res["document_id"],
            relevant_context=raw_res["relevant_context"],
            found=raw_res["found"],
            sources=sources,
            retrieved_chunks=raw_res["retrieved_chunks"],
            model=raw_res["model"],
        )


def generate_ai_response(
    client: Any,
    query: str,
    context: Optional[str] = None,
    store: Optional[FaissStore] = None,
    document_id: Optional[str] = None,
    source_document: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Dict[str, Any]:
    """
    Function interface for generating grounded AI responses for backend consumption.
    """
    generator = ResponseGenerator(
        client=client,
        store=store,
        model_name=model_name,
        top_k=top_k,
        min_score=min_score,
    )
    return generator.generate_response(
        query=query,
        context=context,
        document_id=document_id,
        source_document=source_document,
    )


def print_formatted_response(response: Dict[str, Any]):
    """Pretty print AI response dictionary for CLI testing."""
    print("\n" + "=" * 70)
    print("AI RESPONSE GENERATION OUTPUT")
    print("=" * 70)
    print(f"Query           : {response['query']}")
    print(f"Found           : {response['found']}")
    print(f"Source Document : {response['source_document']}")
    print(f"Document ID     : {response['document_id']}")
    print(f"Model           : {response['model']}")
    print(f"Chunks Used     : {response['retrieved_chunks']}")
    print("\n--- GENERATED ANSWER ---")
    print(response["answer"])
    if response["relevant_context"]:
        print("\n--- RELEVANT CONTEXT ---")
        print(response["relevant_context"][:400] + ("..." if len(response["relevant_context"]) > 400 else ""))
    print("=" * 70 + "\n")


def main():
    """CLI runner to test AI response generation with sample document-based questions."""
    parser = argparse.ArgumentParser(
        description="Day 16: Test AI Response Generation from Document Content"
    )
    parser.add_argument("pdfs", nargs="*", help="Path(s) to PDF document files")
    parser.add_argument("--query", "-q", type=str, default=None, help="Question string to answer")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_GEMINI_MODEL, help="LLM model name")

    args = parser.parse_args()

    print("=" * 70)
    print("DAY 16: AI RESPONSE GENERATION DEMO")
    print("=" * 70)

    generator = ResponseGenerator(model_name=args.model)

    if args.pdfs:
        print(f"\n[STEP 1] Ingesting PDF Document(s): {args.pdfs}")
        generator.load_documents(args.pdfs)
        print(f"[PASS] Successfully indexed {len(args.pdfs)} document(s).")
    else:
        sample_pdf = os.path.join("test_files", "market_analysis_2026.pdf")
        if os.path.exists(sample_pdf):
            print(f"\n[STEP 1] Ingesting default sample PDF: {sample_pdf}")
            generator.load_documents([sample_pdf])

    query = args.query or "What was CyberPulse Systems revenue in Q3 2026?"
    print(f"\n[STEP 2] Generating response for query: {query!r}")

    # Mock callable if API key is not present
    client = None
    if os.getenv("GEMINI_API_KEY"):
        generator.init_llm()
        client = generator.client
    else:
        def mock_cli_llm(prompt: str) -> str:
            return "CyberPulse Systems recorded revenue of $42.5 Million in Q3 2026."
        generator.client = mock_cli_llm

    res = generator.generate_response(query=query)
    print_formatted_response(res)


if __name__ == "__main__":
    main()
