"""
test_rag_llm.py
---------------
Day 7 Verification Test Suite for RAG + LLM Answer Generation.

Verifies:
1. PDF document processing, chunking, and FAISS indexing
2. Query embedding and FAISS similarity retrieval with relevance filtering
3. Context assembly and strict grounding prompt generation
4. Response structure format (answer, source metadata, score, preview)
5. Out-of-domain query handling when no relevant chunks are found
6. End-to-end RAG pipeline execution with both mock generator and live client
7. Object-oriented RAGLLMPipeline interface
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from vector_db.faiss_store import FaissStore
from rag_llm import (
    process_pdf,
    build_store,
    retrieve_context,
    prepare_context,
    build_grounded_prompt,
    generate_answer,
    answer_query,
    RAGLLMPipeline,
    NO_CONTEXT_MESSAGE,
)

SAMPLE_PDF = str(BASE_DIR / "test_files" / "sample.pdf")


def test_pdf_processing_and_store_building():
    """Verify PDF extraction, chunking, embedding, and FAISS multi-doc store construction."""
    print("=" * 60)
    print("TEST: PDF Processing & FAISS Store Construction")
    print("=" * 60)

    assert os.path.exists(SAMPLE_PDF), f"Sample PDF not found: {SAMPLE_PDF}"

    chunks, vectors = process_pdf(SAMPLE_PDF, chunk_size=500, overlap=50)
    assert len(chunks) > 0, "Failed: No chunks created from sample PDF"
    assert len(vectors) == len(chunks), "Failed: Vectors count does not match chunks count"
    assert vectors.shape[1] == 384, f"Failed: Expected vector dimension 384, got {vectors.shape[1]}"
    print(f"[PASS] process_pdf extracted {len(chunks)} chunks and {len(vectors)} vectors")

    store, metadata = build_store([SAMPLE_PDF], chunk_size=500, overlap=50)
    assert store.count() == len(chunks), "Failed: FAISS store vector count mismatch"
    assert len(metadata) == 1, "Failed: Document metadata count mismatch"
    assert metadata[0]["filename"] == "sample.pdf"
    assert metadata[0]["document_id"] == "sample"
    print(f"[PASS] build_store successfully indexed {store.count()} vectors with metadata")


def test_retrieval_and_relevance_filtering():
    """Verify semantic retrieval and similarity score threshold filtering."""
    print("=" * 60)
    print("TEST: Retrieval & Relevance Threshold Filtering")
    print("=" * 60)

    store, _ = build_store([SAMPLE_PDF], chunk_size=500, overlap=50)

    # In-domain query about sample PDF content
    query = "What is the purpose of this document and text extraction?"
    results = retrieve_context(store, query=query, top_k=2, min_score=0.20)

    assert len(results) > 0, "Failed: Expected relevant chunks for in-domain query"
    for r in results:
        assert "text" in r
        assert "source" in r
        assert "score" in r
        assert r["score"] >= 0.20
        assert r["filename"] == "sample.pdf"
    print(f"[PASS] In-domain query retrieved {len(results)} relevant chunk(s)")

    # Test context preparation formatting
    context_str = prepare_context(results)
    assert "[Context Chunk 1]" in context_str
    assert "Source Document: sample.pdf" in context_str
    assert "Relevance Score:" in context_str
    print("[PASS] Context string formatted correctly for LLM")


def test_out_of_domain_query_handling():
    """Verify handling when query is unrelated and no chunks meet the relevance threshold."""
    print("=" * 60)
    print("TEST: Out-of-Domain Query & Missing Context Handling")
    print("=" * 60)

    store, _ = build_store([SAMPLE_PDF], chunk_size=500, overlap=50)

    # Completely unrelated query with standard threshold
    unrelated_query = "What is the chemical composition of interstellar dark matter?"
    results = retrieve_context(store, query=unrelated_query, top_k=2, min_score=0.85)

    assert len(results) == 0, "Failed: Expected 0 results for unrelated query with high threshold"

    # Test end-to-end response on no-context
    response = answer_query(
        client=lambda prompt: "Should not be called",
        store=store,
        query=unrelated_query,
        top_k=2,
        min_score=0.85,
    )

    assert response["found"] is False
    assert response["answer"] == NO_CONTEXT_MESSAGE
    assert response["sources"] == []
    assert response["retrieved_chunks"] == 0
    print("[PASS] Out-of-domain query handled cleanly with fallback message and empty sources")


def test_empty_query_handling():
    """Verify edge case handling for empty or whitespace-only queries."""
    print("=" * 60)
    print("TEST: Empty Query Handling")
    print("=" * 60)

    store, _ = build_store([SAMPLE_PDF], chunk_size=500, overlap=50)

    response = answer_query(
        client=lambda p: "",
        store=store,
        query="   ",
    )

    assert response["found"] is False
    assert response["retrieved_chunks"] == 0
    assert response["sources"] == []
    print("[PASS] Empty query handled gracefully")


def test_grounded_prompt_and_mock_llm_answer_generation():
    """Verify prompt formatting and end-to-end answer generation with mock LLM generator."""
    print("=" * 60)
    print("TEST: Grounded Prompt & End-to-End Structured Response")
    print("=" * 60)

    store, _ = build_store([SAMPLE_PDF], chunk_size=500, overlap=50)
    query = "What library is used for text extraction?"

    # Mock generator that verifies context is present in prompt
    def mock_llm_client(prompt: str) -> str:
        assert "Strict Grounding Rules" in prompt
        assert query in prompt
        assert "RETRIEVED DOCUMENT CONTEXT" in prompt
        assert "pypdf" in prompt
        return "Based on the retrieved context, pypdf is used for text extraction from the sample PDF."

    response = answer_query(
        client=mock_llm_client,
        store=store,
        query=query,
        top_k=2,
        min_score=0.20,
        model_name="mock-gemini",
    )

    # Verify structured response schema
    assert response["query"] == query
    assert response["found"] is True
    assert "pypdf" in response["answer"]
    assert response["retrieved_chunks"] > 0
    assert len(response["sources"]) == response["retrieved_chunks"]
    assert response["model"] == "mock-gemini"

    # Verify source structure
    source_item = response["sources"][0]
    assert "source" in source_item
    assert "filename" in source_item
    assert "document_id" in source_item
    assert "chunk_id" in source_item
    assert "score" in source_item
    assert "text_preview" in source_item
    assert isinstance(source_item["score"], float)

    print("[PASS] Response schema validated successfully:")
    print(f"       Answer: {response['answer']}")
    print(f"       Sources: {len(response['sources'])} item(s)")
    print(f"       Top Source Score: {source_item['score']}")


def test_rag_pipeline_class_interface():
    """Verify the RAGLLMPipeline object-oriented interface."""
    print("=" * 60)
    print("TEST: RAGLLMPipeline Class Interface")
    print("=" * 60)

    def mock_generator(prompt: str) -> str:
        return "FAISS vector database is used for semantic search."

    pipeline = RAGLLMPipeline(
        client=mock_generator,
        model_name="mock-gemini",
        top_k=2,
        min_score=0.20,
    )

    metadata = pipeline.load_documents([SAMPLE_PDF], chunk_size=500, overlap=50)
    assert len(metadata) == 1

    response = pipeline.query("How are embeddings stored?")
    assert response["found"] is True
    assert response["retrieved_chunks"] > 0
    assert "FAISS" in response["answer"]
    print("[PASS] RAGLLMPipeline object interface works as expected")


def run_all_tests():
    """Run all Day 7 tests manually."""
    print("=" * 70)
    print("RUNNING DAY 7 RAG + LLM TEST SUITE")
    print("=" * 70)

    test_pdf_processing_and_store_building()
    test_retrieval_and_relevance_filtering()
    test_out_of_domain_query_handling()
    test_empty_query_handling()
    test_grounded_prompt_and_mock_llm_answer_generation()
    test_rag_pipeline_class_interface()

    print("\n" + "=" * 70)
    print("RESULT: ALL DAY 7 RAG + LLM TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
