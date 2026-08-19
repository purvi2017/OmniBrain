"""
test_rag_retrieval_accuracy.py
------------------------------
Day 8 Test Suite: RAG Retrieval Accuracy, Multi-Document FAISS Pipeline,
Threshold Calibration, and Source Attribution Verification.

Verifies:
1. Ingestion and indexing of multiple PDF documents into a shared FAISS store.
2. Cross-document retrieval accuracy and target document isolation.
3. Similarity threshold tuning and noise suppression (min_score & max_score_drop).
4. Page-level citation and document metadata tracking.
5. Robust out-of-domain and low-relevance query rejection.
6. Document-specific constrained querying (document_id filtering).
7. End-to-end multi-document RAG question answering.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from vector_db.faiss_store import FaissStore
from rag_llm import (
    process_pdf,
    build_store,
    retrieve_context,
    prepare_context,
    answer_query,
    RAGLLMPipeline,
    NO_CONTEXT_MESSAGE,
    DEFAULT_MIN_SCORE,
    DEFAULT_MAX_SCORE_DROP,
)

SAMPLE_PDF_1 = str(BASE_DIR / "test_files" / "sample.pdf")
SAMPLE_PDF_2 = str(BASE_DIR / "test_files" / "ai_architecture.pdf")


def test_multi_document_store_ingestion():
    """Verify multiple PDFs are correctly extracted, chunked, and indexed with page metadata."""
    print("=" * 60)
    print("TEST: Multi-Document Ingestion & FAISS Indexing")
    print("=" * 60)

    assert os.path.exists(SAMPLE_PDF_1), f"Missing test PDF: {SAMPLE_PDF_1}"
    assert os.path.exists(SAMPLE_PDF_2), f"Missing test PDF: {SAMPLE_PDF_2}"

    store, metadata = build_store([SAMPLE_PDF_1, SAMPLE_PDF_2], chunk_size=500, overlap=50)

    assert len(metadata) == 2, f"Expected 2 document records, got {len(metadata)}"
    doc_ids = [m["document_id"] for m in metadata]
    assert "sample" in doc_ids
    assert "ai_architecture" in doc_ids

    total_chunks = sum(m["chunk_count"] for m in metadata)
    assert store.count() == total_chunks, "Store vector count mismatch"

    # Verify unique document IDs in FAISS store
    indexed_doc_ids = store.get_document_ids()
    assert set(indexed_doc_ids) == {"sample", "ai_architecture"}
    print(f"[PASS] Successfully indexed {len(metadata)} documents with {store.count()} total vectors")


def test_cross_document_retrieval_accuracy():
    """Verify queries targeting specific documents retrieve the correct source chunks."""
    print("=" * 60)
    print("TEST: Cross-Document Retrieval & Source Accuracy")
    print("=" * 60)

    store, _ = build_store([SAMPLE_PDF_1, SAMPLE_PDF_2], chunk_size=500, overlap=50)

    # Test Query 1 -> Targets Document 1 (sample.pdf)
    q1 = "What library is used for text extraction in the sample PDF?"
    r1 = retrieve_context(store, query=q1, top_k=2, min_score=0.30)
    assert len(r1) > 0, "Failed to retrieve results for Query 1"
    assert r1[0]["filename"] == "sample.pdf", f"Expected sample.pdf, got {r1[0]['filename']}"
    assert "pypdf" in r1[0]["text"]
    print(f"[PASS] Query 1 correctly retrieved top match from sample.pdf (Score: {r1[0]['score']:.4f})")

    # Test Query 2 -> Targets Document 2 (ai_architecture.pdf, Page 1)
    q2 = "How does the LLM Orchestrator communicate with Google Gemini?"
    r2 = retrieve_context(store, query=q2, top_k=2, min_score=0.30)
    assert len(r2) > 0, "Failed to retrieve results for Query 2"
    assert r2[0]["filename"] == "ai_architecture.pdf", f"Expected ai_architecture.pdf, got {r2[0]['filename']}"
    assert r2[0]["page"] == 1, f"Expected Page 1 citation, got Page {r2[0]['page']}"
    assert "LLM Orchestrator" in r2[0]["text"]
    print(f"[PASS] Query 2 correctly retrieved top match from ai_architecture.pdf Page 1 (Score: {r2[0]['score']:.4f})")

    # Test Query 3 -> Targets Document 2 (ai_architecture.pdf, Page 2)
    q3 = "What are the three evaluation metrics for retrieval precision such as MRR and Hit Rate?"
    r3 = retrieve_context(store, query=q3, top_k=2, min_score=0.30)
    assert len(r3) > 0, "Failed to retrieve results for Query 3"
    assert r3[0]["filename"] == "ai_architecture.pdf"
    assert r3[0]["page"] == 2, f"Expected Page 2 citation, got Page {r3[0]['page']}"
    assert "Mean Reciprocal Rank" in r3[0]["text"] or "MRR" in r3[0]["text"]
    print(f"[PASS] Query 3 correctly retrieved top match from ai_architecture.pdf Page 2 (Score: {r3[0]['score']:.4f})")


def test_similarity_threshold_tuning_and_noise_filtering():
    """Verify adaptive thresholding and relative score delta trimming."""
    print("=" * 60)
    print("TEST: Similarity Threshold Tuning & Noise Filtering")
    print("=" * 60)

    store, _ = build_store([SAMPLE_PDF_1, SAMPLE_PDF_2], chunk_size=500, overlap=50)

    query = "Explain the microservices architecture and event queues."

    # Search with wide top_k
    results = retrieve_context(
        store,
        query=query,
        top_k=5,
        min_score=DEFAULT_MIN_SCORE,
        max_score_drop=DEFAULT_MAX_SCORE_DROP,
    )

    assert len(results) > 0, "Expected relevant results"
    top_score = results[0]["score"]

    for r in results:
        # Verify score is above calibrated minimum
        assert r["score"] >= DEFAULT_MIN_SCORE
        # Verify score is within max allowable drop from top score
        assert (top_score - r["score"]) <= DEFAULT_MAX_SCORE_DROP
        # Verify confidence tag exists
        assert r["confidence"] in ["HIGH", "MEDIUM", "LOW"]

    print(f"[PASS] Adaptive threshold retained {len(results)} high-confidence candidate(s) (Top Score: {top_score:.4f})")


def test_multiple_out_of_domain_queries():
    """Verify handling and rejection of various unanswerable queries."""
    print("=" * 60)
    print("TEST: Out-of-Domain & Low-Relevance Query Rejection")
    print("=" * 60)

    store, _ = build_store([SAMPLE_PDF_1, SAMPLE_PDF_2], chunk_size=500, overlap=50)

    unrelated_queries = [
        "What is the average surface temperature of Venus?",
        "How do you prepare authentic Neapolitan pizza dough?",
        "Who won the Wimbledon gentlemen's singles tournament in 2012?",
    ]

    for q in unrelated_queries:
        response = answer_query(
            client=lambda prompt: "Should never execute",
            store=store,
            query=q,
            top_k=3,
            min_score=0.40,
            max_score_drop=0.20,
        )

        assert response["found"] is False, f"Expected found=False for out-of-domain query: {q}"
        assert response["answer"] == NO_CONTEXT_MESSAGE
        assert len(response["sources"]) == 0
        assert response["retrieved_chunks"] == 0
        print(f"[PASS] Successfully rejected out-of-domain query: '{q[:40]}...'")


def test_document_id_constrained_filtering():
    """Verify that document_id parameter strictly limits retrieval to the chosen document."""
    print("=" * 60)
    print("TEST: Document-ID Constrained Filtering")
    print("=" * 60)

    store, _ = build_store([SAMPLE_PDF_1, SAMPLE_PDF_2], chunk_size=500, overlap=50)

    # Ask a question that could match vector search, but restrict to 'sample'
    query = "How is data vectorized and stored in a database?"
    results = retrieve_context(
        store,
        query=query,
        top_k=3,
        min_score=0.20,
        document_id="sample",
    )

    assert len(results) > 0
    for r in results:
        assert r["document_id"] == "sample", f"Expected document_id 'sample', got '{r['document_id']}'"

    print(f"[PASS] All {len(results)} retrieved chunks strictly belonged to document_id='sample'")


def test_end_to_end_multi_doc_answer_generation():
    """Verify end-to-end answer generation across multiple documents with structured citations."""
    print("=" * 60)
    print("TEST: End-to-End Multi-Document RAG Answer Generation")
    print("=" * 60)

    store, _ = build_store([SAMPLE_PDF_1, SAMPLE_PDF_2], chunk_size=500, overlap=50)
    query = "What are the core microservices in the OmniBrain AI architecture?"

    def mock_gemini(prompt: str) -> str:
        assert "ai_architecture.pdf" in prompt
        assert "Document Parser Service" in prompt or "Microservices" in prompt
        return (
            "According to ai_architecture.pdf (Page 1), the core microservices include "
            "the Document Parser Service, the Embedding Generation Service, the Vector Storage Engine, "
            "and the LLM Orchestrator."
        )

    response = answer_query(
        client=mock_gemini,
        store=store,
        query=query,
        top_k=3,
        min_score=0.30,
        model_name="gemini-2.5-flash",
    )

    assert response["found"] is True
    assert response["retrieved_chunks"] > 0
    assert "Document Parser Service" in response["answer"]
    assert response["sources"][0]["filename"] == "ai_architecture.pdf"
    assert response["sources"][0]["page"] == 1
    assert "confidence" in response["sources"][0]

    print("[PASS] Multi-document RAG response generated successfully with full citations:")
    print(f"       Answer : {response['answer']}")
    print(f"       Source : {response['sources'][0]['filename']} (Page {response['sources'][0]['page']})")
    print(f"       Score  : {response['sources'][0]['score']} ({response['sources'][0]['confidence']})")


def run_all_accuracy_tests():
    """Run complete Day 8 retrieval accuracy test suite."""
    print("=" * 70)
    print("RUNNING DAY 8 RETRIEVAL ACCURACY & MULTI-DOC TEST SUITE")
    print("=" * 70)

    test_multi_document_store_ingestion()
    test_cross_document_retrieval_accuracy()
    test_similarity_threshold_tuning_and_noise_filtering()
    test_multiple_out_of_domain_queries()
    test_document_id_constrained_filtering()
    test_end_to_end_multi_doc_answer_generation()

    print("\n" + "=" * 70)
    print("RESULT: ALL DAY 8 RETRIEVAL ACCURACY TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_all_accuracy_tests()
