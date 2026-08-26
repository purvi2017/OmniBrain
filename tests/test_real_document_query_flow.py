"""
test_real_document_query_flow.py
---------------------------------
Day 14: Real Document Query Flow & Backend Integration Test Suite.

Verifies:
1. Handling of query and document metadata received from Backend.
2. End-to-end RAG pipeline retrieval of relevant chunks from actual PDF documents.
3. Grounded answer generation via LLM using real document content.
4. Validation of mandatory structured response fields:
   - answer
   - source_document
   - document_id
   - relevant_context
5. Verification of precise page-level and source document attributions.
6. Evaluation of relevant-document vs no-relevant-context query scenarios.
7. Robust error and fallback handling across Backend ↔ AI Module communication.
8. REST API /query endpoint verification for real document queries.
"""

import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag_llm import (
    process_backend_query,
    build_store,
    answer_query,
    NO_CONTEXT_MESSAGE,
    RAGLLMPipeline,
)
from schemas import BackendQueryInput, BackendQueryOutput, SourceAttribution
from api import app, get_state

# Real test PDFs
SAMPLE_PDF = str(BASE_DIR / "test_files" / "sample.pdf")
AI_PDF = str(BASE_DIR / "test_files" / "ai_architecture.pdf")
MARKET_PDF = str(BASE_DIR / "test_files" / "market_analysis_2026.pdf")
MULTIPAGE_PDF = str(BASE_DIR / "test_files" / "multipage_manual.pdf")


def mock_real_doc_llm(prompt: str) -> str:
    """Deterministic mock responder for real document query testing."""
    prompt_lower = prompt.lower()

    if "cyberpulse" in prompt_lower or "42.5" in prompt_lower or "revenue" in prompt_lower:
        return "CyberPulse Systems recorded revenue of $42.5 Million in Q3 2026 with a net margin of 24%."
    if "latency" in prompt_lower or "11.2ms" in prompt_lower:
        return "Hybrid FAISS Indexing reduced query latency from 85ms to 11.2ms."
    if "omnibrain" in prompt_lower or "microservice" in prompt_lower or "llm orchestrator" in prompt_lower:
        return "The OmniBrain architecture contains Document Parser Service and LLM Orchestrator."
    if "pypdf" in prompt_lower or "extraction" in prompt_lower:
        return "pypdf is used for text extraction."
    if "installation" in prompt_lower or "manual" in prompt_lower:
        return "The installation manual specifies system setup procedures on page 1."

    return "Based on the retrieved document context, relevant details were identified."


@pytest.fixture(scope="module")
def real_documents_store():
    """Build a shared FAISS store containing real PDF documents."""
    pdf_paths = [SAMPLE_PDF, AI_PDF, MARKET_PDF, MULTIPAGE_PDF]
    store, metadata = build_store(pdf_paths=pdf_paths)
    return store, metadata


@pytest.fixture(autouse=True)
def reset_api_state():
    """Reset AppState before each test."""
    state = get_state()
    state.store = None
    state.sessions.clear()
    state.client = mock_real_doc_llm
    state.model_name = "mock-gemini"
    yield
    state.store = None
    state.sessions.clear()


# ------------------------------------------------------------
# 1. Real Document Grounded Query & Attribution Tests
# ------------------------------------------------------------

def test_real_document_query_grounded_response(real_documents_store):
    """Test executing a query against real indexed documents and verifying all required output fields."""
    store, metadata = real_documents_store
    assert len(metadata) == 4

    query = "What was CyberPulse Systems revenue in Q3 2026?"
    res = process_backend_query(
        client=mock_real_doc_llm,
        query=query,
        store=store,
        min_score=0.20,
    )

    # 1. Verify structured fields are present
    assert "answer" in res
    assert "source_document" in res
    assert "document_id" in res
    assert "relevant_context" in res

    # 2. Verify field content
    assert res["found"] is True
    assert "42.5 Million" in res["answer"]
    assert res["source_document"] == "market_analysis_2026.pdf"
    assert res["document_id"] == "market_analysis_2026"
    assert len(res["relevant_context"]) > 0
    assert "[Context Chunk 1]" in res["relevant_context"]

    # 3. Verify precise source attribution
    assert len(res["sources"]) > 0
    top_src = res["sources"][0]
    assert top_src["filename"] == "market_analysis_2026.pdf"
    assert top_src["document_id"] == "market_analysis_2026"
    assert top_src["page"] >= 1
    assert top_src["chunk_id"] >= 0
    assert top_src["score"] > 0.20
    assert top_src["confidence"] in ("HIGH", "MEDIUM", "LOW")
    assert len(top_src["text_preview"]) > 0


def test_page_level_attribution_multipage_doc(real_documents_store):
    """Verify page attribution when retrieving from a multi-page document."""
    store, metadata = real_documents_store
    query = "What microservices exist in OmniBrain?"

    res = process_backend_query(
        client=mock_real_doc_llm,
        query=query,
        store=store,
        min_score=0.20,
    )

    assert res["found"] is True
    assert res["document_id"] == "ai_architecture"
    assert res["source_document"] == "ai_architecture.pdf"
    assert len(res["sources"]) > 0
    assert "page" in res["sources"][0]
    assert res["sources"][0]["page"] >= 1


# ------------------------------------------------------------
# 2. Relevant vs No-Relevant-Context Scenarios
# ------------------------------------------------------------

def test_relevant_document_specific_filtering(real_documents_store):
    """Test querying a specific document using document_id constraint."""
    store, metadata = real_documents_store

    # Query with specific document_id matching the document
    res_match = process_backend_query(
        client=mock_real_doc_llm,
        query="What library is used for text extraction?",
        store=store,
        document_id="sample",
        min_score=0.20,
    )
    assert res_match["found"] is True
    assert res_match["document_id"] == "sample"
    assert res_match["source_document"] == "sample.pdf"

    # Query with non-matching document_id constraint
    res_mismatch = process_backend_query(
        client=mock_real_doc_llm,
        query="What was CyberPulse Systems revenue in Q3 2026?",
        store=store,
        document_id="sample",  # Revenue info is in market_analysis_2026, not sample
        min_score=0.50,
    )
    assert res_mismatch["found"] is False
    assert res_mismatch["answer"] == NO_CONTEXT_MESSAGE
    assert res_mismatch["document_id"] == "sample"
    assert res_mismatch["source_document"] == "sample.pdf"
    assert res_mismatch["relevant_context"] == ""


def test_no_relevant_context_out_of_domain(real_documents_store):
    """Test an out-of-domain query against real document store returning fallback."""
    store, metadata = real_documents_store
    query = "What is the average rainfall in the Amazon rainforest?"

    res = process_backend_query(
        client=mock_real_doc_llm,
        query=query,
        store=store,
        min_score=0.75,
    )

    assert res["found"] is False
    assert res["answer"] == NO_CONTEXT_MESSAGE
    assert res["source_document"] == "N/A"
    assert res["document_id"] == "N/A"
    assert res["relevant_context"] == ""
    assert res["sources"] == []
    assert res["retrieved_chunks"] == 0


# ------------------------------------------------------------
# 3. Error and Fallback Handling in Backend ↔ AI Module Communication
# ------------------------------------------------------------

def test_backend_query_endpoint_before_ingest():
    """Verify HTTP 503 error handling when backend calls /query without store or direct context."""
    with TestClient(app, raise_server_exceptions=True) as client:
        resp = client.post("/query", json={"query": "What is the system status?"})
        assert resp.status_code == 503
        assert "No documents indexed" in resp.json()["detail"]


def test_backend_query_endpoint_with_direct_context():
    """Verify POST /query works when backend passes direct document context."""
    with TestClient(app, raise_server_exceptions=True) as client:
        payload = {
            "query": "How much did latency decrease?",
            "context": "Hybrid FAISS Indexing reduced query latency from 85ms to 11.2ms.",
            "source_document": "market_analysis_2026.pdf",
            "document_id": "market_analysis_2026",
        }
        resp = client.post("/query", json=payload)
        assert resp.status_code == 200
        body = resp.json()

        assert body["found"] is True
        assert "11.2ms" in body["answer"]
        assert body["source_document"] == payload["source_document"]
        assert body["document_id"] == payload["document_id"]
        assert body["relevant_context"] == payload["context"]


def test_backend_query_llm_exception_handling():
    """Verify error handling when LLM client raises an exception during generation."""
    def failing_llm(prompt: str) -> str:
        raise RuntimeError("LLM Service Unavailable")

    with pytest.raises(RuntimeError, match="LLM Service Unavailable"):
        process_backend_query(
            client=failing_llm,
            query="What is the revenue?",
            context="CyberPulse Systems recorded revenue of $42.5 Million.",
            source_document="report.pdf",
            document_id="report",
        )


def test_schema_serialization_real_document():
    """Verify BackendQueryOutput model serialization for real document queries."""
    raw_res = {
        "query": "What are OmniBrain microservices?",
        "answer": "The OmniBrain architecture contains Document Parser Service and LLM Orchestrator.",
        "source_document": "ai_architecture.pdf",
        "document_id": "ai_architecture",
        "relevant_context": "[Context Chunk 1]\nDocument: ai_architecture.pdf (ID: ai_architecture) | Page: 1\nRelevance Score: 0.8500 (HIGH Confidence)\nContent:\nOmniBrain components...",
        "found": True,
        "sources": [
            {
                "source": "ai_architecture.pdf",
                "filename": "ai_architecture.pdf",
                "document_id": "ai_architecture",
                "chunk_id": 0,
                "page": 1,
                "score": 0.85,
                "confidence": "HIGH",
                "text_preview": "OmniBrain components...",
            }
        ],
        "retrieved_chunks": 1,
        "model": "gemini-2.5-flash",
    }

    sources = [SourceAttribution.from_dict(s) for s in raw_res["sources"]]
    output_model = BackendQueryOutput(
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

    dump = output_model.model_dump()
    assert dump["source_document"] == "ai_architecture.pdf"
    assert dump["document_id"] == "ai_architecture"
    assert dump["found"] is True
    assert len(dump["sources"]) == 1
