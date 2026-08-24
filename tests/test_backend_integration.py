"""
test_backend_integration.py
-----------------------------
Day 12: Backend Query Integration Layer Test Suite.

Verifies:
1. Reception of backend query + document context (direct context or vector store).
2. Input preparation for AI processing (context formatting & grounded prompt assembly).
3. Grounded answer generation flow when context is available.
4. Proper fallback response handling when context is unavailable (empty, whitespace, or missing).
5. Presence and correct population of required response fields:
   - answer
   - source_document
   - document_id
   - relevant_context
6. End-to-end verification across different test queries.
7. Validation against Pydantic BackendQueryInput and BackendQueryOutput schemas.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag_llm import (
    process_backend_query,
    build_store,
    NO_CONTEXT_MESSAGE,
    RAGLLMPipeline,
)
from schemas import BackendQueryInput, BackendQueryOutput, SourceAttribution

SAMPLE_PDF = str(BASE_DIR / "test_files" / "sample.pdf")
AI_PDF = str(BASE_DIR / "test_files" / "ai_architecture.pdf")
MARKET_PDF = str(BASE_DIR / "test_files" / "market_analysis_2026.pdf")


def mock_llm_responder(prompt: str) -> str:
    """Mock LLM callable providing deterministic grounded responses for backend integration tests."""
    prompt_lower = prompt.lower()
    if "latency" in prompt_lower or "85ms" in prompt_lower:
        return "Hybrid FAISS Indexing reduced query latency from 85ms to 11.2ms."
    if "cyberpulse" in prompt_lower or "revenue" in prompt_lower:
        return "CyberPulse Systems recorded revenue of $42.5 Million in Q3 2026."
    if "omnibrain" in prompt_lower or "microservice" in prompt_lower:
        return "The OmniBrain architecture contains Document Parser Service and LLM Orchestrator."
    if "pypdf" in prompt_lower or "extraction" in prompt_lower:
        return "pypdf is used for text extraction."
    return "Based on the provided document context, relevant details were found."


@pytest.fixture(scope="module")
def multi_doc_store():
    """Build a shared FAISS store containing sample.pdf, ai_architecture.pdf, and market_analysis_2026.pdf."""
    pdf_paths = [SAMPLE_PDF, AI_PDF, MARKET_PDF]
    store, metadata = build_store(pdf_paths=pdf_paths)
    return store, metadata


# ------------------------------------------------------------
# 1. Direct Context Input Tests (Context Available Flow)
# ------------------------------------------------------------

def test_direct_context_grounded_answer():
    """Test receiving query + direct context from backend and returning required response fields."""
    direct_context = "CyberPulse Systems achieved a total revenue of $42.5 Million in Q3 2026 with a net margin of 24%."
    query = "What was CyberPulse Systems revenue in Q3 2026?"

    response = process_backend_query(
        client=mock_llm_responder,
        query=query,
        context=direct_context,
        source_document="market_report_q3.pdf",
        document_id="market_report_q3",
    )

    # Verify all 4 required response fields
    assert "answer" in response
    assert "source_document" in response
    assert "document_id" in response
    assert "relevant_context" in response

    # Verify field values
    assert response["found"] is True
    assert "42.5 Million" in response["answer"]
    assert response["source_document"] == "market_report_q3.pdf"
    assert response["document_id"] == "market_report_q3"
    assert response["relevant_context"] == direct_context
    assert len(response["sources"]) > 0
    assert response["retrieved_chunks"] == 1


def test_vector_store_grounded_answer(multi_doc_store):
    """Test receiving query and retrieving context from FAISS store."""
    store, metadata = multi_doc_store
    query = "How much did FAISS indexing reduce query latency?"

    response = process_backend_query(
        client=mock_llm_responder,
        query=query,
        store=store,
        min_score=0.20,
    )

    # Verify required fields
    assert response["found"] is True
    assert "11.2ms" in response["answer"]
    assert response["source_document"] == "market_analysis_2026.pdf"
    assert response["document_id"] == "market_analysis_2026"
    assert len(response["relevant_context"]) > 0
    assert len(response["sources"]) > 0


# ------------------------------------------------------------
# 2. Context Unavailable & Fallback Response Tests
# ------------------------------------------------------------

def test_empty_context_fallback():
    """Test fallback response when direct context is empty or whitespace."""
    query = "What is the company profit?"

    res_empty = process_backend_query(
        client=mock_llm_responder,
        query=query,
        context="",
    )

    assert res_empty["found"] is False
    assert res_empty["answer"] == NO_CONTEXT_MESSAGE
    assert res_empty["source_document"] == "N/A"
    assert res_empty["document_id"] == "N/A"
    assert res_empty["relevant_context"] == ""
    assert res_empty["sources"] == []

    res_spaces = process_backend_query(
        client=mock_llm_responder,
        query=query,
        context="   \n\t  ",
    )
    assert res_spaces["found"] is False
    assert res_spaces["answer"] == NO_CONTEXT_MESSAGE


def test_irrelevant_query_store_fallback(multi_doc_store):
    """Test fallback response when FAISS retrieval finds no relevant context above score threshold."""
    store, metadata = multi_doc_store
    query = "What is the distance between the Earth and the Moon in miles?"

    response = process_backend_query(
        client=mock_llm_responder,
        query=query,
        store=store,
        min_score=0.85,  # High threshold guarantees no candidate matches
    )

    assert response["found"] is False
    assert response["answer"] == NO_CONTEXT_MESSAGE
    assert response["source_document"] == "N/A"
    assert response["document_id"] == "N/A"
    assert response["relevant_context"] == ""
    assert response["sources"] == []
    assert response["retrieved_chunks"] == 0


def test_missing_context_and_store_fallback():
    """Test fallback when neither context nor store is provided."""
    response = process_backend_query(
        client=mock_llm_responder,
        query="Explain quantum computing.",
        context=None,
        store=None,
    )

    assert response["found"] is False
    assert response["answer"] == NO_CONTEXT_MESSAGE
    assert response["source_document"] == "N/A"
    assert response["document_id"] == "N/A"
    assert response["relevant_context"] == ""


def test_empty_query_handling():
    """Test proper handling when query is empty or whitespace."""
    response = process_backend_query(
        client=mock_llm_responder,
        query="   ",
    )
    assert response["found"] is False
    assert "cannot be empty" in response["answer"].lower()
    assert response["source_document"] == "N/A"
    assert response["document_id"] == "N/A"


# ------------------------------------------------------------
# 3. Pydantic Schema Validation Tests
# ------------------------------------------------------------

def test_pydantic_backend_schemas():
    """Verify serialization and validation with BackendQueryInput and BackendQueryOutput schemas."""
    # Test Input Schema
    input_payload = BackendQueryInput(
        query="What microservices exist in OmniBrain?",
        context="OmniBrain architecture uses a Document Parser Service and an LLM Orchestrator.",
        document_id="ai_architecture",
        source_document="ai_architecture.pdf",
    )
    assert input_payload.query == "What microservices exist in OmniBrain?"
    assert input_payload.document_id == "ai_architecture"

    # Process query
    raw_res = process_backend_query(
        client=mock_llm_responder,
        query=input_payload.query,
        context=input_payload.context,
        document_id=input_payload.document_id,
        source_document=input_payload.source_document,
    )

    # Test Output Schema
    sources = [SourceAttribution.from_dict(s) for s in raw_res.get("sources", [])]
    output_payload = BackendQueryOutput(
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

    assert output_payload.found is True
    assert output_payload.source_document == "ai_architecture.pdf"
    assert output_payload.document_id == "ai_architecture"
    assert "Document Parser Service" in output_payload.relevant_context
    assert isinstance(output_payload.sources[0], SourceAttribution)


# ------------------------------------------------------------
# 4. Multi-Query Flow Verification
# ------------------------------------------------------------

def test_multiple_different_queries_flow(multi_doc_store):
    """Verify backend integration interface across multiple diverse queries."""
    store, metadata = multi_doc_store

    test_queries = [
        ("What was CyberPulse Systems revenue in Q3 2026?", "market_analysis_2026"),
        ("What are the key microservices in OmniBrain?", "ai_architecture"),
        ("What library is used for text extraction?", "sample"),
    ]

    for q_text, expected_doc in test_queries:
        res = process_backend_query(
            client=mock_llm_responder,
            query=q_text,
            store=store,
            min_score=0.20,
        )
        assert res["found"] is True, f"Failed for query: {q_text}"
        assert res["document_id"] == expected_doc, f"Expected {expected_doc}, got {res['document_id']}"
        assert res["source_document"] == f"{expected_doc}.pdf"
        assert len(res["answer"]) > 0
        assert len(res["relevant_context"]) > 0
