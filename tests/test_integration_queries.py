"""
test_integration_queries.py
----------------------------
Integration test suite for the AI Module RAG Pipeline interface.

Verifies:
1. Multi-document retrieval and answer generation using multiple document-based queries.
2. Document filtering by document_id.
3. Verification of answer flow and structured source attributions (source, filename, document_id, chunk_id, page, score, confidence, text_preview).
4. Handling of out-of-domain and irrelevant queries (found=False, empty sources, NO_CONTEXT_MESSAGE).
5. Validation of empty and whitespace queries.
6. Unified data schemas (schemas.py) for backend integration.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure project root is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag_llm import RAGLLMPipeline, build_store, answer_query, NO_CONTEXT_MESSAGE
from rag_chat import ConversationalRAGPipeline
from schemas import RAGQueryInput, RAGQueryOutput, SourceAttribution, ChatTurnInput, ChatTurnOutput

SAMPLE_PDF = str(BASE_DIR / "test_files" / "sample.pdf")
AI_PDF = str(BASE_DIR / "test_files" / "ai_architecture.pdf")
MARKET_PDF = str(BASE_DIR / "test_files" / "market_analysis_2026.pdf")


def mock_llm_responder(prompt: str) -> str:
    """Mock LLM callable providing deterministic grounded responses for tests."""
    prompt_lower = prompt.lower()
    if "cyberpulse" in prompt_lower or "revenue" in prompt_lower or "q3 2026" in prompt_lower:
        return "CyberPulse Systems reported record revenue of $42.5 Million in Q3 2026."
    if "latency" in prompt_lower or "faiss indexing" in prompt_lower:
        return "Hybrid FAISS Indexing reduced query latency from 85ms to 11.2ms."
    if "microservice" in prompt_lower or "omnibrain" in prompt_lower:
        return "The OmniBrain architecture uses a Document Parser Service and an LLM Orchestrator."
    if "pypdf" in prompt_lower or "extraction" in prompt_lower:
        return "pypdf is used for text extraction."
    return "Based on the retrieved context, relevant document details were found."


@pytest.fixture(scope="module")
def multi_doc_store():
    """Build a shared FAISS store containing sample.pdf, ai_architecture.pdf, and market_analysis_2026.pdf."""
    pdf_paths = [SAMPLE_PDF, AI_PDF, MARKET_PDF]
    store, metadata = build_store(pdf_paths=pdf_paths)
    return store, metadata


def test_multi_doc_queries_and_source_attribution(multi_doc_store):
    """Test multiple document-based queries across multi-doc store and verify rich source attribution."""
    store, metadata = multi_doc_store
    assert len(metadata) == 3

    pipeline = RAGLLMPipeline(store=store, client=mock_llm_responder, min_score=0.20)

    # Query 1: Market analysis PDF question
    res1 = pipeline.query("What was CyberPulse Systems revenue in Q3 2026?")
    assert res1["found"] is True
    assert "42.5 Million" in res1["answer"]
    assert len(res1["sources"]) > 0
    src1 = res1["sources"][0]
    assert src1["document_id"] == "market_analysis_2026"
    assert src1["filename"] == "market_analysis_2026.pdf"
    assert "CyberPulse" in src1["text_preview"]

    # Query 2: AI Architecture PDF question
    res2 = pipeline.query("What are the key microservices in OmniBrain?")
    assert res2["found"] is True
    assert "Document Parser Service" in res2["answer"] or "LLM Orchestrator" in res2["answer"]
    src2 = res2["sources"][0]
    assert src2["document_id"] == "ai_architecture"

    # Query 3: Sample PDF question
    res3 = pipeline.query("What library is used for text extraction?")
    assert res3["found"] is True
    assert len(res3["sources"]) > 0


def test_document_id_filtering(multi_doc_store):
    """Test constraining queries to a specific document_id."""
    store, metadata = multi_doc_store
    pipeline = RAGLLMPipeline(store=store, client=mock_llm_responder, min_score=0.20)

    # Query restricted to market_analysis_2026
    res = pipeline.query("What is the latency reduction?", document_id="market_analysis_2026")
    assert res["found"] is True
    for src in res["sources"]:
        assert src["document_id"] == "market_analysis_2026"

    # Query asking about market analysis but constrained to sample.pdf (should not find market analysis chunks)
    res_mismatch = pipeline.query("What was CyberPulse Systems revenue?", document_id="sample")
    # Since sample.pdf doesn't mention CyberPulse Systems with score >= min_score
    if res_mismatch["found"]:
        for src in res_mismatch["sources"]:
            assert src["document_id"] == "sample"


def test_out_of_domain_and_no_context_queries(multi_doc_store):
    """Verify handling of completely irrelevant and out-of-domain queries."""
    store, metadata = multi_doc_store
    pipeline = RAGLLMPipeline(store=store, client=mock_llm_responder, min_score=0.75)

    # Query completely unrelated to indexed PDFs
    res = pipeline.query("How do you bake a chocolate birthday cake?")
    assert res["found"] is False
    assert res["retrieved_chunks"] == 0
    assert res["sources"] == []
    assert res["answer"] == NO_CONTEXT_MESSAGE


def test_empty_and_whitespace_query_validation(multi_doc_store):
    """Verify handling of empty string and whitespace-only queries."""
    store, metadata = multi_doc_store

    res_empty = answer_query(client=mock_llm_responder, store=store, query="")
    assert res_empty["found"] is False
    assert res_empty["retrieved_chunks"] == 0
    assert res_empty["sources"] == []
    assert "cannot be empty" in res_empty["answer"].lower()

    res_spaces = answer_query(client=mock_llm_responder, store=store, query="   \n\t  ")
    assert res_spaces["found"] is False
    assert res_spaces["retrieved_chunks"] == 0


def test_schemas_data_contracts(multi_doc_store):
    """Verify backend integration data schemas (schemas.py)."""
    store, metadata = multi_doc_store

    # Test RAGQueryInput validation
    inp = RAGQueryInput(query="What is latency reduction?", top_k=3, min_score=0.20)
    assert inp.query == "What is latency reduction?"

    # Execute query
    raw_res = answer_query(
        client=mock_llm_responder,
        store=store,
        query=inp.query,
        top_k=inp.top_k,
        min_score=inp.min_score,
    )

    # Parse into RAGQueryOutput and SourceAttribution models
    sources = [SourceAttribution.from_dict(s) for s in raw_res.get("sources", [])]
    out = RAGQueryOutput(
        query=raw_res["query"],
        answer=raw_res["answer"],
        found=raw_res["found"],
        sources=sources,
        retrieved_chunks=raw_res["retrieved_chunks"],
        model=raw_res["model"],
    )

    assert out.found is True
    assert len(out.sources) > 0
    assert isinstance(out.sources[0], SourceAttribution)
    assert out.sources[0].filename != ""
    assert out.sources[0].confidence in ("HIGH", "MEDIUM", "LOW")


def test_conversational_chat_integration(multi_doc_store):
    """Verify stateful conversational RAG pipeline with turn management and attributions."""
    store, metadata = multi_doc_store

    chat_pipeline = ConversationalRAGPipeline(
        client=mock_llm_responder,
        session_id="integration-test-session",
    )
    chat_pipeline._pipeline.store = store

    # Turn 1
    turn1 = chat_pipeline.chat(message="What was CyberPulse Systems revenue in Q3 2026?", min_score=0.20)
    assert turn1["turn"] == 1
    assert turn1["found"] is True
    assert turn1["session_id"] == "integration-test-session"

    # Turn 2 (Follow up)
    turn2 = chat_pipeline.chat(message="What were their main R&D achievements?", min_score=0.20)
    assert turn2["turn"] == 2
    assert turn2["found"] is True

    # Validate ChatTurnOutput schema mapping
    sources = [SourceAttribution.from_dict(s) for s in turn2["sources"]]
    chat_out = ChatTurnOutput(
        session_id=turn2["session_id"],
        turn=turn2["turn"],
        query=turn2["query"],
        answer=turn2["answer"],
        found=turn2["found"],
        sources=sources,
        retrieved_chunks=turn2["retrieved_chunks"],
        model=turn2["model"],
    )
    assert chat_out.turn == 2
    assert chat_out.session_id == "integration-test-session"
