# AI Module

## Overview

The AI module is an enterprise-grade Retrieval-Augmented Generation (RAG) system that processes documents (such as PDFs), indexes their contents into a vector database (FAISS), and generates grounded, source-attributed answers using Large Language Models (Gemini API via `google-genai`).

---

## RAG + LLM Expected Flow

```text
User Query
    ↓
Query Embedding (all-MiniLM-L6-v2)
    ↓
FAISS Retrieval (IndexFlatIP Cosine Similarity)
    ↓
Relevance Filtering (Similarity Threshold >= min_score)
    ↓
Context Assembly (Numbered Chunks + Source Metadata)
    ↓
Gemini LLM (Grounded Context Prompt)
    ↓
Generated Answer
    ↓
Structured Response (Answer + Source Document Metadata)
```

---

## Key Features

- **Document Processing**: Robust PDF text extraction (pypdf/PyMuPDF) and sentence-boundary-preserving chunking.
- **Dense Vector Embeddings**: 384-dimensional normalized embeddings via `sentence-transformers/all-MiniLM-L6-v2`.
- **FAISS Vector Storage**: Fast inner product (cosine similarity) search with complete document metadata tracking (`document_id`, `filename`, `chunk_id`, `score`, `text`).
- **Semantic Retrieval**: Top-K retrieval with relevance threshold filtering (`min_score`) to exclude irrelevant context.
- **Context-Grounded LLM Generation**: Strict prompt design ensuring the LLM uses *only* retrieved facts without external hallucination.
- **Missing Context & Out-of-Domain Fallback**: Deterministic, graceful fallback when no document context meets the relevance threshold.
- **Structured Response Format**: Returns a JSON-compatible dictionary with generated answer, retrieval flags, and source document metadata.

---

## Project Structure

```text
ai-module/
│
├── parsers/
│   ├── __init__.py
│   ├── pdf_extractor.py          # PDF text extraction
│   ├── image_extractor.py        # PDF image extraction
│   └── chunker.py                # Overlapping text chunker
│
├── embeddings/
│   ├── __init__.py
│   └── embedder.py               # Sentence-Transformers embedder
│
├── vector_db/
│   ├── __init__.py
│   ├── faiss_store.py            # FAISS vector store with metadata & persistence
│   └── qdrant_store.py           # Optional Qdrant store
│
├── tests/
│   ├── __init__.py
│   ├── test_extraction.py        # PDF extraction tests
│   ├── test_chunking.py          # Chunking boundary tests
│   ├── test_embeddings.py        # Vector embedding tests
│   ├── test_faiss_storage.py     # FAISS storage & persistence tests
│   ├── test_similarity_search.py # FAISS semantic search tests
│   ├── test_chunk_optimization.py# Chunk size optimization benchmarks
│   └── test_rag_llm.py           # Day 7 RAG + LLM end-to-end test suite
│
├── test_files/
│   └── sample.pdf                # Verification test document
│
├── tools/
│   └── render_screenshot.py      # Terminal output screenshot renderer
│
├── process_document.py           # Ingestion & index pipeline CLI
├── rag_retriever.py              # Semantic retrieval pipeline
├── rag_llm.py                    # Day 7 RAG + LLM Answer Generation pipeline
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## Installation & Setup

1. **Activate Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure Gemini API Key**:
   ```powershell
   $env:GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

---

## Usage

### 1. Run RAG + LLM Pipeline (CLI)

Ask questions directly about single or multiple PDF documents:

```powershell
# Single document query
python rag_llm.py test_files/sample.pdf --query "What is the purpose of this document?"

# Multiple documents with custom retrieval parameters
python rag_llm.py doc1.pdf doc2.pdf --query "What are the key findings?" --top-k 3 --min-score 0.30 --model gemini-2.5-flash

# Interactive mode (prompts for question in terminal)
python rag_llm.py test_files/sample.pdf
```

### 2. Programmatic Python API

```python
from rag_llm import RAGLLMPipeline

# Initialize RAG + LLM pipeline
pipeline = RAGLLMPipeline(
    model_name="gemini-2.5-flash",
    top_k=3,
    min_score=0.30,
)

# Ingest and index PDF documents
pipeline.load_documents(["test_files/sample.pdf"])

# Query the pipeline
response = pipeline.query("What library is used for text extraction?")

print("Answer:", response["answer"])
print("Found Context:", response["found"])
print("Sources:", response["sources"])
```

---

## Structured Response Format

Every query returns a structured dictionary:

```json
{
  "query": "What library is used for text extraction?",
  "answer": "Based on the retrieved context, pypdf is used for text extraction from the sample PDF.",
  "found": true,
  "sources": [
    {
      "source": "sample.pdf",
      "filename": "sample.pdf",
      "document_id": "sample",
      "chunk_id": 0,
      "score": 0.7852,
      "text_preview": "This is a sample PDF document created for testing the AI module pipeline. It contains multiple sentences so that the chunking logic..."
    }
  ],
  "retrieved_chunks": 1,
  "model": "gemini-2.5-flash"
}
```

### Handling Missing Context (Out-of-Domain)

When a query cannot be answered from the uploaded document(s) (no chunks meet the relevance threshold):

```json
{
  "query": "What is the capital of Mars?",
  "answer": "I could not find relevant information in the uploaded documents to answer this question.",
  "found": false,
  "sources": [],
  "retrieved_chunks": 0,
  "model": "gemini-2.5-flash"
}
```

---

## Running Verification Tests

Run the complete automated pytest suite:

```powershell
pytest tests/ -v
```

Run the dedicated Day 7 RAG + LLM test suite:

```powershell
python tests/test_rag_llm.py
```