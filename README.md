# AI Module

## Overview

The AI module processes PDF documents and prepares their content for
semantic search and retrieval.

The pipeline supports:

- PDF text extraction
- Image extraction
- Text chunking
- Embedding generation
- FAISS vector storage
- Similarity search
- Top-K document retrieval

---

## Project Structure

```text
ai-module/
│
├── parsers/
│   ├── __init__.py
│   ├── pdf_extractor.py
│   ├── image_extractor.py
│   └── chunker.py
│
├── embeddings/
│   ├── __init__.py
│   └── embedder.py
│
├── vector_db/
│   ├── __init__.py
│   ├── faiss_store.py
│   └── qdrant_store.py
│
├── tests/
│   ├── __init__.py
│   ├── test_extraction.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_faiss_storage.py
│   ├── test_similarity_search.py
│   └── test_chunk_optimization.py
│
├── test_files/
│   └── sample.pdf
│
├── tools/
│   └── render_screenshot.py
│
├── process_document.py
├── requirements.txt
└── README.md