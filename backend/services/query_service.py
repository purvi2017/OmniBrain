import os
import re

import fitz
from fastapi import HTTPException


UPLOAD_DIR = "uploads"


def _normalize_text(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        word
        for word in words
        if len(word) >= 3
    }


def find_relevant_documents(query: str):
    if not os.path.exists(UPLOAD_DIR):
        return []

    query_words = _normalize_text(query)

    if not query_words:
        return []

    relevant_documents = []

    for filename in os.listdir(UPLOAD_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(UPLOAD_DIR, filename)

        try:
            pdf = fitz.open(file_path)

            document_text = "\n".join(
                page.get_text()
                for page in pdf
            )

            pdf.close()

            document_words = _normalize_text(document_text)

            matched_words = query_words.intersection(
                document_words
            )

            # Require at least two matching terms
            # to consider a document relevant.
            if len(matched_words) >= 2:
                relevant_documents.append({
                    "document_id": os.path.splitext(filename)[0],
                    "filename": filename,
                    "path": file_path,
                    "matched_terms": sorted(matched_words),
                    "match_count": len(matched_words)
                })

        except Exception:
            continue

    relevant_documents.sort(
        key=lambda document: document["match_count"],
        reverse=True
    )

    return relevant_documents


def process_query(query: str):
    try:
        relevant_documents = find_relevant_documents(query)

        if not relevant_documents:
            return {
                "status": "no_relevant_document",
                "query": query,
                "answer": (
                    "No relevant document was found for the query. "
                    "AI-generated answers will be available after "
                    "RAG integration."
                ),
                "sources": [],
                "documents": []
            }

        sources = [
            {
                "document_id": document["document_id"],
                "filename": document["filename"]
            }
            for document in relevant_documents
        ]

        return {
            "status": "success",
            "query": query,
            "answer": (
                "Relevant document(s) were found for the query. "
                "The backend is ready for future AI/RAG answer generation."
            ),
            "sources": sources,
            "documents": relevant_documents
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to process query"
        )