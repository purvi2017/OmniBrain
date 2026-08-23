import os
import re

import fitz
from fastapi import HTTPException

from schemas.query import AIQueryRequest


UPLOAD_DIR = "uploads"


def _normalize_text(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        word
        for word in words
        if len(word) >= 3
    }


def _build_relevant_context(
    document_text: str,
    matched_terms: list[str],
    max_length: int = 2000
) -> str:
    sentences = re.split(
        r"(?<=[.!?])\s+",
        document_text.strip()
    )

    relevant_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()

        if any(
            term.lower() in sentence_lower
            for term in matched_terms
        ):
            relevant_sentences.append(sentence.strip())

        if len(" ".join(relevant_sentences)) >= max_length:
            break

    context = " ".join(relevant_sentences).strip()

    if not context:
        context = document_text.strip()

    return context[:max_length]


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

        file_path = os.path.join(
            UPLOAD_DIR,
            filename
        )

        try:
            pdf = fitz.open(file_path)

            document_text = "\n".join(
                page.get_text()
                for page in pdf
            )

            pdf.close()

            document_words = _normalize_text(
                document_text
            )

            matched_words = query_words.intersection(
                document_words
            )

            if len(matched_words) >= 2:
                matched_terms = sorted(matched_words)

                relevant_documents.append({
                    "document_id": os.path.splitext(filename)[0],
                    "filename": filename,
                    "path": file_path,
                    "matched_terms": matched_terms,
                    "match_count": len(matched_terms),
                    "relevant_context": _build_relevant_context(
                        document_text,
                        matched_terms
                    )
                })

        except Exception:
            continue

    relevant_documents.sort(
        key=lambda document: document["match_count"],
        reverse=True
    )

    return relevant_documents


def build_ai_query_request(
    query: str,
    document: dict
) -> AIQueryRequest:
    return AIQueryRequest(
        query=query,
        document_id=document["document_id"],
        source_document=document["filename"],
        relevant_context=document["relevant_context"]
    )


def process_query(query: str):
    try:
        relevant_documents = find_relevant_documents(query)

        if not relevant_documents:
            return {
                "status": "no_relevant_document",
                "query": query,
                "answer": (
                    "No relevant document was found for the query."
                ),
                "source_document": "",
                "document_id": "",
                "relevant_context": "",
                "sources": [],
                "documents": []
            }

        primary_document = relevant_documents[0]

        ai_request = build_ai_query_request(
            query,
            primary_document
        )

        sources = [
            {
                "document_id": document["document_id"],
                "filename": document["filename"]
            }
            for document in relevant_documents
        ]

        documents = [
            {
                "document_id": document["document_id"],
                "filename": document["filename"],
                "path": document["path"],
                "matched_terms": document["matched_terms"],
                "match_count": document["match_count"]
            }
            for document in relevant_documents
        ]

        return {
            "status": "success",
            "query": ai_request.query,
            "answer": (
                "Relevant document context prepared successfully. "
                "The AI module can receive this request payload "
                "for answer generation."
            ),
            "source_document": ai_request.source_document,
            "document_id": ai_request.document_id,
            "relevant_context": ai_request.relevant_context,
            "sources": sources,
            "documents": documents
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to prepare query for AI module"
        )