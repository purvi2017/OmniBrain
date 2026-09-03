import os
import re

import fitz
from fastapi import HTTPException

from schemas.query import AIQueryRequest
from services.ai_service import query_ai_module


UPLOAD_DIR = "uploads"


def _normalize_text(text: str) -> set[str]:
    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

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
            relevant_sentences.append(
                sentence.strip()
            )

        current_length = len(
            " ".join(relevant_sentences)
        )

        if current_length >= max_length:
            break

    context = " ".join(
        relevant_sentences
    ).strip()

    if not context:
        context = document_text.strip()

    return context[:max_length]


def find_relevant_documents(query: str):
    """
    Find uploaded PDF documents that contain
    at least two meaningful query terms.
    """

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

            # Require at least two matching terms.
            if len(matched_words) >= 2:

                matched_terms = sorted(
                    matched_words
                )

                relevant_documents.append({
                    "document_id": os.path.splitext(
                        filename
                    )[0],
                    "filename": filename,
                    "path": file_path,
                    "matched_terms": matched_terms,
                    "match_count": len(matched_terms),
                    "relevant_context": (
                        _build_relevant_context(
                            document_text,
                            matched_terms
                        )
                    )
                })

        except Exception:
            continue

    relevant_documents.sort(
        key=lambda document: document[
            "match_count"
        ],
        reverse=True
    )

    return relevant_documents


def build_ai_query_request(
    query: str,
    document: dict
) -> AIQueryRequest:
    """
    Build the exact request structure expected
    by the AI module.
    """

    return AIQueryRequest(
        query=query,
        document_id=document[
            "document_id"
        ],
        source_document=document[
            "filename"
        ],
        relevant_context=document[
            "relevant_context"
        ]
    )


def _map_ai_sources(ai_sources):
    """
    Convert AI module source attributions into
    the backend's simplified source structure.
    """

    sources = []

    for source in ai_sources or []:

        if not isinstance(source, dict):
            continue

        sources.append({
            "document_id": source.get(
                "document_id",
                ""
            ),
            "filename": source.get(
                "filename",
                source.get(
                    "source",
                    ""
                )
            )
        })

    return sources


async def process_query(query: str):
    """
    Complete backend query flow:

    1. Search uploaded PDFs.
    2. Select the primary document.
    3. Build AI-module request.
    4. Call AI module.
    5. Map AI response into backend response.
    """

    try:
        relevant_documents = (
            find_relevant_documents(query)
        )

        # No relevant document.
        if not relevant_documents:

            return {
                "status": "no_relevant_document",
                "query": query,
                "answer": (
                    "No relevant document was found "
                    "for the query."
                ),
                "source_document": "",
                "document_id": "",
                "relevant_context": "",
                "sources": [],
                "documents": []
            }

        # Highest-ranked document becomes
        # the primary source.
        primary_document = (
            relevant_documents[0]
        )

        # Build AI-module request.
        ai_request = build_ai_query_request(
            query,
            primary_document
        )

        # Call the REAL AI module.
        ai_response = await query_ai_module(
            query=ai_request.query,
            document_id=(
                ai_request.document_id
            ),
            source_document=(
                ai_request.source_document
            ),
            relevant_context=(
                ai_request.relevant_context
            )
        )

        # Map AI response sources.
        ai_sources = _map_ai_sources(
            ai_response.get(
                "sources",
                []
            )
        )

        # If the AI module returns no sources,
        # retain the backend's document source list.
        if not ai_sources:
            ai_sources = [
                {
                    "document_id": document[
                        "document_id"
                    ],
                    "filename": document[
                        "filename"
                    ]
                }
                for document in relevant_documents
            ]

        # Preserve backend document details.
        documents = [
            {
                "document_id": document[
                    "document_id"
                ],
                "filename": document[
                    "filename"
                ],
                "path": document[
                    "path"
                ],
                "matched_terms": document[
                    "matched_terms"
                ],
                "match_count": document[
                    "match_count"
                ]
            }
            for document in relevant_documents
        ]

        return {
            "status": "success",
            "query": query,
            "answer": ai_response.get(
                "answer",
                "AI module returned no answer."
            ),
            "source_document": ai_request.source_document,
            "document_id": ai_request.document_id,
            "relevant_context": ai_request.relevant_context,
            "sources": ai_sources,
            "documents": documents
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Backend-to-AI module communication "
                f"failed: {str(exc)}"
            )
        ) from exc