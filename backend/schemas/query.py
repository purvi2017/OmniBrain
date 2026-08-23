from pydantic import BaseModel, Field
from typing import List


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The question or query submitted by the user."
    )


class AIQueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="Validated user query forwarded to the AI module."
    )
    document_id: str = Field(
        ...,
        description="Primary document ID selected for AI processing."
    )
    source_document: str = Field(
        ...,
        description="Primary source document filename."
    )
    relevant_context: str = Field(
        ...,
        description="Relevant context extracted from the source document."
    )


class QuerySource(BaseModel):
    document_id: str = Field(
        ...,
        description="Unique ID of the source document."
    )
    filename: str = Field(
        ...,
        description="Name of the source PDF document."
    )


class QueryDocument(BaseModel):
    document_id: str = Field(
        ...,
        description="Unique ID of the relevant document."
    )
    filename: str = Field(
        ...,
        description="Name of the relevant PDF document."
    )
    path: str = Field(
        ...,
        description="Path of the relevant document."
    )
    matched_terms: List[str] = Field(
        ...,
        description="Query terms found in the document."
    )
    match_count: int = Field(
        ...,
        description="Number of matched query terms."
    )


class QueryResponse(BaseModel):
    status: str = Field(
        ...,
        description=(
            "Query processing status. Can be success, "
            "no_relevant_document, or error."
        )
    )
    query: str = Field(
        ...,
        description="The normalized user query."
    )
    answer: str = Field(
        ...,
        description="Answer returned by the AI/query processing layer."
    )
    source_document: str = Field(
        ...,
        description="Primary source document used for the result."
    )
    document_id: str = Field(
        ...,
        description="Document ID of the primary source document."
    )
    relevant_context: str = Field(
        ...,
        description="Relevant context prepared for AI/RAG processing."
    )
    sources: List[QuerySource] = Field(
        default_factory=list,
        description="All source documents supporting the result."
    )
    documents: List[QueryDocument] = Field(
        default_factory=list,
        description="Relevant documents found for the query."
    )