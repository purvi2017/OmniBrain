from pydantic import BaseModel, Field
from typing import List


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The question or query submitted by the user."
    )


class RelevantDocument(BaseModel):
    document_id: str = Field(
        ...,
        description="Unique ID of the matching document."
    )
    filename: str = Field(
        ...,
        description="Name of the matching PDF document."
    )
    path: str = Field(
        ...,
        description="Path of the matching document."
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
    message: str = Field(
        ...,
        description="Result message describing the query operation."
    )
    relevant_documents: List[RelevantDocument] = Field(
        default_factory=list,
        description="Documents considered relevant to the query."
    )