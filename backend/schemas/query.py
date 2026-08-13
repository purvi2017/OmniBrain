from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The question or query submitted by the user."
    )


class QueryResponse(BaseModel):
    status: str = Field(
        ...,
        description="Status of the query request."
    )
    query: str = Field(
        ...,
        description="The original user query."
    )
    message: str = Field(
        ...,
        description="Placeholder response until AI query processing is integrated."
    )