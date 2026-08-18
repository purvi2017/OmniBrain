from fastapi import APIRouter, HTTPException

from schemas.query import QueryRequest, QueryResponse
from services.query_service import process_query


router = APIRouter(
    prefix="/query",
    tags=["Query"]
)


@router.post(
    "",
    response_model=QueryResponse,
    summary="Submit a document-based query",
    description=(
        "Accepts a user query, validates it, searches the uploaded PDF "
        "documents for relevant terms, and returns matching documents. "
        "This provides a document-based query workflow that can later "
        "be replaced or extended with AI/RAG retrieval."
    ),
    responses={
        400: {
            "description": "Invalid or empty query"
        },
        500: {
            "description": "Query processing failed"
        }
    }
)
async def query_endpoint(request: QueryRequest):
    try:
        query = request.query.strip()

        if not query:
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )

        return process_query(query)

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to process query request"
        )