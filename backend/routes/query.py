from fastapi import APIRouter, HTTPException
from schemas.query import (
    QueryRequest,
    QueryResponse
)
from services.query_service import process_query

router = APIRouter(
    prefix="/query",
    tags=["Query"]
)


@router.post(
    "",
    response_model=QueryResponse,
    summary="Submit a query to the AI module",
    description=(
        "Validates a user query, searches uploaded PDF "
        "documents, prepares the document ID, source "
        "document, and relevant context, and forwards "
        "the prepared request to the AI module. The "
        "AI response is mapped into the backend response."
    ),
    responses={
        400: {
            "description": "Query cannot be empty"
        },
        422: {
            "description": (
                "Missing or invalid query field"
            )
        },
        502: {
            "description": (
                "AI module communication or AI "
                "module processing failure"
            )
        },
        503: {
            "description": (
                "AI module is unavailable"
            )
        },
        504: {
            "description": (
                "AI module request timed out"
            )
        }
    }
)
async def query_endpoint(
    request: QueryRequest
):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        return await process_query(query)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to complete "
                "Backend-to-AI query flow"
            )
        ) from exc