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
    summary="Submit a query for AI module processing",
    description=(
        "Validates the user query, identifies relevant uploaded "
        "documents, prepares the primary source document and "
        "relevant context, and returns an AI-module-ready response. "
        "The prepared data can be forwarded to the AI module for "
        "answer generation."
    ),
    responses={
        400: {
            "description": "Query cannot be empty"
        },
        422: {
            "description": "Missing or invalid query field"
        },
        500: {
            "description": (
                "Backend query processing or AI module failure"
            )
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

        result = process_query(query)

        return result

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "AI module is unavailable or "
                "query processing failed"
            )
        )