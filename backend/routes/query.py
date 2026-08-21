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
    summary="Submit a query for AI processing",
    description=(
        "Accepts and validates a user query, identifies relevant "
        "uploaded documents, prepares the primary source document "
        "and relevant context, and returns a structured response "
        "for future AI-module integration. The AI module can use "
        "the query, document ID, source document, and relevant "
        "context to generate the final answer."
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
                "Query processing or AI-module integration failure"
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

        return process_query(query)

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to process query for AI integration"
        )