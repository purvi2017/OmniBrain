from fastapi import APIRouter, HTTPException

from schemas.query import QueryRequest, QueryResponse


router = APIRouter(
    prefix="/query",
    tags=["Query"]
)


@router.post(
    "",
    response_model=QueryResponse,
    summary="Submit a query",
    description=(
        "Accepts a user query and returns a placeholder response. "
        "AI processing, embeddings, and vector database integration "
        "will be added in later development stages."
    ),
    responses={
        400: {
            "description": "Invalid query request"
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

        return QueryResponse(
            status="success",
            query=query,
            message="Query received successfully. AI processing is not implemented yet."
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to process query request"
        )