import os
import httpx
from fastapi import HTTPException

AI_MODULE_URL = os.getenv(
    "AI_MODULE_URL",
    "http://127.0.0.1:8001"
)
AI_QUERY_TIMEOUT = float(
    os.getenv("AI_QUERY_TIMEOUT", "30")
)


async def query_ai_module(
    query: str,
    document_id: str,
    source_document: str,
    relevant_context: str
):
    """
    Forward a query to the AI module over HTTP
    and return its parsed JSON response.

    Raises HTTPException with the appropriate
    status code if the AI module is unavailable,
    times out, or returns an invalid response.
    """

    payload = {
        "query": query,
        "context": relevant_context,
        "document_id": document_id,
        "source_document": source_document
    }

    try:
        async with httpx.AsyncClient(
            timeout=AI_QUERY_TIMEOUT
        ) as client:
            response = await client.post(
                f"{AI_MODULE_URL}/query",
                json=payload
            )

        response.raise_for_status()
        return response.json()

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="AI module is unavailable"
        ) from exc

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="AI module request timed out"
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"AI module returned HTTP "
                f"{exc.response.status_code}"
            )
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Failed to communicate with AI module"
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="AI module returned an invalid response"
        ) from exc