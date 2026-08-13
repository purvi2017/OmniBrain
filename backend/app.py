from fastapi import FastAPI

from routes.upload import router as upload_router
from routes.documents import router as documents_router
from routes.query import router as query_router


app = FastAPI(
    title="OmniBrain API",
    version="1.0.0",
    description=(
        "Backend API for OmniBrain. "
        "Provides PDF upload, document listing, and query API "
        "structure for future AI integration."
    )
)


@app.get(
    "/",
    summary="Home",
    description="Returns the basic status message of the OmniBrain API."
)
def home():
    return {
        "message": "🚀 OmniBrain API Running Successfully!"
    }


@app.get(
    "/health",
    summary="Health Check",
    description="Checks whether the OmniBrain backend API is healthy."
)
def health():
    return {
        "status": "Healthy"
    }


# Register API routes
app.include_router(upload_router)
app.include_router(documents_router)
app.include_router(query_router)