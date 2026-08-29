from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.upload import router as upload_router
from routes.query import router as query_router
from routes.documents import router as documents_router
from routes.process import router as process_router

app = FastAPI(
    title="OmniBrain Backend",
    description=(
        "Backend API for OmniBrain. Provides PDF upload, document "
        "listing, document processing, and AI-powered querying."
    ),
    version="1.0.0"
)

# CORS configuration — matches the frontend dev server (Live Server, port 5500).
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(process_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}