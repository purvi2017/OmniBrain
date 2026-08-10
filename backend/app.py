from fastapi import FastAPI

app = FastAPI(
    title="OmniBrain API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "🚀 OmniBrain API Running Successfully!"
    }

@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }