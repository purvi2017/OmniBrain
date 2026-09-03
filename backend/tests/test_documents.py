from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_get_documents():
    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert "documents" in data


def test_get_document_not_found():
    response = client.get("/document/nonexistent-file.pdf")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_delete_document_not_found():
    response = client.delete("/document/nonexistent-file.pdf")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_upload_invalid_file():
    response = client.post(
        "/upload",
        files={
            "file": (
                "test.txt",
                b"This is not a PDF file.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400