# OmniBrain Integration Plan

## System Flow

User
 ↓
Frontend
 ↓
FastAPI Backend
 ↓
Document Processing
 ↓
Embeddings Generation
 ↓
FAISS Vector Database
 ↓
LLM Response
 ↓
Frontend Display

## Workflow

1. User uploads PDF
2. Backend stores PDF
3. AI module extracts text
4. Text is divided into chunks
5. Embeddings are generated
6. Embeddings are stored in FAISS
7. User asks a question
8. Relevant chunks are retrieved
9. LLM generates answer
10. Answer displayed in UI