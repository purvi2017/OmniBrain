# Daily Progress

## Day 1

### Purvi
- Created GitHub repository
- Added collaborators
- Created branches
- Updated README
- Created project documentation

### Narsimha
- Created backend branch
- Setup FastAPI project
- Added Health API
- Added Upload API

### Srikanth
- Created AI module structure
- Added parsers folder
- Added embeddings folder
- Added vector database folder

---

## Day 2

### Purvi
- Reviewed backend progress
- Reviewed AI module progress
- Created integration plan
- Updated project documentation

### Narsimha
- Improving upload API
- Working on PDF validation
- Working on exception handling

### Srikanth
- Testing document processing pipeline
- Working on embeddings
- Working on vector database integration
  
---

## Day 3

### Purvi
- Reviewed backend and AI module deliverables
- Updated team roles and project ownership
- Updated project documentation
- Defined frontend requirements and user flow
- Created frontend development plan
- Designed initial UI structure for OmniBrain
- Planned integration between frontend, backend, and AI module
- Started frontend module implementation

### Narsimha
- Added PDF-only validation
- Added 10 MB file size validation
- Implemented UUID filename generation
- Added GET /documents endpoint
- Improved Swagger documentation
- Tested file size rejection (HTTP 413)
- Created POST /query endpoint
- Added QueryRequest schema
- Added QueryResponse schema
- Implemented query validation
- Added Swagger documentation
- Tested valid and invalid requests
- Pushed backend updates to GitHub

### Srikanth
- Completed document processing pipeline
- Implemented PDF text extraction
- Implemented image extraction
- Added chunking module
- Added embedding generation
- Integrated FAISS vector database
- Added Qdrant backend support
- Updated AI module documentation

---

## Day 4

### Purvi
- Developed OmniBrain frontend landing page
- Added professional landing page structure
- Added feature section for PDF Upload, AI Search, and Fast Retrieval
- Created PDF upload page
- Integrated frontend PDF upload with FastAPI `/upload` endpoint
- Added PDF file selection and validation
- Added upload success and error status handling
- Tested PDF upload from the frontend
- Created and pushed frontend updates to the `frontend` branch

### Narsimha
- Improved backend document management functionality
- Implemented document deletion API
- Added DELETE `/document/{document_id}` endpoint
- Tested document deletion through Swagger
- Verified successful document deletion response
- Continued backend API improvements
- Pushed backend updates to GitHub

### Srikanth
- Continued development of the AI/RAG module
- Worked on document processing and retrieval components
- Continued integration of embeddings and vector search
- Worked on AI module integration with the backend workflow
- Continued testing of the document retrieval pipeline

  ----

## Day 5

### Purvi
- Reviewed backend and AI module progress
- Coordinated document management and RAG module tasks
- Reviewed backend document upload, listing, and deletion workflow
- Reviewed AI retrieval pipeline implementation
- Updated project progress documentation
- Planned next steps for backend and AI module integration

### Narsimha
- Improved POST /upload API response structure
- Verified uploaded documents are stored with unique document IDs
- Improved GET /documents API to return document details
- Verified DELETE /document/{document_id} API with proper error handling
- Added handling for invalid and non-existing document IDs
- Tested Upload → List → Delete workflow through Swagger
- Verified backend CORS configuration for frontend integration
- Pushed Day 5 backend changes to the backend branch

### Srikanth
- Implemented RAG retrieval pipeline
- Integrated all-MiniLM-L6-v2 embedding model
- Generated 384-dimensional query embeddings
- Integrated FAISS vector index for similarity search
- Implemented relevant chunk retrieval
- Added similarity threshold filtering
- Tested RAG retrieval with multiple queries
- Verified relevant context retrieval from the vector database

## Day 6

### Purvi
- Coordinated Day 6 tasks for backend and AI modules
- Reviewed Narsimha and Srikanth's Day 5 deliverables
- Defined next steps for document processing and RAG integration
- Planned backend and AI module integration workflow
- Reviewed branch-wise project progress
- Updated daily project progress documentation


### Narsimha
- Worked on document processing API using document_id
- Added validation for invalid/non-existing document IDs
- Added proper error handling for document processing
- Prepared backend flow for PDF processing and AI module integration
- Tested document processing workflow through Swagger
- Pushed backend updates to GitHub


### Srikanth
- Worked on dynamic document processing for the RAG pipeline
- Prepared PDF text extraction and text chunking workflow
- Worked on embedding generation for processed document chunks
- Continued FAISS-based document retrieval
- Worked towards supporting real uploaded documents
- Tested document retrieval with different queries
- Prepared AI module for backend integration
