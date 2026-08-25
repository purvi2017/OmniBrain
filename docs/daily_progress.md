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

--------

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

-------------

## Day 7

### Purvi

- Focused on exam preparation and temporarily reduced project involvement
- Coordinated Day 7 tasks with the Backend and AI team members
- Reviewed the planned Backend–AI integration workflow
- Kept track of team progress and pending development tasks
  
### Narsimha

- Continued development of the document processing workflow
- Worked on connecting document processing with the AI module
- Prepared document path and document_id flow for AI processing
- Worked on processing status and error handling
- Tested the document processing workflow through Swagger
- Prepared the backend for integration with the RAG pipeline

### Srikanth

- Worked on integrating the RAG retrieval pipeline with an LLM
- Prepared the query → retrieval → context → answer generation flow
- Passed retrieved document chunks as context for answer generation
- Worked on generating answers based on retrieved document content
- Added handling for cases with no relevant context
- Tested the RAG + LLM pipeline with different queries
- Prepared the AI module for backend integration

-------------

## Day 8

### Purvi

* Focused on exam preparation
* Temporarily paused project development activities due to ongoing exams
* Coordinated with team members regarding their individual tasks
* Reviewed overall project progress

### Narsimha

* Improved `POST /query` API handling
* Added handling for queries with no relevant documents
* Improved query validation and error handling
* Tested valid and invalid queries through Swagger
* Verified `no_relevant_document` response for unrelated queries
* Tested document-based query workflow
* Improved API response structure
* Continued backend API testing

### Srikanth

* Improved RAG retrieval pipeline
* Tested FAISS-based document retrieval with multiple queries
* Improved handling of irrelevant and out-of-domain queries
* Added/verified similarity-based filtering
* Tested multi-document retrieval
* Verified retrieved context and source information
* Tested RAG answer generation workflow
* Continued AI module testing and validation

---

## Day 9

### Purvi

* Continued exam preparation
* No active project development due to ongoing exams
* Coordinated with team members and reviewed project progress
* Kept track of backend and AI module development

### Narsimha

* Continued testing and refinement of the `/query` API
* Tested queries against uploaded documents
* Verified proper response when relevant documents are found
* Verified proper handling when no relevant document is found
* Improved backend exception handling
* Verified API responses through Swagger
* Reviewed backend readiness for future AI module integration
* Continued maintaining the `backend` branch

### Srikanth

* Completed comprehensive testing of the RAG pipeline
* Tested chunking, embeddings and FAISS vector storage
* Tested RAG retrieval accuracy with multiple queries
* Tested multi-document retrieval and similarity filtering
* Verified empty-query and out-of-domain query handling
* Tested grounded prompt and LLM answer generation
* Completed end-to-end multi-document answer generation testing
* **All 16 AI module tests passed successfully**
* Verified the AI module is ready for the next integration stage

---

## Day 10

###Purvi

- Continued exam preparation
- No active project development due to ongoing exams
- Reviewed team progress and pending tasks
- Coordinated with Backend and AI team members
  
###Narsimha

- Improved POST /query API response structure
- Added proper handling for answer and source information
- Tested queries with relevant and irrelevant documents
- Improved error handling for query processing
- Tested empty and invalid queries through Swagger
- Updated query API documentation and response schema
- Prepared the backend query API for future AI integration
- Committed and pushed changes to the backend branch
  
###Srikanth

- Prepared the AI/RAG module for backend integration
- Defined input and output structure for the RAG pipeline
- Improved query-to-answer processing flow
- Added relevant source/document information to the response
- Tested AI module with multiple document-based queries
- Verified handling of irrelevant queries and missing context
- Ran the complete AI module test suite
- Verified that existing tests continue to pass
- Committed and pushed changes to the ai-module branch

--------

## Day 11

###Purvi

- Continued exam preparation
- No active project development due to ongoing exams
- Reviewed team progress and pending tasks
- Coordinated with Backend and AI team members

### Narsimha

- Improved backend document processing workflow
- Refined /process/{document_id} API handling
- Added proper validation for document processing requests
- Tested processing of valid uploaded documents
- Added handling for invalid or non-existing document_id
- Verified document status and processing responses through Swagger
- Improved API error handling and response messages
- Tested the complete Upload → Process → Query backend flow
- Updated backend documentation and pushed changes to the backend branch
  
### Srikanth

- Continued improvement of the RAG pipeline
- Worked on improving retrieval quality for document-based queries
- Tested retrieval with different similarity thresholds
- Verified relevant chunk selection from uploaded documents
- Improved source/document metadata handling in retrieved results
- Tested RAG pipeline with multiple queries and documents
- Verified no-relevant-context handling
- Re-ran the AI module test suite after improvements
- Updated AI module documentation and pushed changes to the ai-module branch

---------

## Day 12

### Purvi

- Continued frontend development for the OmniBrain project.
- Worked on connecting the frontend with the backend APIs.
- Tested the PDF upload flow and API communication.
- Continued development of the document query interface.
- Coordinated with the team and reviewed overall project progress.
  
### Narsimha

- Continued backend development and API integration.
- Worked on testing the /upload API and document handling flow.
- Tested backend endpoints using Swagger.
- Continued work on the /query API integration.
- Coordinated with the frontend team for API requirements.
  
### Srikanth

- Continued development of the AI module for document-based querying.
- Worked on the query processing flow and response handling.
- Tested document query functionality.
- Reviewed the integration requirements between AI and backend modules.
- Coordinated with the team on AI module progress.

----------

## Day 13

### Purvi

- Completed the frontend structure for the OmniBrain project.
- Completed and tested the PDF upload flow.
- Integrated the frontend with the /upload backend API.
- Completed the query page and integrated the /query API.
- Tested document upload and query responses successfully.
- Committed and pushed the completed frontend work to the frontend branch.
  
### Narsimha

- Continued backend API testing and refinement.
- Tested the /upload and /query endpoints with the frontend.
- Verified API responses and frontend-backend communication.
- Checked error handling and request-response flow.
- Coordinated with the team for successful API integration.
  
### Srikanth

- Continued testing and refinement of the AI document query functionality.
- Worked on query processing and response generation.
- Tested different document-related queries.
- Reviewed AI response integration with the backend /query API.
- Coordinated with the frontend and backend team members for integration.

----------

## Day 14 — Daily Progress

### Purvi – Frontend

- Completed the query page structure for OmniBrain.
- Implemented the query input and Ask Query interaction.
- Integrated the frontend with the backend /query API.
- Tested empty query validation and successful query requests.
- Verified that answer, source document, document ID, and relevant context are displayed correctly.
- Tested the query flow with multiple questions.
- Continued preparing the frontend for upcoming Backend + AI Module integration.
  
### Narsimha – Backend

- Continued development and testing of the Backend /query API.
- Worked on Backend ↔ AI Module communication requirements.
- Verified query request and response handling.
- Tested document-related information and structured API responses.
- Reviewed error handling and integration requirements for the real document flow.
- Continued preparing the backend for AI Module integration.
  
### Srikanth – AI Module

- Continued work on the AI Module integration with the Backend.
- Worked on handling queries and relevant document context received from the Backend.
- Tested the document-based query and response generation flow.
- Reviewed structured response fields required for Backend integration.
- Tested relevant-context and fallback scenarios.
- Continued preparing the AI Module for real document-based responses.
