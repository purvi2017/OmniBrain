# 🧠 OmniBrain – Agentic Multi-Modal RAG Orchestrator

## 📌 Project Overview

**OmniBrain** is an AI-powered Agentic Retrieval-Augmented Generation (RAG) system designed for intelligent document understanding and question answering.

The system allows users to upload PDF documents and ask natural-language questions about their content. OmniBrain processes the document, extracts and chunks its content, generates vector embeddings, stores them in a FAISS vector index, retrieves relevant information for a user's query, and uses a Gemini LLM to generate a context-grounded response.

The project combines **PDF processing, semantic search, vector retrieval, RAG, conversational AI, and a web-based frontend** into a single system.

---

## 🎯 Objectives

The main objectives of OmniBrain are:

- To build an AI-powered document question-answering system.
- To implement Retrieval-Augmented Generation (RAG).
- To retrieve relevant information from uploaded PDF documents.
- To convert document content into vector embeddings.
- To perform semantic similarity search using FAISS.
- To generate grounded responses using Gemini LLM.
- To support conversational queries using session-based history.
- To provide source document information with generated answers.
- To provide an easy-to-use web interface for document interaction.

---

# ✨ Key Features

### 📄 PDF Document Upload

Users can upload PDF documents through the frontend interface. The uploaded documents are processed and prepared for intelligent querying.

### 🔍 Document Processing

The system extracts text from uploaded PDF documents and processes the extracted content before indexing it.

### ✂️ Text Chunking

Large document content is divided into smaller chunks so that relevant sections can be efficiently retrieved during question answering.

### 🧠 Vector Embeddings

Document chunks are converted into numerical vector representations using the **all-MiniLM-L6-v2** embedding model.

The generated embeddings have a dimension of **384**.

### 🗂️ FAISS Vector Search

The generated embeddings are stored in a **FAISS vector index**, allowing the system to perform fast similarity-based retrieval.

### 🤖 Retrieval-Augmented Generation

When a user asks a question, OmniBrain retrieves relevant document content and provides that context to the Gemini LLM before generating the answer.

### 💬 Conversational RAG

The system supports session-based conversations, allowing follow-up questions while maintaining relevant conversation history.

### 📚 Source-Aware Responses

The frontend displays the source document and relevant context associated with the generated response.

### 🚫 Out-of-Context Handling

If the requested information cannot be found in the uploaded documents, the system can return a message indicating that relevant information was not found instead of generating an unsupported answer.

### 🎨 Interactive Frontend

The project provides a modern web interface for:

- Uploading documents
- Asking questions
- Viewing responses
- Viewing source documents
- Viewing retrieved context
- Using suggested queries

---

# 🏗️ System Architecture

OmniBrain consists of three main components:

### 1. Frontend

The frontend provides the user interface for document upload and querying.

**Technologies:**
- HTML
- CSS
- JavaScript

### 2. Backend

The FastAPI backend manages document-related requests and communicates between the frontend and AI module.

**Technology:**
- Python
- FastAPI

### 3. AI / RAG Module

The AI module performs document ingestion, embedding generation, vector retrieval, conversational processing, and Gemini-based response generation.

**Technologies:**
- Python
- FAISS
- Sentence Transformers
- Gemini LLM
- RAG

---

# 🔄 Overall System Workflow

```text
                    ┌─────────────────┐
                    │      USER       │
                    │                 │
                    │ PDF / Question  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    FRONTEND     │
                    │ HTML / CSS / JS │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     FASTAPI     │
                    │     BACKEND     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   AI / RAG      │
                    │     MODULE      │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
             ┌─────────────┐   ┌─────────────┐
             │    FAISS    │   │   Gemini    │
             │ Vector      │   │    LLM      │
             │ Retrieval   │   │ Generation  │
             └──────┬──────┘   └──────┬──────┘
                    │                 │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ FINAL RESPONSE  │
                    │ + Source/Context│
                    └─────────────────┘
```

---

# 📄 Document Ingestion Workflow

The document ingestion pipeline follows these steps:

```text
PDF Document
      ↓
PDF Upload
      ↓
Text Extraction
      ↓
Text Processing
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
FAISS Indexing
      ↓
Document Ready for Querying
```

### Processing Details

The system:

1. Receives the uploaded PDF.
2. Extracts text from the document.
3. Processes and cleans the extracted content.
4. Divides the content into chunks.
5. Generates embeddings for the chunks.
6. Stores the vectors in FAISS.
7. Makes the document available for semantic retrieval.

---

# 🔎 Query Processing Workflow

When the user asks a question:

```text
User Question
      ↓
Query Processing
      ↓
Query Embedding
      ↓
FAISS Similarity Search
      ↓
Relevant Document Chunks
      ↓
Context Construction
      ↓
Gemini LLM
      ↓
Generated Response
      ↓
Frontend
```

The retrieved document context is used to generate a grounded response.

---

# 💬 Conversational RAG Workflow

OmniBrain also supports conversational interactions.

```text
Previous Conversation
          +
Current Question
          ↓
Conversation Processing
          ↓
Relevant Context Retrieval
          ↓
FAISS Search
          ↓
Context + Conversation
          ↓
Gemini LLM
          ↓
Context-Aware Response
```

This allows users to ask follow-up questions without repeating the complete context.

---

# 🧠 RAG Pipeline

The core RAG architecture is:

```text
                 DOCUMENT
                    │
                    ▼
             Text Extraction
                    │
                    ▼
              Text Chunking
                    │
                    ▼
            Embedding Model
                    │
                    ▼
              FAISS Index
                    │
                    │
                    │
USER QUERY ─────────┘
     │
     ▼
Query Embedding
     │
     ▼
Similarity Retrieval
     │
     ▼
Relevant Context
     │
     ▼
Gemini LLM
     │
     ▼
Final Answer
```

---

# 🛠️ Technologies Used

## Frontend

* HTML5
* CSS3
* JavaScript
* Google Fonts
* Live Server

## Backend

* Python
* FastAPI
* Uvicorn
* REST APIs
* CORS

## AI / RAG

* Retrieval-Augmented Generation
* FAISS
* Sentence Transformers
* all-MiniLM-L6-v2
* Gemini LLM
* Semantic Search
* Conversational RAG

## Document Processing

* PDF Text Extraction
* Text Cleaning
* Text Chunking
* Metadata Handling

## Version Control

* Git
* GitHub
* Git Branching
* Git Worktree

---

# 📁 Project Structure

```text
OmniBrain/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── uploads/
│
├── frontend/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │   └── app.js
│   │
│   ├── image/
│   │   └── logo.png
│   │
│   └── pages/
│       ├── index.html
│       ├── dashboard.html
│       ├── upload.html
│       └── query.html
│
├── docs/
│
├── api.py
├── rag_chat.py
├── rag_llm.py
├── rag_retriever.py
├── rag_response_generator.py
├── process_document.py
│
├── parsers/
├── embeddings/
├── models/
├── routes/
├── services/
│
└── README.md
```

---

# 🔌 Main API Functionality

The project provides APIs for:

* Health checking
* PDF uploading
* Document listing
* Document deletion
* Document processing
* Document querying
* AI document ingestion
* RAG querying
* Conversational chat
* Session management

---

# 🧪 Testing & Validation

The integrated OmniBrain system was tested using a sample PDF document:

**Data Science & Analytics.pdf**

### Document Processing Result

```text
Pages Extracted: 11
Total Characters: 7496
Chunks Created: 13
Embedding Vectors: 13
Vector Dimension: 384
Documents Indexed: 1
Chunks Indexed: 13
```

The document was successfully added to the FAISS index.

---

## ✅ Query Testing

Different types of queries were tested successfully.

### Direct Information Query

**Question:**

> What is the objective of the Customer Segmentation Using K-Means project?

The system successfully retrieved the relevant information from the uploaded PDF.

### Data-Based Query

**Question:**

> What data is used for customer segmentation in the K-Means project?

The system retrieved relevant attributes such as:

* Age
* Income
* Frequency
* Spending

### Key Points Query

**Question:**

> What are the key points?

The system successfully retrieved the important information related to the project from the document.

### Preprocessing Query

**Question:**

> What preprocessing step is required before applying K-Means clustering?

The system correctly retrieved that **scaling** is required before applying K-Means clustering.

### Out-of-Context Query

An unrelated question was also tested.

The system returned:

> I could not find relevant information in the uploaded documents to answer this question.

This confirms that the system can identify when relevant information is not available in the uploaded document.

---

# 🌟 Advantages

* Intelligent document question answering
* Semantic rather than simple keyword-based search
* Context-grounded AI responses
* Fast vector retrieval using FAISS
* Conversational question answering
* Source document information
* User-friendly interface
* Modular architecture
* Easy integration of AI and backend services
* Reduces manual document searching

---

# 🚀 Future Enhancements

Possible future improvements include:

* Advanced multi-modal document understanding
* Image and chart analysis
* OCR integration
* Table understanding
* Multiple specialized AI agents
* Advanced agent routing
* Qdrant or other vector database support
* Authentication and user management
* Cloud deployment
* Improved document management
* Support for additional LLM providers

---

# 👥 Team Members

| Name            | Role                                           |
| --------------- | ---------------------------------------------- |
| **Purvi Patel** | Team Leader & Integration / Frontend Developer |
| **Narsimha**    | Backend Developer                              |
| **Srikanth**    | AI Module Developer                            |

---

# 👩‍💻 Team Responsibilities

### Purvi Patel

* Team leadership and coordination
* Frontend development
* Frontend-backend integration
* API integration
* Final integration testing
* Git/GitHub management
* Project documentation

### Narsimha

* Backend development
* FastAPI implementation
* Document upload APIs
* Document processing APIs
* Backend services
* Backend integration

### Srikanth

* AI/RAG module development
* PDF processing
* Text chunking
* Embedding generation
* FAISS implementation
* Gemini LLM integration
* Conversational RAG

---

# 📊 Project Status

| Component                      | Status         |
| ------------------------------ | -------------- |
| Frontend                       | ✅ Completed    |
| Backend                        | ✅ Completed    |
| AI/RAG Module                  | ✅ Completed    |
| Frontend API Integration       | ✅ Completed    |
| Backend + AI Integration       | ✅ Completed    |
| Frontend + Backend Integration | ✅ Completed    |
| PDF Upload                     | ✅ Tested       |
| PDF Processing                 | ✅ Tested       |
| Embedding Generation           | ✅ Tested       |
| FAISS Retrieval                | ✅ Tested       |
| RAG Query                      | ✅ Tested       |
| Conversational RAG             | ✅ Implemented  |
| Out-of-Context Handling        | ✅ Tested       |
| Final Integration Testing      | ✅ Completed    |
| GitHub Integration             | ✅ Completed    |
| Documentation                  | ✅ Completed

---

# 📌 GitHub Repository

**OmniBrain – Agentic Multi-Modal RAG Orchestrator**

[https://github.com/purvi2017/OmniBrain](https://github.com/purvi2017/OmniBrain)

---

# 🏁 Conclusion

OmniBrain is an AI-powered document intelligence system that combines **PDF processing, semantic search, vector databases, Retrieval-Augmented Generation, Gemini LLM, and conversational AI**.

The system enables users to upload documents and interact with their content through natural-language questions while providing relevant, context-grounded responses.

The project successfully integrates the **Frontend, FastAPI Backend, AI/RAG Module, FAISS Vector Retrieval, and Gemini LLM** into a complete document question-answering solution.

---

## 🧠 OmniBrain

**Know More. Achieve More.**
