# OmniBrain – Agentic Multi-Modal RAG Orchestrator

## Overview

OmniBrain is an AI-powered Agentic Multi-Modal RAG (Retrieval-Augmented Generation) system being developed as part of the Axlero Solutions Internship Program.

The platform enables users to upload documents, retrieve relevant information, and generate intelligent, context-aware responses using advanced AI techniques. OmniBrain is designed to process complex documents containing text, tables, charts, and images while reducing hallucinations through grounded retrieval and agent-based reasoning.

---

## Problem Statement

Traditional RAG systems face several limitations:

- Difficulty processing multi-modal documents
- Limited reasoning across multiple data sources
- Poor retrieval accuracy in large documents
- Increased chances of AI hallucinations
- Lack of intelligent workflow orchestration

OmniBrain addresses these challenges through an agent-based architecture that combines document processing, retrieval, reasoning, and intelligent task routing.

---

## Current Progress

### Phase 1 – Project Foundation ✅

- Project Architecture Designed
- Team Roles Defined
- FastAPI Backend Setup
- Frontend Structure Created
- PDF Upload Module Implemented
- File Storage System Configured
- API Documentation Enabled using Swagger UI
- GitHub Repository Initialized
- Frontend–Backend Integration Completed

---

## Key Features

### Implemented Features

- PDF Upload System
- FastAPI Backend
- Frontend Upload Interface
- REST API Architecture
- Secure File Storage
- Swagger API Documentation
- Frontend–Backend Integration

### Upcoming Features

- PDF Text Extraction
- Multi-Modal Document Processing
- Image Extraction and Analysis
- Vector Database Integration (ChromaDB)
- Semantic Search
- Context-Aware Question Answering
- Agentic Workflow using LangGraph
- LLM Integration
- Hallucination Reduction through Grounded Retrieval
- Streamlit Dashboard
- Multi-Agent Reasoning Pipeline

---

## Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript

### AI & RAG (Upcoming)

- LangChain
- LangGraph
- ChromaDB
- OpenAI / Gemini API
- Sentence Transformers

### Development Tools

- Git & GitHub
- VS Code
- Postman
- Swagger UI

---

## Team Members

| Name | Role |
|--------|--------|
| Purvi Patel | Team Leader, Frontend Developer & Integration |
| Narsimha | Backend Developer |
| Srikanth | AI Module Developer |
| Aayan | Documentation & Testing Support |

---

## Project Structure

```text
OmniBrain/
│
├── backend/
│   ├── uploads/
│   ├── utils/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │   └── app.js
│   │
│   └── pages/
│       ├── index.html
│       ├── upload.html
│       ├── dashboard.html
│       └── query.html
│
├── docs/
│   ├── architecture.md
│   ├── objectives.md
│   ├── project_overview.md
│   ├── problem_statement.md
│   ├── team_roles.md
│   ├── integration_plan.md
│   └── daily_progress.md
│
├── README.md
├── .gitignore
└── requirements.txt
