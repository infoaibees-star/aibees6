import warnings
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional

from query import RAGQueryEngine
from config import RETRIEVER_TOP_K

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backend")

app = FastAPI(
    title="Enterprise RAG Backend API",
    description="FastAPI service exposing Vertex AI Vector Search medical knowledge base",
    version="2.0.0"
)

# Global query engine instance (lazy initialized)
_engine = None

def get_query_engine() -> RAGQueryEngine:
    global _engine
    if _engine is None:
        try:
            log.info("Initializing global RAGQueryEngine...")
            _engine = RAGQueryEngine()
            log.info("RAGQueryEngine successfully initialized.")
        except Exception as exc:
            log.error("Failed to initialize RAGQueryEngine: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query engine initialization failed: {str(exc)}"
            )
    return _engine

# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str = Field(..., example="What is the dosage for Metformin?")
    history: Optional[List[Message]] = Field(default=None, example=[])
    top_k: Optional[int] = Field(default=None, example=5)

class SourceDoc(BaseModel):
    source_file: str
    page: Optional[int]
    source_gcs: str
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Lightweight health check endpoint for Cloud Run container probes."""
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    """
    Query the RAG pipeline with a medical question.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question parameter cannot be empty."
        )

    engine = get_query_engine()
    try:
        top_k = request.top_k if request.top_k is not None else RETRIEVER_TOP_K
        history = request.history if request.history is not None else []
        log.info("Invoking query: '%s' (history_len=%d, top_k=%d)", request.question, len(history), top_k)
        result = engine.query(request.question, history=history, top_k=top_k)
        return result
    except Exception as exc:
        log.error("Query execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(exc)}"
        )
