"""
Environment-driven configuration.

Every value can be overridden from `.env` or the process environment.
Import the module-level ``settings`` singleton; it is parsed once at startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

Backend = Literal["vertexai", "ollama", "keyword"]

VALID_BACKENDS: tuple[Backend, ...] = ("vertexai", "ollama", "keyword")

_TRUTHY = {"1", "true", "yes", "on"}


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name) or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name) or default)
    except ValueError:
        return default


def _resolve_backend() -> Backend:
    """RAG_BACKEND wins; USE_VERTEXAI is kept as a legacy fallback."""
    explicit = _str("RAG_BACKEND").lower()
    if explicit in VALID_BACKENDS:
        return explicit  # type: ignore[return-value]
    legacy = _str("USE_VERTEXAI").lower()
    if legacy:
        return "vertexai" if legacy in _TRUTHY else "keyword"
    return "vertexai"


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    # Which pipeline answers questions:
    #   vertexai — Gemini on Vertex AI + FAISS vector search (needs GCP auth)
    #   ollama   — local Ollama LLM + FAISS vector search (fully offline)
    #   keyword  — no LLM at all; keyword retrieval + extractive answer
    backend: Backend

    # Google Cloud (backend="vertexai")
    gcp_project_id: str
    gcp_region: str

    # Ollama (backend="ollama")
    ollama_base_url: str
    ollama_model: str
    ollama_embedding_model: str

    # Models
    embedding_model: str
    llm_model: str
    temperature: float
    max_output_tokens: int
    top_k: int
    top_p: float

    # Documents
    default_document: str
    chunk_size: int
    chunk_overlap: int
    retriever_top_k: int

    # Index
    index_root: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            backend=_resolve_backend(),
            gcp_project_id=_str("GCP_PROJECT_ID"),
            gcp_region=_str("GCP_REGION", "us-central1"),
            ollama_base_url=_str("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=_str("OLLAMA_MODEL", "llama3.2:latest"),
            ollama_embedding_model=_str("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            embedding_model=_str("EMBEDDING_MODEL", "text-embedding-005"),
            llm_model=_str("GEMINI_MODEL") or _str("LLM_MODEL", "gemini-2.5-pro"),
            temperature=_float("LLM_TEMPERATURE", 0.1),
            max_output_tokens=_int("MAX_OUTPUT_TOKENS", 1024),
            top_k=_int("LLM_TOP_K", 40),
            top_p=_float("LLM_TOP_P", 0.8),
            default_document=_str("PDF_PATH", "customer_credit_data_100_records.pdf"),
            chunk_size=_int("CHUNK_SIZE", 500),
            chunk_overlap=_int("CHUNK_OVERLAP", 100),
            retriever_top_k=_int("RETRIEVER_TOP_K", 4),
            index_root=_str("FAISS_INDEX_DIR", "faiss_index"),
        )

    @property
    def uses_vector_store(self) -> bool:
        """True when the backend embeds chunks into FAISS."""
        return self.backend in ("vertexai", "ollama")

    def validate(self) -> list[str]:
        """Return human-readable problems that would stop the app from running."""
        problems: list[str] = []
        if self.backend == "vertexai" and not self.gcp_project_id:
            problems.append(
                "GCP_PROJECT_ID is not set (required for RAG_BACKEND=vertexai). "
                "Set it in .env, or switch to RAG_BACKEND=ollama / RAG_BACKEND=keyword."
            )
        if self.chunk_overlap >= self.chunk_size:
            problems.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.retriever_top_k < 1:
            problems.append("RETRIEVER_TOP_K must be at least 1.")
        return problems


settings = Settings.from_env()
