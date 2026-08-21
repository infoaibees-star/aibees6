# =============================================================================
# config.py  —  Central configuration for the Enterprise RAG Pipeline
# Edit this file or set environment variables before running any pipeline step.
# =============================================================================
import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env if it exists in the same folder as this config file
_local_env = Path(__file__).parent / ".env"
if _local_env.exists():
    load_dotenv(_local_env)
else:
    load_dotenv()

# ── GCP Project ───────────────────────────────────────────────────────────────
PROJECT_ID  = os.getenv("PROJECT_ID", os.getenv("GCP_PROJECT_ID", "project-f1be597bb"))
REGION      = os.getenv("REGION", os.getenv("GCP_REGION", "us-central1"))

# ── GCS Buckets ───────────────────────────────────────────────────────────────
SOURCE_BUCKET   = os.getenv("SOURCE_BUCKET", "{your_name}-raw-pdfs")
PDF_PREFIX      = os.getenv("PDF_PREFIX", "pdfs/")
EMBED_BUCKET    = os.getenv("EMBED_BUCKET", "aib-embeddings-yourname-0821")
EMBED_BUCKET_URI = f"gs://{EMBED_BUCKET}"

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
DIMENSIONS      = int(os.getenv("DIMENSIONS", "3072"))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Vertex AI Vector Search ───────────────────────────────────────────────────
INDEX_DISPLAY_NAME  = os.getenv("INDEX_DISPLAY_NAME", "rag_vind_yourname_0821")
DEPLOYED_INDEX_ID   = os.getenv("DEPLOYED_INDEX_ID", "rag_vind_yourname_0821_de")

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL         = os.getenv("LLM_MODEL", "gemini-2.5-pro")
LLM_TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS", "1024"))
RETRIEVER_TOP_K   = int(os.getenv("RETRIEVER_TOP_K", "5"))

# ── Tracker (incremental ingestion state) ────────────────────────────────────
TRACKER_BLOB = os.getenv("TRACKER_BLOB", "metadata/ingested_files.json")

# ── Local state file (written by step_02, read by step_03 / step_04) ─────────
RAG_CONFIG_FILE = os.getenv("RAG_CONFIG_FILE", ".rag_config.json")