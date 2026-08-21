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
PROJECT_ID  = os.getenv("PROJECT_ID", os.getenv("GCP_PROJECT_ID", "project-f1be597b-8476-45ad-a2b"))
REGION      = os.getenv("REGION", os.getenv("GCP_REGION", "us-central1"))

# ── GCS Buckets ───────────────────────────────────────────────────────────────
EMBED_BUCKET    = os.getenv("EMBED_BUCKET", "aib-embeddings-0821")
EMBED_BUCKET_URI = f"gs://{EMBED_BUCKET}"

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
DIMENSIONS      = int(os.getenv("DIMENSIONS", "3072"))

# ── Vertex AI Vector Search ───────────────────────────────────────────────────
# INDEX_ID and ENDPOINT_ID are dynamic and must be passed as environment variables
INDEX_ID    = os.getenv("INDEX_ID")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")

if not INDEX_ID or not ENDPOINT_ID:
    # Try reading from setup's local .rag_config.json for seamless local developer experience
    _fallback_config = Path(__file__).parent.parent / "setup" / ".rag_config.json"
    if _fallback_config.exists():
        try:
            import json
            with open(_fallback_config) as f:
                _data = json.load(f)
                INDEX_ID = INDEX_ID or _data.get("index_id")
                ENDPOINT_ID = ENDPOINT_ID or _data.get("endpoint_id")
        except Exception:
            pass

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL         = os.getenv("LLM_MODEL", "gemini-2.5-pro")
LLM_TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS", "1024"))
RETRIEVER_TOP_K   = int(os.getenv("RETRIEVER_TOP_K", "5"))
