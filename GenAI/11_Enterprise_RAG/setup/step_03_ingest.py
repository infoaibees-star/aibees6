#!/usr/bin/env python3
"""
step_03_ingest.py
─────────────────
STEP 3 — Ingest PDF documents from GCS into the Vector Search index.

Reads GOOGLE_API_KEY from .env for the embedding model.
Embeddings are written to EMBED_BUCKET via VectorSearchVectorStore.

Modes:
  --mode full          Ingest ALL PDFs under PDF_PREFIX (default)
  --mode incremental   Only ingest PDFs not yet in the tracker

Run:
    python step_03_ingest.py --mode full
    python step_03_ingest.py --mode incremental
"""

import argparse
import warnings
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

# ── Load .env first ───────────────────────────────────────────────────────────
from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# ── Remaining imports ─────────────────────────────────────────────────────────
from google.cloud import aiplatform, storage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VectorSearchVectorStore, VertexAIEmbeddings

from config import (
    PROJECT_ID, REGION,
    SOURCE_BUCKET, PDF_PREFIX,
    EMBED_BUCKET, EMBED_BUCKET_URI,
    CHUNK_SIZE, CHUNK_OVERLAP,
    TRACKER_BLOB, RAG_CONFIG_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EMBEDDING_MODEL  = "models/gemini-embedding-001"
MAX_EMBED_BATCH  = 200   # Vertex AI embedding API limit is 250; stay under


# ── GCS helpers ───────────────────────────────────────────────────────────────

def gcs_client() -> storage.Client:
    return storage.Client(project=PROJECT_ID)


def list_pdfs(prefix: str) -> list[str]:
    client = gcs_client()
    blobs  = client.bucket(SOURCE_BUCKET).list_blobs(prefix=prefix)
    pdfs   = [b.name for b in blobs if b.name.lower().endswith(".pdf")]
    return pdfs


def download_pdf(blob_name: str) -> str:
    """
    Download a PDF from GCS to a local temp file. Returns the local file path."""
    client = gcs_client()
    blob   = client.bucket(SOURCE_BUCKET).blob(blob_name)
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    blob.download_to_file(tmp)
    tmp.close()
    return tmp.name


def verify_embeddings_in_gcs() -> int:
    """Count embedding-related objects written to EMBED_BUCKET."""
    client = gcs_client()
    blobs  = list(client.bucket(EMBED_BUCKET).list_blobs())
    count  = len(blobs)
    log.info("GCS verification — gs://%s contains %d object(s):", EMBED_BUCKET, count)
    for b in blobs[:10]:   # show first 10
        log.info("  • %s  (%d bytes)", b.name, b.size)
    if count > 10:
        log.info("  … and %d more", count - 10)
    return count


# ── Tracker ───────────────────────────────────────────────────────────────────

def load_tracker() -> set:
    """
    Reads a tracking file stored in GCS that remembers which PDFs have already been ingested
    — so incremental mode knows what to skip.
    """
    client = gcs_client()
    blob   = client.bucket(EMBED_BUCKET).blob(TRACKER_BLOB)
    if blob.exists():
        data = json.loads(blob.download_as_text())
        return set(data.get("ingested", []))
    return set()


def save_tracker(ingested: set) -> None:
    client = gcs_client()
    blob   = client.bucket(EMBED_BUCKET).blob(TRACKER_BLOB)
    blob.upload_from_string(
        json.dumps({"ingested": sorted(ingested)}, indent=2),
        content_type="application/json",
    )
    log.info("Tracker saved — %d file(s) recorded in gs://%s/%s",
             len(ingested), EMBED_BUCKET, TRACKER_BLOB)


# ── PDF processing ────────────────────────────────────────────────────────────

def load_and_chunk(local_path: str, gcs_name: str) -> list:
    loader = PyPDFLoader(local_path)
    pages  = loader.load()
    if not pages:
        raise ValueError("PDF has 0 pages — may be corrupt or an HTML redirect.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source_gcs"]  = f"gs://{SOURCE_BUCKET}/{gcs_name}"
        chunk.metadata["source_file"] = Path(gcs_name).name
    return chunks


# ── Batched upsert ────────────────────────────────────────────────────────────

def upsert_in_batches(vector_store, texts: list[str], metadatas: list[dict]) -> None:
    """
    Sends chunks to Vertex AI Vector Search in small groups (batches) instead of all at once
    — because the embedding API has a limit on how many texts it can process in one call.
    """
    total   = len(texts)
    n_batch = math.ceil(total / MAX_EMBED_BATCH)
    log.info("  Upserting %d chunk(s) in %d batch(es) of ≤%d …",
             total, n_batch, MAX_EMBED_BATCH)
    for i in range(n_batch):
        start = i * MAX_EMBED_BATCH
        end   = min(start + MAX_EMBED_BATCH, total)
        vector_store.add_texts(
            texts=texts[start:end],
            metadatas=metadatas[start:end],
        )
        log.info("  Batch %d/%d → %d vectors upserted", i + 1, n_batch, end - start)


# ── Main ingestion loop ───────────────────────────────────────────────────────

def ingest(prefix: str, incremental: bool) -> None:
    # Attempt to load from env vars first
    index_id = os.getenv("INDEX_ID")
    endpoint_id = os.getenv("ENDPOINT_ID")

    if not index_id or not endpoint_id:
        if not Path(RAG_CONFIG_FILE).exists():
            log.error("%s not found and INDEX_ID/ENDPOINT_ID env vars not set. Run step_02_create_index.py first.", RAG_CONFIG_FILE)
            sys.exit(1)
        with open(RAG_CONFIG_FILE) as f:
            config = json.load(f)
        index_id = index_id or config.get("index_id")
        endpoint_id = endpoint_id or config.get("endpoint_id")

    log.info("Index ID    : %s", index_id)
    log.info("Endpoint ID : %s", endpoint_id)

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=EMBED_BUCKET_URI)

    # Embedder — VertexAIEmbeddings uses ADC
    log.info("Loading Vertex AI embedding model: gemini-embedding-001")
    embedder = VertexAIEmbeddings(
        model_name="gemini-embedding-001",
        project=PROJECT_ID,
        location=REGION,
    )

    log.info("Connecting to Vector Search …")
    vector_store = VectorSearchVectorStore.from_components(
        project_id=PROJECT_ID,
        region=REGION,
        gcs_bucket_name=EMBED_BUCKET,
        index_id=index_id,
        endpoint_id=endpoint_id,
        embedding=embedder,
        stream_update=True,
    )

    all_pdfs = list_pdfs(prefix)
    log.info("Found %d PDF(s) under gs://%s/%s", len(all_pdfs), SOURCE_BUCKET, prefix)

    if not all_pdfs:
        log.error("No PDFs found! Upload files first:")
        log.error("  bash download_medical_pdfs.sh")
        sys.exit(1)

    if incremental:
        ingested  = load_tracker()
        to_ingest = [p for p in all_pdfs if p not in ingested]
        log.info("Incremental — %d new / %d already ingested",
                 len(to_ingest), len(ingested))
    else:
        ingested  = set()
        to_ingest = all_pdfs

    if not to_ingest:
        log.info("Nothing new to ingest.")
        return

    success, failed = 0, 0

    for blob_name in to_ingest:
        log.info("─" * 55)
        log.info("Processing: gs://%s/%s", SOURCE_BUCKET, blob_name)
        local_path = None
        try:
            local_path = download_pdf(blob_name)
            log.info("  Downloaded → %s", local_path)

            chunks = load_and_chunk(local_path, blob_name)
            log.info("  Chunks generated: %d", len(chunks))

            texts     = [c.page_content for c in chunks]
            metadatas = [c.metadata     for c in chunks]

            upsert_in_batches(vector_store, texts, metadatas)

            ingested.add(blob_name)
            success += 1
            log.info("  ✓ Ingested: %s", blob_name)

        except Exception as exc:
            log.error("  ✗ Failed: %s — %s", blob_name, exc)
            failed += 1
        finally:
            if local_path and os.path.exists(local_path):
                os.unlink(local_path)

    # Save tracker
    if success > 0:
        save_tracker(ingested)

    # ── Verify GCS actually has embedding objects ─────────────────────────────
    log.info("─" * 55)
    log.info("Verifying GCS embedding artefacts …")
    count = verify_embeddings_in_gcs()
    if count == 0:
        log.warning("⚠  No objects found in gs://%s after ingestion!", EMBED_BUCKET)
        log.warning("   Check that the index endpoint is DEPLOYED and stream_update=True.")
    else:
        log.info("✓  GCS artefacts verified: %d object(s) present", count)

    log.info("=" * 55)
    log.info("Ingestion complete — success: %d  failed: %d", success, failed)
    if failed:
        log.warning("Re-run with --mode incremental to retry failed files.")
    else:
        log.info("Next step: python step_04_query.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3 — Ingest PDFs into Vector Search")
    parser.add_argument("--mode",   choices=["full", "incremental"], default="full")
    parser.add_argument("--prefix", default=PDF_PREFIX)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log.info("=" * 55)
    log.info("STEP 3 — PDF Ingestion")
    log.info("Mode   : %s", args.mode)
    log.info("Prefix : gs://%s/%s", SOURCE_BUCKET, args.prefix)
    log.info("=" * 55)
    ingest(prefix=args.prefix, incremental=(args.mode == "incremental"))

