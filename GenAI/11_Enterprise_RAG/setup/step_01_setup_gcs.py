#!/usr/bin/env python3
"""
step_01_setup_gcs.py
────────────────────
STEP 1 — Create GCS buckets required by the RAG pipeline.

Creates:
  • SOURCE_BUCKET  – where you upload raw PDF files
  • EMBED_BUCKET   – where Vector Search stores embedding artefacts

Run:
    python step_01_setup_gcs.py

Safe to re-run; existing buckets are skipped, not overwritten.
"""

import logging
import sys

from google.cloud import storage
from google.api_core.exceptions import Conflict

from config import (
    PROJECT_ID, REGION,
    SOURCE_BUCKET, EMBED_BUCKET,
    PDF_PREFIX,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def create_bucket(client: storage.Client, bucket_name: str, region: str) -> None:
    """Create a regional GCS bucket; skip if it already exists."""
    try:
        bucket = client.bucket(bucket_name)
        bucket.storage_class = "STANDARD"
        client.create_bucket(bucket, location=region)
        log.info("✓  Created bucket: gs://%s  (region=%s)", bucket_name, region)
    except Conflict:
        log.info("–  Bucket already exists, skipping: gs://%s", bucket_name)
    except Exception as exc:
        log.error("✗  Failed to create gs://%s: %s", bucket_name, exc)
        sys.exit(1)


def create_placeholder_prefix(client: storage.Client, bucket_name: str, prefix: str) -> None:
    """Create an empty placeholder object so the PDF prefix is visible in the console.
        else we dont see the folder until we upload the first pdf file
    """
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(f"{prefix}.keep")
    if not blob.exists():
        blob.upload_from_string(b"")
        log.info("✓  Created placeholder: gs://%s/%s.keep", bucket_name, prefix)


def main() -> None:
    log.info("=" * 55)
    log.info("STEP 1 — GCS Bucket Setup")
    log.info("Project : %s", PROJECT_ID)
    log.info("Region  : %s", REGION)
    log.info("=" * 55)

    client = storage.Client(project=PROJECT_ID)

    # 1a. Source bucket for raw PDFs
    create_bucket(client, SOURCE_BUCKET, REGION)
    create_placeholder_prefix(client, SOURCE_BUCKET, PDF_PREFIX)

    # 1b. Embedding artefacts bucket
    create_bucket(client, EMBED_BUCKET, REGION)

    log.info("-" * 55)
    log.info("Upload your PDF files to:  gs://%s/%s", SOURCE_BUCKET, PDF_PREFIX)
    log.info("Then run:  python step_02_create_index.py")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
