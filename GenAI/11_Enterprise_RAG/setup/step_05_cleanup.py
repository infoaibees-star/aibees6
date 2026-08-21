#!/usr/bin/env python3
"""
step_05_cleanup.py
──────────────────
STEP 5 — Delete all GCP resources created by the RAG pipeline.

Resources removed (in dependency order):
  1. Deployed index     (un-deploy from endpoint)
  2. Index endpoint     (delete)
  3. Vector Search index(delete)
  4. EMBED_BUCKET       (delete all objects + bucket)
  5. SOURCE_BUCKET      (optional — pass --delete-source to include)
  6. .rag_config.json   (local state file)

Run:
    python step_05_cleanup.py                       # interactive prompts
    python step_05_cleanup.py --yes                 # non-interactive (CI)
    python step_05_cleanup.py --yes --delete-source # also wipe raw PDF bucket
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from google.api_core.exceptions import NotFound
from google.cloud import aiplatform, storage

from config import (
    PROJECT_ID, REGION, EMBED_BUCKET_URI,
    EMBED_BUCKET, SOURCE_BUCKET, RAG_CONFIG_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def confirm(prompt: str, auto_yes: bool) -> bool:
    if auto_yes:
        log.info("AUTO-YES → %s", prompt)
        return True
    ans = input(f"\n  {prompt} [y/N]: ").strip().lower()
    return ans == "y"


def delete_bucket(bucket_name: str, auto_yes: bool) -> None:
    if not confirm(f"Delete gs://{bucket_name} and ALL its contents?", auto_yes):
        log.info("Skipped: gs://%s", bucket_name)
        return
    client = storage.Client(project=PROJECT_ID)
    try:
        bucket = client.get_bucket(bucket_name)
        blobs  = list(bucket.list_blobs())
        log.info("Deleting %d object(s) from gs://%s …", len(blobs), bucket_name)
        bucket.delete_blobs(blobs)
        bucket.delete()
        log.info("✓  Deleted bucket: gs://%s", bucket_name)
    except NotFound:
        log.warning("Bucket not found (already deleted): gs://%s", bucket_name)
    except Exception as exc:
        log.error("Failed to delete gs://%s: %s", bucket_name, exc)


# ── Vertex AI resource cleanup ────────────────────────────────────────────────

def load_config() -> dict:
    index_id = os.getenv("INDEX_ID")
    endpoint_id = os.getenv("ENDPOINT_ID")
    if index_id and endpoint_id:
        return {"index_id": index_id, "endpoint_id": endpoint_id}
    if not Path(RAG_CONFIG_FILE).exists():
        log.warning("%s not found and INDEX_ID/ENDPOINT_ID env vars not set — will try to list resources from API.", RAG_CONFIG_FILE)
        return {}
    with open(RAG_CONFIG_FILE) as f:
        return json.load(f)


def cleanup_vertex(config: dict, auto_yes: bool) -> None:
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=EMBED_BUCKET_URI)

    endpoint_id = config.get("endpoint_id")
    index_id    = config.get("index_id")

    # ── 1. Un-deploy + delete endpoint ────────────────────────────────────────
    if endpoint_id:
        if confirm(f"Un-deploy and delete endpoint {endpoint_id}?", auto_yes):
            try:
                endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=endpoint_id
                )
                for di in endpoint.deployed_indexes:
                    log.info("Un-deploying index %s …", di.id)
                    endpoint.undeploy_index(deployed_index_id=di.id)
                log.info("Deleting endpoint …")
                endpoint.delete(force=True)
                log.info("✓  Endpoint deleted: %s", endpoint_id)
            except NotFound:
                log.warning("Endpoint not found (already deleted): %s", endpoint_id)
            except Exception as exc:
                log.error("Failed to delete endpoint: %s", exc)
        else:
            log.info("Skipped endpoint deletion.")
    else:
        log.warning("No endpoint_id in config — skipping.")

    # ── 2. Delete index ────────────────────────────────────────────────────────
    if index_id:
        if confirm(f"Delete Vector Search index {index_id}?", auto_yes):
            try:
                index = aiplatform.MatchingEngineIndex(index_name=index_id)
                index.delete()
                log.info("✓  Index deleted: %s", index_id)
            except NotFound:
                log.warning("Index not found (already deleted): %s", index_id)
            except Exception as exc:
                log.error("Failed to delete index: %s", exc)
        else:
            log.info("Skipped index deletion.")
    else:
        log.warning("No index_id in config — skipping.")


def remove_local_config(auto_yes: bool) -> None:
    p = Path(RAG_CONFIG_FILE)
    if p.exists():
        if confirm(f"Remove local config file {RAG_CONFIG_FILE}?", auto_yes):
            p.unlink()
            log.info("✓  Removed %s", RAG_CONFIG_FILE)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5 — Cleanup GCP Resources")
    parser.add_argument("--yes",           action="store_true", help="Non-interactive: auto-confirm all")
    parser.add_argument("--delete-source", action="store_true", help="Also delete the raw PDF source bucket")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 55)
    log.info("STEP 5 — RAG Pipeline Cleanup")
    log.info("Project : %s", PROJECT_ID)
    log.info("=" * 55)
    log.info("⚠  This permanently deletes GCP resources.")

    if not args.yes:
        if not confirm("Proceed with cleanup?", False):
            print("Aborted.")
            sys.exit(0)

    config = load_config()

    # 1–2. Vertex AI (endpoint → index)
    cleanup_vertex(config, args.yes)

    # 3. Embedding artefacts bucket
    delete_bucket(EMBED_BUCKET, args.yes)

    # 4. (Optional) Source PDF bucket
    if args.delete_source:
        delete_bucket(SOURCE_BUCKET, args.yes)
    else:
        log.info("Source bucket gs://%s preserved. Pass --delete-source to remove.", SOURCE_BUCKET)

    # 5. Local config
    remove_local_config(args.yes)

    log.info("=" * 55)
    log.info("Cleanup done. Verify in the GCP Console that no resources remain.")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
