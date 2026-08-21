#!/usr/bin/env python3
"""
step_02_create_index.py
───────────────────────
STEP 2 — Create a Vertex AI Vector Search index and deploy it to an endpoint.

  • Index type  : Tree-AH (ScaNN) — best for large-scale ANN search
  • Update mode : STREAM_UPDATE  — supports incremental document ingestion
                                   without full index rebuilds
  • Distance    : DOT_PRODUCT_DISTANCE (equivalent to cosine for normalised
                  gemini-embedding-001 vectors)

Saves index_id and endpoint_id to .rag_config.json for use by later steps.

Run:
    python step_02_create_index.py

⚠  This step takes ~30–45 minutes (index creation + endpoint deployment).
   Run it once. Re-running safely skips if .rag_config.json already exists.
"""

import json
import logging
import sys
from pathlib import Path

from google.cloud import aiplatform

from config import (
    PROJECT_ID, REGION, EMBED_BUCKET_URI,
    INDEX_DISPLAY_NAME, DEPLOYED_INDEX_ID,
    DIMENSIONS, RAG_CONFIG_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def create_index() -> aiplatform.MatchingEngineIndex:
    log.info("Creating Vector Search index: %s", INDEX_DISPLAY_NAME)
    log.info("  dimensions              = %d", DIMENSIONS)
    log.info("  update_method           = STREAM_UPDATE")
    log.info("  distance                = DOT_PRODUCT_DISTANCE")
    log.info("  leaf_node_embedding_count    = 500")
    log.info("  leaf_nodes_to_search_percent = 7")
    log.info("This may take 10–20 minutes …")

    # Use the enum for distance measure type — required by newer SDK versions
    # to avoid the 'algorithmConfig missing' FailedPrecondition error.
    from google.cloud.aiplatform.matching_engine.matching_engine_index_config import (
        DistanceMeasureType,
    )

    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        dimensions=DIMENSIONS,
        approximate_neighbors_count=150,
        # These two params are required — without them the API returns:
        # "algorithmConfig is required but missing from the metadata"
        leaf_node_embedding_count=500,       # vectors per leaf node
        leaf_nodes_to_search_percent=7,      # % of leaves searched per query
        distance_measure_type=DistanceMeasureType.DOT_PRODUCT_DISTANCE,
        # STREAM_UPDATE: vectors upserted immediately, no full rebuild needed
        index_update_method="STREAM_UPDATE",
        description="Enterprise RAG index — gemini-embedding-001 @ 3072 dims",
    )
    log.info("✓  Index created: %s", index.resource_name)
    return index


def create_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    log.info("Creating index endpoint …")
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=f"{INDEX_DISPLAY_NAME}-endpoint",
        public_endpoint_enabled=True,
        description="Enterprise RAG endpoint",
    )
    log.info("✓  Endpoint created: %s", endpoint.resource_name)
    return endpoint


def deploy_index(
    endpoint: aiplatform.MatchingEngineIndexEndpoint,
    index: aiplatform.MatchingEngineIndex,
) -> aiplatform.MatchingEngineIndexEndpoint:
    log.info("Deploying index to endpoint (this takes ~20 min) …")
    endpoint = endpoint.deploy_index(
        index=index,
        deployed_index_id=DEPLOYED_INDEX_ID,
        display_name=DEPLOYED_INDEX_ID,
        # Autoscaling: start with 1 replica, scale up to 2 under load
        min_replica_count=1,
        max_replica_count=2,
    )
    log.info("✓  Index deployed. Deployed indexes: %s", endpoint.deployed_indexes)
    return endpoint


def save_config(index: aiplatform.MatchingEngineIndex,
                endpoint: aiplatform.MatchingEngineIndexEndpoint) -> None:
    config = {
        "index_id":    index.name,
        "endpoint_id": endpoint.name,
    }
    with open(RAG_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    log.info("✓  Saved resource IDs to %s", RAG_CONFIG_FILE)
    log.info("\n" + "="*55)
    log.info("ENDPOINT HANDOFF INFORMATION")
    log.info("Set the following environment variables for your backend service:")
    log.info("  export INDEX_ID=%s", index.name)
    log.info("  export ENDPOINT_ID=%s", endpoint.name)
    log.info("="*55 + "\n")


def main() -> None:
    log.info("=" * 55)
    log.info("STEP 2 — Create & Deploy Vector Search Index")
    log.info("Project : %s", PROJECT_ID)
    log.info("Region  : %s", REGION)
    log.info("=" * 55)

    # Skip if already done
    if Path(RAG_CONFIG_FILE).exists():
        log.info("✓  %s already exists — index already created.", RAG_CONFIG_FILE)
        log.info("   Delete %s to force re-creation.", RAG_CONFIG_FILE)
        log.info("   Proceeding to next step: python step_03_ingest.py")
        sys.exit(0)

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=EMBED_BUCKET_URI)

    index    = create_index()
    endpoint = create_endpoint()
    endpoint = deploy_index(endpoint, index)
    save_config(index, endpoint)

    log.info("-" * 55)
    log.info("Next step:  python step_03_ingest.py")
    log.info("=" * 55)


if __name__ == "__main__":
    main()

