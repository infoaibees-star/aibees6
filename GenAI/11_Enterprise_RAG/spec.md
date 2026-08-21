# Specification: Deployable 3-Service Layout for Enterprise_RAG (v2)

This specification outlines the architectural transition of the single-directory Vertex AI Vector Search RAG pipeline into a robust, deployable 3-service layout under `v2/`. All paths are relative to `~/usecase_2/Enterprise_RAG/v2/`.

---

## 1. Directory Structure

The `v2/` directory will be organized as follows:

```text
v2/
├── spec.md                     # This specification document
├── plan.md                     # The step-by-step implementation plan
├── README.md                   # Operator guide containing deployment commands & env vars
├── setup/                      # One-time RAG pipeline setup, run from notebook (never containerized)
│   ├── .env.example            # Template for local environment variables
│   ├── config.py               # Config module reading from env vars (with local defaults)
│   ├── requirements.txt        # Local setup dependencies
│   ├── step_01_setup_gcs.py    # Bucket creation
│   ├── step_02_create_index.py # Vector index and endpoint creation (outputs .rag_config.json)
│   ├── step_03_ingest.py       # Document parsing, chunking, and stream update ingestion
│   ├── step_04_query.py        # CLI interactive/standalone query interface for testing
│   └── step_05_cleanup.py      # Tear down GCP resources
├── backend/                    # FastAPI query service (Service #1)
│   ├── config.py               # Config module reading from env vars
│   ├── query.py                # Core RAG querying logic, ported from step_04_query.py
│   ├── main.py                 # FastAPI application and endpoint definitions
│   ├── requirements.txt        # Server-side python dependencies
│   └── Dockerfile              # Containerization recipe for Cloud Run (runs Uvicorn)
└── frontend/                   # Streamlit user interface (Service #2)
    ├── app.py                  # Streamlit application consuming Backend REST API
    ├── requirements.txt        # Client-side python dependencies (No google-cloud-* dependencies)
    ├── Dockerfile              # Containerization recipe for Cloud Run (runs Streamlit)
    └── .streamlit/
        └── config.toml         # Streamlit visual configuration (theme configuration placeholder)
```

---

## 2. Configuration & Environment Variables Single Source of Truth

To ensure environments are dynamic and secure, **no literal project IDs, bucket names, index IDs, endpoint IDs, or API keys will be hardcoded in Python files**.

### Environment Variables Matrix

| Variable Name | Purpose | Setup | Backend | Frontend | Default/Example |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `PROJECT_ID` | GCP Project ID | Yes | Yes | No | `project-f1be597b-8476-45ad-a2b` |
| `REGION` | GCP Resource Region | Yes | Yes | No | `us-central1` |
| `SOURCE_BUCKET` | Bucket holding raw PDFs | Yes | No | No | `aib6-raw-pdfs` |
| `PDF_PREFIX` | Folder path inside SOURCE_BUCKET | Yes | No | No | `pdfs/` |
| `EMBED_BUCKET` | Bucket for index artifacts & tracker | Yes | Yes | No | `aib-embeddings-0821` |
| `INDEX_DISPLAY_NAME` | Vector Search Index Display Name | Yes | No | No | `rag_vind_0821` |
| `DEPLOYED_INDEX_ID` | Vector Search Deployed Index ID | Yes | No | No | `rag_vind_0821_de` |
| `INDEX_ID` | ID of the created Vector Search Index | No | Yes | No | Generated in step_02 (e.g. `284759201948`) |
| `ENDPOINT_ID` | ID of the deployed Index Endpoint | No | Yes | No | Generated in step_02 (e.g. `482910485729`) |
| `GOOGLE_API_KEY` | API Key for Gemini Embedding API | Yes | Yes | No | AI Studio Developer API Key |
| `BACKEND_URL` | Base URL of the Backend service | No | No | Yes | `http://localhost:8080` (Local) / Cloud Run URL |

### Configuration Design (config.py)
Both `v2/setup/config.py` and `v2/backend/config.py` will load values from environment variables using `os.getenv()`. If variables are not present, they will fall back to local defaults matching the current setup (except for sensitive/dynamic items like `INDEX_ID` and `ENDPOINT_ID`, which are strictly required in the backend environment).

---

## 3. Service Components

### A. Setup Pipeline (`v2/setup/`)
The setup scripts are port-compatible replicas of the original pipeline scripts but decoupled from hardcoded python values:
- **`config.py`**: Refactored to leverage `os.getenv()`.
- **`step_02_create_index.py`**: Upon completion, it writes `index_id` and `endpoint_id` to `.rag_config.json` locally and outputs a clear prompt displaying the EXACT environment variables to set for the backend.
- **`step_03_ingest.py`**: Loads `.rag_config.json` locally if env variables are not supplied. Performs incremental chunking and uploads to the vector store.
- **`step_04_query.py`**: Preserved as a fast local diagnostic script.
- **`step_05_cleanup.py`**: Reads `.rag_config.json` and performs total teardown.

### B. Backend Service (`v2/backend/`)
A high-performance FastAPI service running under Uvicorn.
- **Core Querying (`query.py`)**: Imports from Vertex AI SDK and LangChain to query Vector Search. Reuses `format_docs` and `build_chain` from `step_04_query.py` but exposes them programmatically.
- **FastAPI Endpoints (`main.py`)**:
  - `GET /health` -> Returns `{"status": "healthy"}` for container probes.
  - `POST /query` -> Accepts JSON payload:
    ```json
    {
      "question": "What is the dosage for Metformin?",
      "top_k": 5
    }
    ```
    Returns:
    ```json
    {
      "answer": "Metformin dosage is...",
      "sources": [
        {
          "source_file": "general_drugs_guide_expanded.pdf",
          "page": 2,
          "source_gcs": "gs://aib6-raw-pdfs/pdfs/general_drugs_guide_expanded.pdf",
          "snippet": "Metformin Hydrochloride is indicated for..."
        }
      ]
    }
    ```
- **Authentication**: Fully relies on **Application Default Credentials (ADC)** when querying Vertex AI and GCS. **No service account private key files are stored or baked inside the Docker image.** The local or Cloud Run service account must hold `Vertex AI User` and `Storage Object Viewer` roles.
- **Network & Docker**:
  - Binds to `0.0.0.0` and listens on the port injected via the environment variable `$PORT` (defaulting to `8080`).
  - Utilizes standard multi-stage/optimized python slim builds.

### C. Frontend Service (`v2/frontend/`)
A clean, interactive Streamlit application.
- **Zero GCP SDK Footprint**: The Streamlit application communicates with the FastAPI backend solely using standard HTTP requests via `requests`. No `google-cloud-aiplatform` or `google-cloud-storage` dependencies in `v2/frontend/requirements.txt`.
- **Query Screen**:
  - Search input box.
  - Interactive "Retrieve Answer" button.
  - Renders the generated LLM response cleanly.
  - Expandable "Sources & Snippets" section mapping back to retrieved pages.
- **Network & Docker**:
  - Streamlit must bind to `0.0.0.0` and the environment-provided `$PORT`.
  - Docker execution command:
    ```bash
    streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
    ```

---

## 4. Verification & Handoff Strategy

1. **Local Setup Run**: Operator runs `step_01`, `step_02`, and `step_03` in `v2/setup/`. This generates GCS buckets, Vector Index, deploys the Endpoint, and ingests initial data.
2. **Handoff Vector search Endpoint**: The local file `.rag_config.json` captures `index_id` and `endpoint_id`. These are fed to the Backend service as:
   - `INDEX_ID`
   - `ENDPOINT_ID`
3. **Backend Service Deployment**: Deployed with backend env variables configured. Tested using `curl` against `GET /health` and `POST /query`.
4. **Frontend Service Deployment**: Deployed with `BACKEND_URL` pointing to the Backend Cloud Run service endpoint.

---

## 5. Architectural Alignment & Constraints Check

- **Constraint**: No shared package `rag_core`.
  - *Compliance*: `setup/` and `backend/` have distinct dependency trees and configurations. Code duplication is kept to basic initialization.
- **Constraint**: No hardcoded GCP Project IDs / Index IDs.
  - *Compliance*: Checked. All `.py` files use `os.getenv()`.
- **Constraint**: Streamlit must bind to dynamic `$PORT`.
  - *Compliance*: Checked. Handled in Streamlit command arguments.
- **Constraint**: Original `setup/` remains byte-for-byte unchanged.
  - *Compliance*: Checked. All work is isolated in `v2/`.
