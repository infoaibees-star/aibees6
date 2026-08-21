# Implementation Plan: 3-Service Layout for Enterprise_RAG (v2)

This document tracks the specific development tasks completed to build, verify, and package the 3-service deployable version of Enterprise_RAG under `v2/`.

---

## Phase 1: Research & Preparation (Completed)
- [x] Read and inspect original read-only source scripts inside `setup/`.
- [x] Create `v2/` directory structure.
- [x] Write `v2/spec.md`.

---

## Phase 2: Setup Pipeline Setup (Completed)
- [x] **Task 2.1**: Copy original setup scripts from `setup/` to `v2/setup/` (ignoring app.py and local caches).
- [x] **Task 2.2**: Refactor `v2/setup/config.py` to pull values from environment variables using `os.getenv` and local default values, supporting both `PROJECT_ID` and `GCP_PROJECT_ID`.
- [x] **Task 2.3**: Update `v2/setup/step_02_create_index.py` to print dynamic environment variable suggestion instructions on terminal output when finished.
- [x] **Task 2.4**: Create `v2/setup/.env.example` as a template for user execution setups.
- [x] **Task 2.5**: Migrate ingestion model from developer `GoogleGenerativeAIEmbeddings` to native `VertexAIEmbeddings` using GCP-level Application Default Credentials (ADC), bypassing the AI Studio blocked API endpoints.

---

## Phase 3: Backend API Service (Completed)
- [x] **Task 3.1**: Create `v2/backend/config.py` mirroring `v2/setup/config.py` with fallback checks to read from the setup's `.rag_config.json` configuration file during local testing.
- [x] **Task 3.2**: Create `v2/backend/query.py`. Port and refactor retriever and LCEL chain code to run natively under Vertex AI.
- [x] **Task 3.3**: Create `v2/backend/main.py` exposing lightweight GET `/health` container probes and POST `/query` API endpoints.
- [x] **Task 3.4**: Create `v2/backend/requirements.txt`.
- [x] **Task 3.5**: Create `v2/backend/Dockerfile` using lightweight Python slim and uvicorn binding on dynamic `$PORT` environments.

---

## Phase 4: Frontend Web Interface (Completed)
- [x] **Task 4.1**: Create `v2/frontend/app.py` using Streamlit.
- [x] **Task 4.2**: Create `v2/frontend/requirements.txt` ensuring **ZERO** GCP SDK dependencies (Only `streamlit`, `requests`, `python-dotenv`).
- [x] **Task 4.3**: Create `v2/frontend/Dockerfile` and entrypoint shell command configurations.
- [x] **Task 4.4**: Create `v2/frontend/theme.py` implementing stunning customized "Medi Bee Assist - by AIBees" branding with custom base64 SVG user & bot chat avatars.

---

## Phase 5: Documentation & Manual Testing (Completed)
- [x] **Task 5.1**: Write a comprehensive `v2/README.md` with dynamic visual directory tree, step-by-step handoff guide, and zero-key Vertex AI deployment guidelines.
