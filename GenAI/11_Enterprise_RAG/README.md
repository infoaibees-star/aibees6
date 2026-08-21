# Enterprise Medical RAG v2: Deployable 3-Service Pipeline

This directory contains the production-ready, deployable version of the Vertex AI Vector Search RAG pipeline, refactored into three modular and decoupled components.

---

## 1. Directory Layout

```text
11_Enterprise_RAG/
├── spec.md                     # Design specification document
├── plan.md                     # Implementation task tracker
├── README.md                   # This deployment & runtime operator guide
├── setup/                      # Local scripts (ran in notebook/terminal, never containerized)
│   ├── config.py, .env.example # Local configs & variables loading
│   ├── step_01_setup_gcs.py    # Standard GCS bucket creator
│   ├── step_02_create_index.py # Deploys MatchingEngineIndex & Endpoint
│   ├── step_03_ingest.py       # Decoupled Recursive splitting chunk ingestion
│   ├── step_04_query.py        # CLI diagnostic query pipeline
│   └── step_05_cleanup.py      # GCP resource teardown utility
├── backend/                    # Containerized FastAPI REST API (Cloud Run Service #1)
│   ├── main.py, query.py       # Exposes GET /health & POST /query routes
│   ├── config.py               # Self-contained configuration loader
│   ├── requirements.txt        # Server dependencies (aiplatform, fastapi, etc.)
│   └── Dockerfile              # Production docker recipe running Uvicorn
└── frontend/                   # Containerized Streamlit UI (Cloud Run Service #2)
    ├── app.py, theme.py        # Brand-aligned "Medi Bee Assist" chat layout
    ├── requirements.txt        # Client-only dependencies (Zero google-cloud-* packages)
    ├── Dockerfile              # Streamlit container configuration running on dynamic $PORT
    └── .streamlit/config.toml  # Headless Streamlit parameters placeholder
```

---

## 2. Environment Variables Reference

## 2.1 - Please go to setup\config.py and do the changes.. Below values are just a sample
| Variable Name | Purpose | Required By | Example Value |
| :--- | :--- | :---: | :--- |
| `PROJECT_ID` | GCP Project ID | Setup, Backend | `project-f1be597b` |
| `REGION` | GCP Resource Region | Setup, Backend | `us-central1` |
| `SOURCE_BUCKET`| GCS Bucket for raw PDFs | Setup | `XXXX-pdfs` |
| `PDF_PREFIX` | Directory inside SOURCE_BUCKET | Setup | `pdfs/` |
| `EMBED_BUCKET` | GCS Bucket for vector index artefacts | Setup, Backend | `enddding_folder_{name}` |
| `INDEX_ID` | Resource name / ID of the Vector index | Backend, Setup | `111111` (Auto-generated in Step 2) |
| `ENDPOINT_ID` | Resource name / ID of the index Endpoint | Backend, Setup | `48292222210485729` (Auto-generated in Step 2) |
| `BACKEND_URL` | Base HTTP endpoint of Backend service | Frontend | `https://backend-service-abc-uc.a.run.app` |
| `PORT` | Local or Cloud Run container network port | Backend, Frontend | `8080` (Default) |

---

## 3. Deployment Workflow & Setup Handoff

### Step 1: Run the Setup Pipeline
Navigate to `11_Enterprise_RAG/setup/` and set up your local workspace environment:
```bash
cd 11_Enterprise_RAG/setup
cp .env.example .env
# Edit .env and supply your PROJECT_ID and other options
```

Create a virtual environment, install requirements, and run the pipeline:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create buckets and upload raw PDFs
python step_01_setup_gcs.py

# Create Vector Index & Deploy Endpoint (Takes ~30–45 mins. Run once.)
python step_02_create_index.py
```

Upon success, `step_02_create_index.py` saves output identifiers to `.rag_config.json` and prints the **Handoff Environment Variables** to the console:
```text
=======================================================
ENDPOINT HANDOFF INFORMATION
Set the following environment variables for your backend service: This should match with your project's configuration
  export INDEX_ID=1234567890123456789
  export ENDPOINT_ID=9876543210987654321
=======================================================
```

Run Ingestion (full or incremental):
```bash
# Upload your PDF guides to gs://<SOURCE_BUCKET>/pdfs/
python step_03_ingest.py --mode full
```

*(Optional)* Run CLI query tool to verify your search endpoint:
```bash
python step_04_query.py --question "Who should use Aspirin?"
```

---



# local run without Docker

# backend:


   1 cd ~/11_Enterprise_RAG/backend {please replace with your actual 'backend' path}
   
   2 python3 -m venv .venv && source .venv/bin/activate
   3 pip install -r requirements.txt
   4 uvicorn main:app --host 0.0.0.0 --port 8080


# Frontend:

Run and Test the Completed app_v2.py Locally:

  Open a JupyterLab Terminal tab and launch your new frontend:

   1 cd ~/11_Enterprise_RAG/frontend {please replace with your actual frontend path}
   
   2 python3 -m venv .venv && source .venv/bin/activate
   3 pip install -r requirements.txt
   4
   5 # Point to your local FastAPI backend and run app_v2.py
   6 export BACKEND_URL="http://127.0.0.1:8080"
   7 streamlit run app_v2.py --server.port=8501 --server.address=0.0.0.0

  Open the App in Your Browser:
   1. Look at your current JupyterLab browser URL:
     https://[some-id]-dot-[region].notebooks.googleusercontent.com/lab/tree/...
   2. Replace /lab (and everything after it) with /proxy/8501/ (including the trailing slash /):
      - Modified URL format: 
       https://[some-id]-dot-[region].notebooks.googleusercontent.com/proxy/8501/

  Try the Conversational Memory Flow:
   1. Ask: "Who should use Insulin ?"
   2. Ask: "what about Aspirin ?"
   3. Ask: "can you compare these 2 and prepare a table ?"

# Deployment 


### Step 2: Deploy Backend API Service (Cloud Run Service #1)
The Backend Service runs FastAPI, Uvicorn, and uses **Application Default Credentials (ADC)** to connect to Vertex AI + GCS (using VertexAIEmbeddings and ChatVertexAI). **No service-account key files or external API keys are baked inside or required in the environment.**

Deploy to Cloud Run via `gcloud run deploy`:
```bash
cd ../backend

gcloud run deploy enterprise-rag-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="PROJECT_ID=project-f1be597b-8476-45ad-a2b,REGION=us-central1,EMBED_BUCKET=aib-embeddings-0821,INDEX_ID=YOUR_INDEX_ID_HERE,ENDPOINT_ID=YOUR_ENDPOINT_ID_HERE"
```

Once deployment completes, note down the generated service URL (e.g., `https://enterprise-rag-backend-xyz-uc.a.run.app`).

#### Verify the Backend:
```bash
# Test lightweight health check probe
curl https://<backend-url>/health
# Response: {"status": "healthy"}

# Test POST query endpoint
curl -X POST https://<backend-url>/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who should use Aspirin?", "top_k": 3}'
```

---

### Step 3: Deploy Frontend Web Interface (Cloud Run Service #2)
The Frontend Service runs Streamlit, binds to dynamic `$PORT`, and communicates with the backend purely via HTTP JSON payloads. It has zero GCP SDK footprint.

Deploy to Cloud Run via `gcloud run deploy`:
```bash
cd ../frontend

gcloud run deploy enterprise-rag-frontend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="BACKEND_URL=https://enterprise-rag-backend-xyz-uc.a.run.app"
```

Open the generated web URL in your browser to access the interactive medical RAG assistant!

---

## 4. Run Services Locally with Docker

To test the containerization setup locally before deploying, you can spin up the services using standard docker commands:

### Run Backend Locally
```bash
cd 11_Enterprise_RAG/backend
docker build -t enterprise-rag-backend .

docker run -p 8080:8080 \
  -e INDEX_ID="YOUR_INDEX_ID" \
  -e ENDPOINT_ID="YOUR_ENDPOINT_ID" \
  enterprise-rag-backend
```

### Run Frontend Locally
```bash
cd 11_Enterprise_RAG/frontend
docker build -t enterprise-rag-frontend .

docker run -p 8501:8501 \
  -e BACKEND_URL="http://localhost:8080" \
  enterprise-rag-frontend
```
And access it at `http://localhost:8501`.



# Deployment with live parameters

  Step 1: Deploy the Backend API Service
  First, we deploy your FastAPI backend so it can handle database vector searches securely:

   1 # 1. Navigate to your backend directory
   2 cd ~/11_Enterprise_RAG/backend
   3
   4 # 2. Deploy to Cloud Run using your active configurations
   5 gcloud run deploy enterprise-rag-backend \
   6   --source . \
   7   --region us-central1 \
   8   --allow-unauthenticated \
   9   --set-env-vars="GCP_PROJECT_ID=project-f1be597b-8476-45ad-a2b,GCP_REGION=us-central1,EMBED_BUCKET=aib-embeddings-0821,INDEX_ID=2162754764998180864,ENDPOINT_ID=1517143527400669184"
  When the deployment finishes, the terminal will output a Service URL (e.g., https://enterprise-rag-backend-abc-uc.a.run.app). Copy this URL as you will need it for the frontend.

  ---

  Step 2: Deploy the "Medi Bee Assist" Frontend
  Next, deploy your polished Streamlit frontend and point it directly to your newly deployed backend URL:

   1 # 1. Navigate to your frontend directory
   2 cd ~/11_Enterprise_RAG/frontend
   3
   4 # 2. Deploy to Cloud Run, pasting your Backend Service URL below
   5 gcloud run deploy medi-bee-frontend \
   6   --source . \
   7   --region us-central1 \
   8   --allow-unauthenticated \
   9   --set-env-vars="BACKEND_URL=https://enterprise-rag-backend-607627010247.us-central1.run.app"
  (Replace [YOUR_BACKEND_CLOUD_RUN_URL_FROM_STEP_1] with your real backend URL).


