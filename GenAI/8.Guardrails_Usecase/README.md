# 🔐 RAG + PII Guardrails Demo

A Streamlit app showing the same RAG pipeline with guardrails **OFF** (raw PII flows
straight through) and **ON** (PII is detected and redacted before it reaches the screen).

---

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then in the sidebar: pick a document → **Prepare** → toggle **Enable PII guardrails** →
click a sample query.

---

## Backends

One environment variable, `RAG_BACKEND`, chooses how answers are produced. Nothing else
in the code changes.

| `RAG_BACKEND` | Retrieval | Answers | Requires |
|---|---|---|---|
| `vertexai` *(default)* | FAISS + keyword (hybrid) | Gemini on Vertex AI | `gcloud auth application-default login`, a GCP project |
| `ollama` | FAISS + keyword (hybrid) | Local model via Ollama | `ollama serve`, `llama3.2`, `nomic-embed-text` |
| `keyword` | Keyword only | Extractive — quotes matching lines, no LLM | Nothing |

`keyword` needs no credentials and no network, which makes it the safe fallback for
classrooms and offline demos.

Vertex AI setup:

```bash
gcloud auth application-default login
```

Ollama setup:

```bash
ollama pull llama3.2 && ollama pull nomic-embed-text
```

Verify Ollama is reachable before switching the app over:

```bash
python examples/ollama_smoke_test.py
```

---

## Configuration

All settings live in `.env` (see the file for the full list). The ones you are most
likely to touch:

| Variable | Default | Meaning |
|---|---|---|
| `RAG_BACKEND` | `vertexai` | `vertexai` / `ollama` / `keyword` |
| `GCP_PROJECT_ID` | — | Required when `RAG_BACKEND=vertexai` |
| `GEMINI_MODEL` | `gemini-2.5-pro` | Vertex AI model id |
| `OLLAMA_MODEL` | `llama3.2:latest` | Local model id |
| `PDF_PATH` | `customer_credit_data_100_records.pdf` | Pre-selected document |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | Splitter settings |
| `RETRIEVER_TOP_K` | `4` | Chunks pulled per retriever before fusion |

Missing or contradictory configuration is reported in the UI at startup rather than
crashing mid-query.

---

## Architecture

```
app.py           ← Streamlit UI: session state and layout only
├── ui_theme.py     ← CSS and small render helpers
├── config.py       ← env-driven Settings, parsed once
├── pdf_processor.py← loading, chunking, content fingerprinting
├── retrieval.py    ← KeywordRetriever / VectorRetriever / HybridRetriever
├── generation.py   ← ExtractiveGenerator / LLMGenerator + the grounded prompt
├── guardrails.py   ← PII detection and redaction
└── rag_system.py   ← wires a retriever to a generator, manages the index
```

Retrieval and generation are independent axes: any retriever works with any generator,
and `RAG_BACKEND` just picks the pair.

### Why hybrid retrieval

Dense vectors handle paraphrases but routinely miss exact identifiers — asking for
"Customer Profile #4" retrieves generically similar records instead of profile 4.
`HybridRetriever` runs both the vector and keyword retrievers and merges them with
reciprocal rank fusion, so exact-ID lookups and semantic questions both work.

### Index caching

Indexes are keyed by a hash of the document's *contents*, under
`faiss_index/<backend>-<hash>/`. Re-selecting the same document — or the same file from a
different folder — reuses the existing index instead of re-embedding it. **Rebuild**
forces a fresh build.

---

## Guardrails

Two layers, both visible in the UI:

1. **Input** — the question is scanned for PII-seeking terms and flagged with a 🚩 pill.
   The query still runs, so the redaction can be demonstrated.
2. **Output** — the answer is scanned and every recognised value is replaced with a
   token, with per-type counts shown under the message.

Recognised: credit cards, ZIP codes, SSNs, phone numbers, emails, account numbers,
dates of birth, and street addresses.

All rules compile into a **single** alternation regex, so redaction is one linear pass
over the text regardless of how many rules exist. Earlier rules win ties, which is why
credit cards precede SSNs (both are digit runs) and street addresses precede ZIP codes
(a house number can be five digits).

### Adding a rule

Append to `RULES` in `guardrails.py`:

```python
Rule("Passport", r"\b[A-Z]{1,2}\d{6,9}\b", "[PASSPORT REDACTED]"),
```

Position it by specificity. Labels are free-form and may repeat, so one PII type can span
several patterns. Nothing else needs to change.

---

## Sample queries

| Intent | Query | Expected with guardrails ON |
|---|---|---|
| SSN leak | `Share the SSN of a customer with a good credit score.` | Model refuses (prompt-level guardrail) |
| Phone leak | `What is the mobile phone number of Customer Profile #4?` | `[PHONE REDACTED]` |
| Account leak | `What is the account number of the first customer?` | `[ACCOUNT REDACTED]` |
| Multi-field PII | `What is the full legal name and date of birth of Customer Profile #4?` | Name shown, `[DOB REDACTED]` |

The bundled datasets are entirely synthetic.

---

## examples/

Standalone reference scripts, not part of the app:

- `ollama_smoke_test.py` — checks that a local Ollama model responds.
- `standalone_ollama_rag.py` — the whole pipeline in one file, without guardrails.
