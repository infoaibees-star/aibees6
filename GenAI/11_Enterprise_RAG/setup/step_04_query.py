#!/usr/bin/env python3
"""
step_04_query.py
────────────────
STEP 4 — Query the RAG pipeline interactively or via CLI.

Reads GOOGLE_API_KEY from a .env file in the same directory.
Vector Search and LLM run on Vertex AI; embeddings at query
time use the Gemini Developer API via GoogleGenerativeAIEmbeddings.

Setup:
    cp .env.example .env
    # add your key to .env
    pip install python-dotenv

Run (interactive REPL):
    python step_04_query.py

Run (single question):
    python step_04_query.py --question "What is the dosage for Metformin?"

Options:
    --question   TEXT    Single question (non-interactive)
    --top-k      INT     Number of chunks to retrieve (default from config)
    --verbose            Print full retrieved source chunks
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
import os
import sys
from pathlib import Path

# ── Load .env before any other imports ─────────────────────────────────────────
from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# ── GCP + LangChain imports ───────────────────────────────────────────────────
from google.cloud import aiplatform
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_vertexai import ChatVertexAI, VectorSearchVectorStore, VertexAIEmbeddings

from config import (
    PROJECT_ID, REGION,
    EMBED_BUCKET, EMBED_BUCKET_URI,
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    RETRIEVER_TOP_K, RAG_CONFIG_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

EMBEDDING_MODEL = "models/gemini-embedding-001"


# ── Prompt ────────────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a precise medical information assistant.
Answer only using the context provided below.
If the answer is not in the context, say:
"I don't have enough information in the knowledge base to answer this."

Context:
{context}

Question: {question}

Answer:""")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    index_id = os.getenv("INDEX_ID")
    endpoint_id = os.getenv("ENDPOINT_ID")
    if index_id and endpoint_id:
        return {"index_id": index_id, "endpoint_id": endpoint_id}
    if not Path(RAG_CONFIG_FILE).exists():
        log.error("%s not found and INDEX_ID/ENDPOINT_ID env vars not set. Run step_02_create_index.py first.", RAG_CONFIG_FILE)
        sys.exit(1)
    with open(RAG_CONFIG_FILE) as f:
        return json.load(f)


def format_docs(docs: list) -> str:
    """Concatenate retrieved chunks into a single context string."""
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_file', 'unknown')}"
        + (f", page {d.metadata.get('page')}" if d.metadata.get("page") is not None else "")
        + f"]\n{d.page_content}"
        for d in docs
    )


def build_chain(config: dict, top_k: int):
    # Embedder — VertexAIEmbeddings uses ADC
    log.info("Initialising Vertex AI embedding model: gemini-embedding-001")
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
        index_id=config["index_id"],
        endpoint_id=config["endpoint_id"],
        embedding=embedder,
        stream_update=True,
    )

    log.info("Loading LLM: %s", LLM_MODEL)
    llm = ChatVertexAI(
        model_name=LLM_MODEL,
        project=PROJECT_ID,
        location=REGION,
        max_output_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # Pure LCEL chain — no langchain.chains dependency
    rag_chain = (
        RunnableParallel({
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    log.info("✓  RAG chain ready (top_k=%d)", top_k)

    def invoke(question: str) -> dict:
        docs   = retriever.invoke(question)
        answer = rag_chain.invoke(question)
        return {"answer": answer, "sources": docs}

    return invoke


# ── Query execution ───────────────────────────────────────────────────────────

def run_query(chain, question: str, verbose: bool) -> None:
    print("\n" + "─" * 60)
    print(f"  Question : {question}")
    print("─" * 60)

    result  = chain(question)
    answer  = result.get("answer", "No answer returned.")
    sources = result.get("sources", [])

    print(f"\n  Answer:\n  {answer}\n")

    if verbose and sources:
        print("  Retrieved source chunks:")
        for i, doc in enumerate(sources, 1):
            meta = doc.metadata
            print(f"\n  [{i}] {meta.get('source_file', 'unknown')}"
                  + (f" — page {meta.get('page')}" if meta.get("page") is not None else ""))
            print(f"      GCS : {meta.get('source_gcs', '')}")
            snippet = doc.page_content[:250].replace("\n", " ")
            print(f"      Text: {snippet} …")
    elif sources:
        print("  Sources:")
        seen = set()
        for doc in sources:
            fname = doc.metadata.get("source_file", "unknown")
            page  = doc.metadata.get("page")
            key   = (fname, page)
            if key not in seen:
                seen.add(key)
                page_str = f" (page {page})" if page is not None else ""
                print(f"    • {fname}{page_str}")

    print("─" * 60 + "\n")


# ── Interactive REPL ──────────────────────────────────────────────────────────

def repl(chain, verbose: bool) -> None:
    print("\n" + "=" * 60)
    print("  Enterprise RAG — Medical Knowledge Base")
    print("  Type your question and press Enter.")
    print("  Commands:  :quit   :verbose   :help")
    print("=" * 60)

    while True:
        try:
            raw = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue
        if raw in (":quit", ":q", "exit", "quit"):
            print("Goodbye.")
            break
        if raw == ":verbose":
            verbose = not verbose
            print(f"  Verbose: {'ON' if verbose else 'OFF'}")
            continue
        if raw == ":help":
            print("  :quit    — exit")
            print("  :verbose — toggle full chunk display")
            continue

        try:
            run_query(chain, raw, verbose)
        except Exception as exc:
            log.error("Query failed: %s", exc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4 — Query the RAG pipeline")
    parser.add_argument("--question", "-q", type=str, default=None)
    parser.add_argument("--top-k",    "-k", type=int, default=RETRIEVER_TOP_K)
    parser.add_argument("--verbose",  "-v", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 55)
    log.info("STEP 4 — RAG Query")
    log.info("LLM    : %s", LLM_MODEL)
    log.info("Top-K  : %d", args.top_k)
    log.info("=" * 55)

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=EMBED_BUCKET_URI)

    config = load_config()
    chain  = build_chain(config, top_k=args.top_k)

    if args.question:
        run_query(chain, args.question, args.verbose)
    else:
        repl(chain, verbose=args.verbose)