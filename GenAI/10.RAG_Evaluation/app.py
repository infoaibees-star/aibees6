# AIBees Academy — Knowledge Assistant  (INCREMENTAL RAG: Vertex AI + FAISS)
# Each PDF is merged into ONE shared FAISS index (the "knowledge base"),
# with a registry that skips files already ingested. Styling lives in theme.py.
#
# This version adds two evaluation tabs on top of the chat:
#   • RAGAS    — retrieval quality (needs a ground-truth answer)
#   • DeepEval — safety & hallucination checks (ground truth optional)
# Both evaluators use the SAME Vertex AI models configured below.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows OpenMP fix (FAISS + eval libs). Harmless on Linux/Mac.

import json
import shutil
import hashlib
import asyncio
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import theme  # <- our branding file (unchanged)

# Import our two simple evaluation files
from ragas_eval import run_ragas_evaluation, display_ragas_scores
from deepeval_eval import run_deepeval_evaluation, display_deepeval_scores

# Streamlit runs each script in a thread without an event loop; make sure one exists
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()
PROJECT  = os.getenv("GCP_PROJECT_ID")
LOCATION = "us-central1"

# One SHARED index for all documents, plus a registry of what's already in it
FAISS_DIR      = "faiss_store"
COMBINED_INDEX = os.path.join(FAISS_DIR, "combined_index")
REGISTRY_FILE  = os.path.join(FAISS_DIR, "registry.json")
os.makedirs(FAISS_DIR, exist_ok=True)

# ── Models (Vertex AI) ────────────────────────────────────────────────
embeddings = VertexAIEmbeddings(
    model_name="gemini-embedding-001",
    project=PROJECT,
    location=LOCATION,
)

llm = ChatVertexAI(
    model_name="gemini-2.5-pro",
    project=PROJECT,
    location=LOCATION,
    temperature=0.3,
)

# ── Registry helpers (which PDFs are already in the knowledge base) ────
def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_registry(registry: dict) -> None:
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

# ── RAG helpers ───────────────────────────────────────────────────────
def read_pdf(pdf_file) -> str:
    """Read all text (paragraphs + table) from the uploaded PDF."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def ingest_pdf(pdf_file) -> tuple[bool, str]:
    """
    INCREMENTAL ingestion:
    extract -> fingerprint -> skip if duplicate -> chunk -> embed ->
    MERGE into the one shared FAISS index.
    """
    registry = load_registry()
    text     = read_pdf(pdf_file)
    fingerprint = hashlib.md5(text.encode()).hexdigest()   # content fingerprint

    # Deduplicate: same content already ingested? -> do nothing
    if fingerprint in registry:
        return False, f"⚠️ **{pdf_file.name}** is already in the knowledge base. Skipped."

    # 1) SPLIT into overlapping chunks, 2) EMBED into a small new index
    chunks    = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(text)
    new_store = FAISS.from_texts(chunks, embeddings)

    # 3) MERGE the new index into the shared one (this is the "incremental" bit)
    if os.path.exists(COMBINED_INDEX):
        combined = FAISS.load_local(COMBINED_INDEX, embeddings,
                                    allow_dangerous_deserialization=True)
        combined.merge_from(new_store)        # ← add new vectors to the existing KB
    else:
        combined = new_store                  # first document creates the KB

    combined.save_local(COMBINED_INDEX)
    registry[fingerprint] = pdf_file.name
    save_registry(registry)
    return True, f"✅ **{pdf_file.name}** ingested — {len(chunks)} chunks added."


def load_combined_index():
    """Load the shared knowledge base if it exists yet."""
    if os.path.exists(COMBINED_INDEX):
        return FAISS.load_local(COMBINED_INDEX, embeddings,
                                allow_dangerous_deserialization=True)
    return None


def build_messages(context: str, history: list[dict], question: str) -> list[dict]:
    """System instruction + past turns (memory) + current question with context."""
    system = {"role": "system", "content": (
        "You are the AIBees Academy Knowledge Assistant. "
        "Use the DOCUMENT CONTEXT to answer questions, and if a factual answer is "
        "not in the context, say so honestly. "
        "You can also refer to the earlier conversation when the user asks about it "
        "(for example, what they asked before)."
    )}
    past = []
    for turn in history:
        past.append({"role": "user",      "content": turn["question"]})
        past.append({"role": "assistant", "content": turn["answer"]})
    user = {"role": "user", "content": (
        f"--- Document Context ---\n{context}\n--- End Context ---\n\n"
        f"Question: {question}"
    )}
    return [system] + past + [user]


# ── Page setup ────────────────────────────────────────────────────────
st.set_page_config(page_title="AIBees Knowledge Assistant", page_icon="🐝", layout="centered")
theme.inject_css()
theme.render_header()

# ── Session state ─────────────────────────────────────────────────────
st.session_state.setdefault("history", [])
st.session_state.setdefault("db", load_combined_index())   # load KB on startup
# Track the most recent turn so the evaluation tabs know what to score
st.session_state.setdefault("last_question", "")
st.session_state.setdefault("last_answer", "")
st.session_state.setdefault("last_chunks", [])

# ── Sidebar: ingest + knowledge base + controls ───────────────────────
with st.sidebar:
    theme.render_sidebar_logo()

    st.markdown("### 📄 Upload Documents")
    st.caption("Each PDF is merged into the shared knowledge base.")
    uploaded = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if uploaded and st.button("➕ Ingest into Knowledge Base"):
        with st.spinner("🐝 Embedding and merging…"):
            ok, message = ingest_pdf(uploaded)
        st.markdown(message)
        if ok:
            st.session_state.db = load_combined_index()   # refresh with new vectors

    st.markdown("### 📚 Knowledge Base")
    registry = load_registry()
    if registry:
        for fname in registry.values():
            st.markdown(f"📄 `{fname}`")
        st.caption(f"{len(registry)} document(s) loaded")
    else:
        st.caption("No documents ingested yet.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Reset KB"):
            if os.path.exists(COMBINED_INDEX):
                shutil.rmtree(COMBINED_INDEX)
            if os.path.exists(REGISTRY_FILE):
                os.remove(REGISTRY_FILE)
            st.session_state.db = None
            st.session_state.history = []
            st.session_state.last_question = ""
            st.session_state.last_answer = ""
            st.session_state.last_chunks = []
            st.rerun()
    with col2:
        if st.button("💬 Clear Chat"):
            st.session_state.history = []
            st.rerun()

    st.markdown("### 💡 Try asking")
    for q in [
        "How do I reset my password?",
        "What are the VPN setup steps?",
        "What is the deployment window?",
        "How are databases backed up?",
        "How do I request AWS access?",
    ]:
        st.markdown(f"› {q}")

# ── Main area ─────────────────────────────────────────────────────────
if not st.session_state.db:
    st.info("Upload a PDF and click **Ingest** in the sidebar to start asking questions.")
    st.stop()

# ── 3 Tabs: Chat | RAGAS | DeepEval ───────────────────────────────────
tab_chat, tab_ragas, tab_deepeval = st.tabs([
    "💬 Chat",
    "📊 RAGAS Evaluation",
    "🛡️ DeepEval Evaluation",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════
with tab_chat:

    # Replay the conversation so far
    for turn in st.session_state.history:
        st.chat_message("user", avatar=theme.USER_AVATAR).markdown(turn["question"])
        st.chat_message("assistant", avatar=theme.BOT_AVATAR).markdown(turn["answer"])

    # Handle a new question
    query = st.chat_input("Ask the AIBees Knowledge Assistant anything…")
    if query:
        st.chat_message("user", avatar=theme.USER_AVATAR).markdown(query)

        # RETRIEVE — top-4 chunks from the SHARED knowledge base (across all PDFs)
        chunks = st.session_state.db.similarity_search(query, k=4)
        context = "\n\n".join(c.page_content for c in chunks)

        # GENERATE — send only those chunks (not whole documents) to the LLM
        messages = build_messages(context, st.session_state.history, query)
        with st.chat_message("assistant", avatar=theme.BOT_AVATAR):
            with st.spinner("🐝 Buzzing through the knowledge base…"):
                answer = llm.invoke(messages).content.strip()
            st.markdown(answer)

        # Remember this turn so the evaluation tabs can score it
        st.session_state.last_question = query
        st.session_state.last_answer   = answer
        st.session_state.last_chunks   = [c.page_content for c in chunks]

        st.session_state.history.append({"question": query, "answer": answer})


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — RAGAS EVALUATION  (retrieval quality)
# ══════════════════════════════════════════════════════════════════════
with tab_ragas:

    st.markdown("### 📊 RAGAS — Retrieval Quality Evaluation")
    st.markdown(
        "RAGAS checks how well the **retrieval step** of your RAG pipeline is working. "
        "It needs the question, the assistant's answer, the retrieved chunks, and a "
        "ground-truth answer."
    )

    if st.session_state.last_question:
        st.info(f"**Last question:** {st.session_state.last_question}")
        st.info(f"**Last answer:** {st.session_state.last_answer[:200]}...")
    else:
        st.warning("Ask a question in the Chat tab first, then come here to evaluate it.")
        st.stop()

    ground_truth = st.text_area(
        "✍️ Enter the expected correct answer (ground truth):",
        placeholder="e.g. To reset your password, go to the IT portal and click Forgot Password...",
        height=100,
    )

    if st.button("▶️ Run RAGAS Evaluation", type="primary"):
        if not ground_truth.strip():
            st.warning("Please enter a ground truth answer before running RAGAS.")
        else:
            with st.spinner("⏳ Running RAGAS metrics... this may take 30-60 seconds"):
                scores = run_ragas_evaluation(
                    question         = st.session_state.last_question,
                    answer           = st.session_state.last_answer,
                    retrieved_chunks = st.session_state.last_chunks,
                    ground_truth     = ground_truth,
                    llm              = llm,
                    embedding_model  = embeddings,
                )
            display_ragas_scores(scores)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — DEEPEVAL EVALUATION  (safety & hallucination)
# ══════════════════════════════════════════════════════════════════════
with tab_deepeval:

    st.markdown("### 🛡️ DeepEval — Safety & Hallucination Checks")
    st.markdown(
        "DeepEval checks the **quality and safety** of the assistant's response. "
        "It does NOT need a ground truth — it evaluates the answer directly using "
        "the retrieved chunks."
    )

    if st.session_state.last_question:
        st.info(f"**Last question:** {st.session_state.last_question}")
        st.info(f"**Last answer:** {st.session_state.last_answer[:200]}...")
    else:
        st.warning("Ask a question in the Chat tab first, then come here to evaluate it.")
        st.stop()

    ground_truth_optional = st.text_area(
        "✍️ Optional: Enter expected answer (helps with recall metric):",
        placeholder="Leave blank if you just want hallucination / toxicity / bias checks",
        height=80,
    )

    if st.button("▶️ Run DeepEval Evaluation", type="primary"):
        with st.spinner("⏳ Running DeepEval checks... this may take 30-60 seconds"):
            scores = run_deepeval_evaluation(
                question         = st.session_state.last_question,
                answer           = st.session_state.last_answer,
                retrieved_chunks = st.session_state.last_chunks,
                ground_truth     = ground_truth_optional,
            )
        display_deepeval_scores(scores)