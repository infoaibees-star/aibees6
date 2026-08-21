# AIBees Academy — IT Helpdesk Assistant  (RAG: Vertex AI + FAISS)
# All styling lives in theme.py, so this file is only the RAG logic.
import os
import hashlib
import asyncio
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import theme  # <- our branding file

# Streamlit runs each script in a thread without an event loop; make sure one exists
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()
PROJECT  = os.getenv("GCP_PROJECT_ID")
LOCATION = "us-central1"
FAISS_DIR = "faiss_store"
os.makedirs(FAISS_DIR, exist_ok=True)

# ── Models (Vertex AI) ────────────────────────────────────────────────
embeddings = VertexAIEmbeddings(
    model_name="gemini-embedding-001",   # turns text into vectors for search
    project=os.getenv("GCP_PROJECT_ID"),
    location="us-central1",
)

llm = ChatVertexAI(
    model_name="gemini-2.5-pro",
    project=os.getenv("GCP_PROJECT_ID"),
    location="us-central1",
    temperature=0.3,
)

# ── RAG helpers ───────────────────────────────────────────────────────
def read_pdf(pdf_file) -> str:
    """Read all text (paragraphs + table) from the uploaded PDF."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def get_vectorstore(pdf_file) -> FAISS:
    """Build (or load) a FAISS index for this PDF, cached by its content."""
    text = read_pdf(pdf_file)
    index_path = os.path.join(FAISS_DIR, hashlib.md5(text.encode()).hexdigest())

    if not os.path.exists(index_path):
        # 1) SPLIT the document into small overlapping chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(text)
        # 2) EMBED each chunk and store the vectors in FAISS, then save to disk
        with st.spinner("Indexing document (happens only once)…"):
            FAISS.from_texts(chunks, embeddings).save_local(index_path)

    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def build_messages(context: str, history: list[dict], question: str) -> list[dict]:
    """System instruction + past turns (memory) + current question with context."""
    system = {"role": "system", "content": (
        "You are the AIBees IT Helpdesk Assistant. "
        "Answer using ONLY the document context below. "
        "If the answer is not in the context, say so honestly."
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
st.set_page_config(page_title="AIBees Knowledge Assistant - 8th Aug 2026", page_icon="🐝", layout="centered")
theme.inject_css()
theme.render_header()

# ── Session state ─────────────────────────────────────────────────────
st.session_state.setdefault("history", [])
st.session_state.setdefault("db", None)
st.session_state.setdefault("pdf_name", None)

# ── Sidebar: upload + controls ────────────────────────────────────────
with st.sidebar:
    theme.render_sidebar_logo()
    st.markdown("### 📄 Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if uploaded and uploaded.name != st.session_state.pdf_name:
        st.session_state.db = get_vectorstore(uploaded)      # index the new file
        st.session_state.pdf_name = uploaded.name
        st.session_state.history = []
        st.success(f"Loaded: {uploaded.name}")

    if st.session_state.pdf_name:
        st.markdown(f"Active: `{st.session_state.pdf_name}`")

    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.rerun()

    st.markdown("### 💡 Try asking")
    for q in [
        "How do I reset my password?",
        "What are the VPN setup steps?",
        "How do I report a phishing email?",
        "What is the laptop replacement process?",
        "What Wi-Fi networks are available?",
        "What happens to my access when I leave?"
        ]:
        st.markdown(f"› {q}")

# ── Main area ─────────────────────────────────────────────────────────
if not st.session_state.db:
    st.info("Upload a PDF from the sidebar to start asking questions.")
    st.stop()

# Replay the conversation so far
for turn in st.session_state.history:
    st.chat_message("user", avatar=theme.USER_AVATAR).markdown(turn["question"])
    st.chat_message("assistant", avatar=theme.BOT_AVATAR).markdown(turn["answer"])

# Handle a new question
query = st.chat_input("Ask a question about the document…")
if query:
    st.chat_message("user", avatar=theme.USER_AVATAR).markdown(query)

    # RETRIEVE — find the 4 chunks most similar to the question (the "R" in RAG)
    chunks = st.session_state.db.similarity_search(query, k=4)
    context = "\n\n".join(c.page_content for c in chunks)

    # GENERATE — send ONLY those chunks (not the whole document) to the LLM
    messages = build_messages(context, st.session_state.history, query)
    with st.chat_message("assistant", avatar=theme.BOT_AVATAR):
        with st.spinner("Thinking…"):
            answer = llm.invoke(messages).content.strip()
        st.markdown(answer)

    st.session_state.history.append({"question": query, "answer": answer})
