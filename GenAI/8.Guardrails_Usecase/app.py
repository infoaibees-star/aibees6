"""
RAG + PII Guardrails Demo — Streamlit entry point.

This module is deliberately thin: it owns session state and layout only.
Retrieval lives in `retrieval.py`, generation in `generation.py`, orchestration
in `rag_system.py`, redaction in `guardrails.py`, and styling in `ui_theme.py`.
"""

from __future__ import annotations

import glob
import logging
import os

import streamlit as st

import ui_theme
from config import settings
from guardrails import PIIGuardrail
from pdf_processor import SUPPORTED_EXTENSIONS
from rag_system import RAGError, RAGSystem

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"

SAMPLE_QUERIES = (
    "Share the SSN of a customer with a good credit score.",
    "What is the mobile phone number of Customer Profile #4?",
    "What is the account number of the first customer?",
    "List customers with a credit score above 800.",
    "What is the full legal name and date of birth of Customer Profile #4?",
)

st.set_page_config(
    page_title="PII Guardrails Demo",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Cached singletons — built once, reused across Streamlit reruns
# ------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_guardrail() -> PIIGuardrail:
    return PIIGuardrail()


@st.cache_resource(show_spinner=False)
def _build_rag(cache_key: tuple[str, ...]) -> RAGSystem:
    return RAGSystem(settings)


def get_rag() -> RAGSystem:
    """Model clients are expensive to construct, so keep exactly one instance.

    The cache is keyed on the settings that determine the pipeline, so editing
    `.env` rebuilds it on the next rerun instead of serving a stale system.
    """
    return _build_rag(
        (
            settings.backend,
            settings.llm_model,
            settings.embedding_model,
            settings.ollama_model,
            settings.ollama_embedding_model,
        )
    )


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("document_path", None)
    st.session_state.setdefault("index_ready", False)
    st.session_state.setdefault("use_guardrails", False)
    st.session_state.setdefault("stats", {"queries": 0, "responses_redacted": 0, "values_redacted": 0})


def reset_stats() -> None:
    st.session_state.messages = []
    st.session_state.stats = {"queries": 0, "responses_redacted": 0, "values_redacted": 0}


# ------------------------------------------------------------------
# Actions
# ------------------------------------------------------------------


DOCUMENT_DIRS = (".", UPLOAD_DIR, "examples")


def local_documents() -> list[str]:
    """Documents already sitting next to the app, so the demo runs with zero uploads.

    The document named by PDF_PATH is listed first so it is pre-selected, matched
    by file name so it is found wherever it has been moved to.
    """
    found: list[str] = []
    for directory in DOCUMENT_DIRS:
        for ext in SUPPORTED_EXTENSIONS:
            found.extend(sorted(glob.glob(os.path.join(directory, f"*{ext}"))))
    found = [os.path.normpath(path) for path in found]

    default = os.path.basename(settings.default_document)
    preferred = next((p for p in found if os.path.basename(p) == default), None)
    if preferred:
        found.remove(preferred)
        found.insert(0, preferred)
    return found


def save_upload(uploaded) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    target = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(target, "wb") as handle:
        handle.write(uploaded.getbuffer())
    return target


def prepare_index(force_rebuild: bool) -> None:
    path = st.session_state.document_path
    if not path:
        st.error("Select or upload a document first.")
        return
    try:
        rag = get_rag()
        with st.spinner("Preparing index…"):
            rebuilt = rag.prepare(path, force_rebuild=force_rebuild)
        st.session_state.index_ready = True
        st.success(
            f"{'Built' if rebuilt else 'Reused cached'} index — "
            f"{rag.chunk_count} chunks via {rag.backend_name}."
        )
    except (RAGError, FileNotFoundError, ValueError) as exc:
        st.session_state.index_ready = False
        st.error(str(exc))
    except Exception as exc:
        st.session_state.index_ready = False
        logger.exception("Index preparation failed")
        st.error(f"Index preparation failed: {exc}")


def answer_question(question: str) -> dict:
    """Run the pipeline and apply output guardrails. Returns an assistant message."""
    guardrail = get_guardrail()
    stats = st.session_state.stats
    stats["queries"] += 1

    try:
        answer = get_rag().ask(question)
    except Exception as exc:
        logger.exception("Query failed")
        return {"role": "assistant", "content": f"⚠️ Could not generate a response: {exc}"}

    if not st.session_state.use_guardrails:
        return {"role": "assistant", "content": answer.text}

    result = guardrail.redact(answer.text)
    if result.was_modified:
        stats["responses_redacted"] += 1
        stats["values_redacted"] += result.total

    return {
        "role": "assistant",
        "content": result.redacted,
        "summary": result.summary if result.was_modified else None,
        "counts": result.counts,
    }


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("flagged"):
            ui_theme.pill("🚩 PII-seeking terms: " + ", ".join(message["flagged"]), kind="flag")
        if message.get("summary"):
            ui_theme.pill("⚠️ " + message["summary"])
            with st.expander("🔍 Redaction details"):
                st.json(message.get("counts", {}))


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🔐 PII Guardrails Demo")
        st.caption(f"Backend: `{settings.backend}`")

        st.markdown("### 📄 1. Choose data")
        available = local_documents()
        if available:
            current = st.session_state.document_path
            index = available.index(current) if current in available else 0
            chosen = st.selectbox("Document", available, index=index)
            if chosen != st.session_state.document_path:
                st.session_state.document_path = chosen
                st.session_state.index_ready = False

        uploaded = st.file_uploader("…or upload a PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded is not None:
            path = save_upload(uploaded)
            if path != st.session_state.document_path:
                st.session_state.document_path = path
                st.session_state.index_ready = False
                st.rerun()

        st.markdown("### ⚙️ 2. Build index")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Prepare", use_container_width=True):
                prepare_index(force_rebuild=False)
        with col_b:
            if st.button("Rebuild", use_container_width=True):
                prepare_index(force_rebuild=True)
        st.caption("Prepare reuses a cached index for the same file; Rebuild forces re-indexing.")

        st.markdown("### 🛡️ 3. Guardrails")
        st.session_state.use_guardrails = st.toggle(
            "Enable PII guardrails", value=st.session_state.use_guardrails
        )
        if st.session_state.use_guardrails:
            ui_theme.badge("🔒 GUARDRAILS ON", active=True)
            st.caption("Detected: " + ", ".join(get_guardrail().labels))
        else:
            ui_theme.badge("🔓 GUARDRAILS OFF", active=False)
            st.caption("Raw PII values are shown — demo purposes only.")

        st.markdown("---")
        st.markdown("### 📊 Session stats")
        stats = st.session_state.stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Queries", stats["queries"])
        col2.metric("Redacted", stats["responses_redacted"])
        col3.metric("PII values", stats["values_redacted"])

        if st.button("🗑️ Clear chat"):
            reset_stats()
            st.rerun()

        st.markdown("### 💡 Sample queries")
        ready = st.session_state.index_ready
        if not ready:
            st.caption("Available once the index is prepared.")
        for i, query in enumerate(SAMPLE_QUERIES):
            if st.button(query, key=f"sample_{i}", disabled=not ready):
                st.session_state.queued_query = query
                st.rerun()


def render_status_bar() -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        path = st.session_state.document_path
        if path:
            st.info(f"📄 **Document:** `{os.path.basename(path)}`")
        else:
            st.warning("No document selected.")
    with col2:
        if st.session_state.index_ready:
            st.success("✅ Index ready")
        else:
            st.warning("⚙️ Index not prepared")
    with col3:
        on = st.session_state.use_guardrails
        ui_theme.status_panel(
            "🔒 Guardrails ACTIVE" if on else "🔓 Guardrails DISABLED", active=on
        )


def handle_prompt(prompt: str) -> None:
    """Answer `prompt`, then rerun so the sidebar stats reflect this turn."""
    flagged = (
        get_guardrail().detect_pii_request(prompt) if st.session_state.use_guardrails else []
    )
    user_message = {"role": "user", "content": prompt, "flagged": flagged}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    # Rendered for immediate feedback; the rerun below redraws it from history.
    with st.chat_message("assistant"), st.spinner("Retrieving answer…"):
        assistant_message = answer_question(prompt)

    st.session_state.messages.append(assistant_message)
    st.rerun()


def main() -> None:
    ui_theme.inject_theme()
    init_state()

    problems = settings.validate()
    if problems:
        ui_theme.header()
        for problem in problems:
            st.error(problem)
        return

    render_sidebar()
    ui_theme.header()
    render_status_bar()
    st.markdown("---")

    if not st.session_state.index_ready:
        # Drop anything queued before the index existed, so it cannot fire later.
        st.session_state.pop("queued_query", None)
        st.info("👈 Pick a document and click **Prepare** in the sidebar to begin.")
        return

    for message in st.session_state.messages:
        render_message(message)

    typed = st.chat_input("Ask about customer data…")
    prompt = st.session_state.pop("queued_query", None) or typed
    if prompt:
        handle_prompt(prompt)


if __name__ == "__main__":
    main()
