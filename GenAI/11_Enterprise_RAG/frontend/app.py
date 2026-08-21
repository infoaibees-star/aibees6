import os
import streamlit as st
import requests
import theme

# Load config from env or set defaults
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080").rstrip("/")

st.set_page_config(
    page_title="Medi Bee Assist - by AIBees",
    page_icon="🐝",
    layout="wide"
)

# Initialize session state for persisting chat history and sources across runs
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "sources" not in st.session_state:
    st.session_state["sources"] = []

# Inject custom brand colors, font, and styles
theme.inject_css()

# Render brand header card fixed at the top
theme.render_header()

# Sidebar brand logo & controls
st.sidebar.markdown('<div style="margin-top: -20px;"></div>', unsafe_allow_html=True)
theme.render_sidebar_logo()

st.sidebar.header("Configuration")
backend_service = st.sidebar.text_input("Backend URL", value=BACKEND_URL)
top_k = st.sidebar.slider("Chunks to Retrieve (Top K)", min_value=1, max_value=10, value=5)

if st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = []
    st.session_state["sources"] = []
    st.rerun()

st.sidebar.markdown("""
---
### Product Profile
Medi Bee Assist is a state-of-the-art Healthcare AI Bot product developed by AI Bees.
""")

# Render retrieved reference sources on the left hand side sidebar if they exist
if st.session_state["sources"]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Retrieved References")
    for idx, src in enumerate(st.session_state["sources"], 1):
        filename = src.get("source_file", "unknown")
        page = src.get("page")
        page_str = f"Page {page}" if page is not None else "Unknown page"
        gcs_path = src.get("source_gcs", "")
        snippet = src.get("snippet", "")
        
        with st.sidebar.expander(f"[{idx}] {filename} ({page_str})"):
            if gcs_path:
                st.markdown(f"**GCS URI:**\n`{gcs_path}`")
            st.caption("Source Snippet:")
            st.markdown(f"*{snippet}*")

# Create a container for the chat messages history so they render in the middle of the screen
chat_container = st.container()

with chat_container:
    # Render historical chat dialogue
    for msg in st.session_state["messages"]:
        avatar_img = theme.USER_AVATAR if msg["role"] == "user" else theme.BOT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar_img):
            st.markdown(msg["content"])

# Input form (keep at bottom of the main screen)
st.write("")
question = st.text_input("Enter your medical question:", placeholder="e.g., What is the recommended dosage for Metformin?", key="query_input")

if st.button("Retrieve Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # Append User question to session history immediately to display it in the dialog
        st.session_state["messages"].append({"role": "user", "content": question})
        
        # Trigger an immediate rerender of the user message while we call the backend
        with chat_container:
            with st.chat_message("user", avatar=theme.USER_AVATAR):
                st.write(question)
            
            with st.spinner("Retrieving clinical references and generating response..."):
                try:
                    # Prepare payload with preceding chat history
                    payload = {
                        "question": question,
                        "history": st.session_state["messages"][:-1],  # exclude current question from history
                        "top_k": top_k
                    }
                    
                    # Post request to Backend
                    response = requests.post(f"{backend_service}/query", json=payload, timeout=120)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer")
                        sources = data.get("sources", [])
                        
                        # Store retrieved sources in session state to persist and display in sidebar
                        st.session_state["sources"] = sources
                        
                        # Append Assistant response to session history
                        st.session_state["messages"].append({"role": "assistant", "content": answer})
                        
                        # Trigger a full rerun to flush both the chat canvas and the sidebar together
                        st.rerun()
                    else:
                        st.error(f"Error {response.status_code} from backend: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to backend at {backend_service}. Error: {e}")
