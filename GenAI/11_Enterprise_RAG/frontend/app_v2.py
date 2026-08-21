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

# Initialize session state for persisting chat history, sources, and submission triggers
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "sources" not in st.session_state:
    st.session_state["sources"] = []
if "submitted_query" not in st.session_state:
    st.session_state["submitted_query"] = ""

# Callback function to handle Enter key submission and clear input box
def handle_submit():
    query = st.session_state["query_input_box"]
    if query.strip():
        st.session_state["submitted_query"] = query.strip()
    st.session_state["query_input_box"] = ""  # Clear the input box immediately

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
    st.session_state["submitted_query"] = ""
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
        
        # Clean snippet to prevent breaking HTML string attributes
        clean_snippet = snippet.replace('"', '&quot;').replace("'", "&#39;")
        
        # Render a beautiful, custom aligned, warm HIVE feature card
        st.sidebar.markdown(f"""
        <div style="
            background-color: white; 
            border: 1px solid #FAF2EE; 
            border-radius: 12px; 
            padding: 12px; 
            margin-bottom: 10px; 
            box-shadow: 0 4px 12px rgba(43,30,25,0.02);
        ">
            <div style="font-weight: 800; color: #E85C1C; font-size: 0.80rem; margin-bottom: 6px;">
                [{idx}] {filename} ({page_str})
            </div>
            <div style="font-size: 0.74rem; color: #5C4D46; font-style: italic; line-height: 1.3; margin-bottom: 6px;">
                "{clean_snippet}"
            </div>
            {f'<div style="font-family: monospace; font-size: 0.68rem; color: #E85C1C; background-color: #FAF2EE; padding: 3px 6px; border-radius: 4px; word-break: break-all;">{gcs_path}</div>' if gcs_path else ''}
        </div>
        """, unsafe_allow_html=True)

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
question = st.text_input(
    "Enter your medical question:", 
    placeholder="e.g., What is the recommended dosage for Metformin?", 
    key="query_input_box",
    on_change=handle_submit
)

# Button triggers click submission
btn_clicked = st.button("Retrieve Answer", type="primary")
if btn_clicked and st.session_state["query_input_box"].strip():
    st.session_state["submitted_query"] = st.session_state["query_input_box"].strip()
    st.session_state["query_input_box"] = ""  # Clear the input box
    st.rerun()

# ── RAG Pipeline Query Handler ────────────────────────────────────────
if st.session_state["submitted_query"]:
    query_to_process = st.session_state["submitted_query"]
    # Clear the submission trigger so it doesn't process on subsequent runs
    st.session_state["submitted_query"] = ""
    
    # Append User question to session history immediately to display it in the dialog
    st.session_state["messages"].append({"role": "user", "content": query_to_process})
    
    # Trigger an immediate rerender of the user message while we call the backend
    with chat_container:
        with st.chat_message("user", avatar=theme.USER_AVATAR):
            st.write(query_to_process)
        
        with st.spinner("Retrieving clinical references and generating response..."):
            try:
                # Prepare payload with preceding chat history
                payload = {
                    "question": query_to_process,
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
