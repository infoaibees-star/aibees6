import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
import streamlit as st
load_dotenv()

llm = ChatVertexAI(
    model_name="gemini-2.5-pro",
    project = os.getenv("GCP_PROJECT_ID"),
    location = "us-central1",
    temperature=1,   # 0-2
   
)

st.title("Ask Gemini (Vertex AI)")
question = st.text_input("Enter your question here:")

if st.button("Get Answer") and question:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
    response = llm.invoke(messages).content
    st.write("Answer:", response)

