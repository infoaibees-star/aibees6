import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
import streamlit as st

load_dotenv()

llm = ChatVertexAI(
    model_name="gemini-2.5-pro",
    project=os.getenv("GCP_PROJECT_ID"),
    location="us-central1",
    temperature=0,   # 0 = stick to the document, don't be creative
)

st.title("Chat with your Document")

# 1. Upload the PDF
pdf = st.file_uploader("Upload a PDF", type="pdf")

# 2. Read ALL text from the PDF (table + paragraphs)
document_text = ""
if pdf:
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    print("printing doc object")
    print(doc)
    print(type(doc))
    for page in doc:
        document_text += page.get_text()
    st.success("Document loaded!")

# 3. Ask a question
question = st.text_input("Ask a question about the document:")

if st.button("Get Answer") and question and document_text:
    messages = [
        {"role": "system", "content":
            "Answer the question using ONLY the document below. "
            "If the answer is not in it, say 'I don't know from this document.'\n\n"
            + document_text},
        {"role": "user", "content": question},
    ]
    answer = llm.invoke(messages).content
    st.write("Answer:", answer)

## Questions to Ask:
# "What is the price of the AeroBook 14 laptop?"
# "Which products are currently out of stock?"
# "What is the return policy for earbuds?"  
# ## Reading from tables 
# "List all products under ₹5,000."
# "What are the specs of the NovaVision 55-inch TV?"
# "Which product has ID NK-310?" 