import logging
import os
from google.cloud import aiplatform
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_vertexai import ChatVertexAI, VectorSearchVectorStore, VertexAIEmbeddings

from config import (
    PROJECT_ID, REGION, EMBED_BUCKET, EMBED_BUCKET_URI,
    INDEX_ID, ENDPOINT_ID,
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    RETRIEVER_TOP_K,
)

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "models/gemini-embedding-001"

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a precise medical information assistant.
Answer only using the context provided below.
If the answer is not in the context, say:
"I don't have enough information in the knowledge base to answer this."

Context:
{context}

Question: {question}

Answer:""")

CONDENSE_PROMPT = ChatPromptTemplate.from_template("""
Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the conversation history. Do NOT answer the question, just rephrase it.

Conversation History:
{chat_history}

Follow-Up Question: {question}

Standalone Question:""")


def format_docs(docs: list) -> str:
    """Concatenate retrieved chunks into a single context string."""
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_file', 'unknown')}"
        + (f", page {d.metadata.get('page')}" if d.metadata.get("page") is not None else "")
        + f"]\n{d.page_content}"
        for d in docs
    )


class RAGQueryEngine:
    def __init__(self):
        if not INDEX_ID or not ENDPOINT_ID:
            raise ValueError("INDEX_ID and ENDPOINT_ID environment variables must be set.")

        log.info("Initializing Vertex AI platform (project=%s, region=%s)", PROJECT_ID, REGION)
        aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=EMBED_BUCKET_URI)

        log.info("Initializing Vertex AI embedding model: gemini-embedding-001")
        self.embedder = VertexAIEmbeddings(
            model_name="gemini-embedding-001",
            project=PROJECT_ID,
            location=REGION,
        )

        log.info("Connecting to Vector Search (Index: %s, Endpoint: %s)", INDEX_ID, ENDPOINT_ID)
        self.vector_store = VectorSearchVectorStore.from_components(
            project_id=PROJECT_ID,
            region=REGION,
            gcs_bucket_name=EMBED_BUCKET,
            index_id=INDEX_ID,
            endpoint_id=ENDPOINT_ID,
            embedding=self.embedder,
            stream_update=True,
        )

        log.info("Loading LLM: %s", LLM_MODEL)
        self.llm = ChatVertexAI(
            model_name=LLM_MODEL,
            project=PROJECT_ID,
            location=REGION,
            max_output_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )

    def query(self, question: str, history: list = [], top_k: int = RETRIEVER_TOP_K) -> dict:
        # 1. Condense question if history is provided
        standalone_question = question
        if history:
            log.info("Condensing follow-up question based on chat history...")
            formatted_history = ""
            for msg in history:
                role_name = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
                content_text = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                role = "User" if role_name == "user" else "Assistant"
                formatted_history += f"{role}: {content_text}\n"
            
            condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
            try:
                standalone_question = condense_chain.invoke({
                    "chat_history": formatted_history,
                    "question": question
                }).strip()
                log.info("Condensed standalone query: '%s'", standalone_question)
            except Exception as exc:
                log.warning("Question condensation failed, falling back to original query: %s", exc)

        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

        # Build execution chain
        rag_chain = (
            RunnableParallel({
                "context":  retriever | format_docs,
                "question": RunnablePassthrough(),
            })
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

        # Retrieve documents and answer using standalone question
        docs = retriever.invoke(standalone_question)
        answer = rag_chain.invoke(standalone_question)

        # Structure response
        sources = []
        for d in docs:
            meta = d.metadata
            sources.append({
                "source_file": meta.get("source_file", "unknown"),
                "page": meta.get("page"),
                "source_gcs": meta.get("source_gcs", ""),
                "snippet": d.page_content
            })

        return {
            "answer": answer,
            "sources": sources
        }
