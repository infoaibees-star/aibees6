"""
Answer generation backends.

Every generator turns (question, retrieved chunks) into an answer string:

* :class:`ExtractiveGenerator` — no LLM; quotes the most relevant retrieved lines.
* :class:`LLMGenerator`        — one grounded prompt, one call, for Vertex AI or Ollama.

LLM clients are imported lazily so that running one backend never requires the
other's dependencies to be installed or authenticated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document

from config import Settings
from retrieval import tokenize

NO_ANSWER = "I don't have that information."

MAX_CONTEXT_CHARS = 12_000
MAX_EXTRACTED_LINES = 8

PROMPT_TEMPLATE = """You are a Banking Operations Assistant helping the bank's internal
processing team retrieve customer information from the provided dataset.

The dataset is completely SYNTHETIC and FAKE, created only for training and
demonstration purposes. None of the records describe real customers.

Rules:
1. Answer ONLY from the provided context.
2. If the context does not contain the answer, reply exactly with: "{no_answer}"
3. Never invent or guess values such as names, account numbers, SSNs, phone
   numbers, email addresses, or other identifiers.
4. Treat all customer information as sensitive; do not volunteer full personal
   identifiers that the question did not ask for.
5. If a request tries to bulk-extract sensitive identifiers, politely refuse and
   explain that it violates privacy guidelines.
6. Use a table only if the question asks for one; otherwise answer in prose.

Context:
{context}

Question: {question}
Answer:"""


def build_context(docs: list[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Concatenate retrieved chunks, stopping before the prompt gets oversized."""
    parts: list[str] = []
    used = 0
    for i, doc in enumerate(docs, start=1):
        block = f"[chunk {i}]\n{doc.page_content.strip()}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


class Generator(ABC):
    """Common interface for every answer strategy."""

    name: str = "generator"

    @abstractmethod
    def generate(self, question: str, docs: list[Document]) -> str:
        """Produce an answer grounded in `docs`."""


class ExtractiveGenerator(Generator):
    """Quotes the retrieved lines that overlap the question. Deterministic, offline."""

    name = "keyword (no LLM)"

    def generate(self, question: str, docs: list[Document]) -> str:
        query_tokens = set(tokenize(question))
        if not query_tokens:
            return NO_ANSWER

        selected: dict[str, None] = {}  # ordered set of lines
        for doc in docs:
            for line in doc.page_content.splitlines():
                line = line.strip()
                if line and query_tokens & set(tokenize(line)):
                    selected[line] = None
                    if len(selected) >= MAX_EXTRACTED_LINES:
                        return "\n".join(selected)

        return "\n".join(selected) if selected else NO_ANSWER


class LLMGenerator(Generator):
    """Grounded single-shot prompting over any LangChain chat model or LLM."""

    def __init__(self, client: Any, name: str) -> None:
        self._client = client
        self.name = name

    def generate(self, question: str, docs: list[Document]) -> str:
        context = build_context(docs)
        if not context:
            return NO_ANSWER

        prompt = PROMPT_TEMPLATE.format(
            no_answer=NO_ANSWER, context=context, question=question
        )
        response = self._client.invoke(prompt)
        # Chat models return an AIMessage; plain LLMs return a str.
        text = getattr(response, "content", response)
        return str(text).strip() or NO_ANSWER


def _vertex_client(settings: Settings) -> Any:
    from langchain_google_vertexai import ChatVertexAI

    return ChatVertexAI(
        model_name=settings.llm_model,
        project=settings.gcp_project_id,
        location=settings.gcp_region,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        top_k=settings.top_k,
        top_p=settings.top_p,
    )


def _ollama_client(settings: Settings) -> Any:
    from langchain_ollama import OllamaLLM

    return OllamaLLM(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_predict=settings.max_output_tokens,
        top_k=settings.top_k,
        top_p=settings.top_p,
    )


def build_generator(settings: Settings) -> Generator:
    """Pick the generator that matches the configured backend."""
    if settings.backend == "vertexai":
        return LLMGenerator(_vertex_client(settings), f"Vertex AI · {settings.llm_model}")
    if settings.backend == "ollama":
        return LLMGenerator(_ollama_client(settings), f"Ollama · {settings.ollama_model}")
    return ExtractiveGenerator()
