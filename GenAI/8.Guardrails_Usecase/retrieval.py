"""
Retrieval backends.

Two interchangeable implementations behind one interface:

* :class:`KeywordRetriever` — dependency-free inverted index with IDF scoring.
* :class:`VectorRetriever`  — FAISS similarity search over embedded chunks.

The inverted index is what makes keyword mode cheap: chunks are tokenised once at
build time, so a query only touches the postings of the words it actually
contains instead of re-tokenising every chunk.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Iterable

from langchain_core.documents import Document

from config import Settings

CHUNKS_FILE = "chunks.jsonl"

_WORD_RE = re.compile(r"[a-z0-9]+")


class IndexNotReadyError(RuntimeError):
    """Raised when a retriever is queried before it has an index."""


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _dump_chunks(directory: str, chunks: Iterable[Document]) -> None:
    with open(os.path.join(directory, CHUNKS_FILE), "w", encoding="utf-8") as handle:
        for doc in chunks:
            record = {"page_content": doc.page_content, "metadata": doc.metadata}
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_chunks(directory: str) -> list[Document]:
    path = os.path.join(directory, CHUNKS_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing '{path}' — rebuild the index.")
    docs: list[Document] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                obj: dict[str, Any] = json.loads(line)
                docs.append(
                    Document(
                        page_content=obj.get("page_content", ""),
                        metadata=obj.get("metadata") or {},
                    )
                )
    return docs


class Retriever(ABC):
    """Common interface for every retrieval strategy."""

    @staticmethod
    @abstractmethod
    def exists(directory: str) -> bool:
        """True if `directory` holds a reusable index for this retriever."""

    @abstractmethod
    def build(self, chunks: list[Document], directory: str) -> None:
        """Index `chunks` and persist the result to `directory`."""

    @abstractmethod
    def load(self, directory: str) -> None:
        """Restore a previously built index."""

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[Document]:
        """Return the `k` chunks most relevant to `query`."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of indexed chunks."""


class KeywordRetriever(Retriever):
    """Inverted-index retriever with IDF-weighted overlap scoring. No LLM, no network."""

    def __init__(self) -> None:
        self._chunks: list[Document] = []
        self._postings: dict[str, list[int]] = {}
        self._idf: dict[str, float] = {}

    @staticmethod
    def exists(directory: str) -> bool:
        return os.path.exists(os.path.join(directory, CHUNKS_FILE))

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Document], directory: str) -> None:
        self.index_chunks(chunks)
        _dump_chunks(directory, chunks)

    def load(self, directory: str) -> None:
        self.index_chunks(_read_chunks(directory))

    def index_chunks(self, chunks: list[Document]) -> None:
        """Build the in-memory inverted index without touching disk."""
        postings: dict[str, list[int]] = defaultdict(list)
        for i, doc in enumerate(chunks):
            for token in set(tokenize(doc.page_content)):
                postings[token].append(i)

        total = max(len(chunks), 1)
        self._chunks = chunks
        self._postings = dict(postings)
        # Smoothed IDF: frequent boilerplate words contribute almost nothing.
        self._idf = {
            token: math.log(1 + total / len(ids)) for token, ids in self._postings.items()
        }

    def retrieve(self, query: str, k: int) -> list[Document]:
        if not self._chunks:
            raise IndexNotReadyError("Keyword index is empty. Build or load it first.")

        scores: dict[int, float] = defaultdict(float)
        for token in set(tokenize(query)):
            weight = self._idf.get(token)
            if weight is None:
                continue
            for idx in self._postings[token]:
                scores[idx] += weight

        if not scores:
            return []
        best = heapq.nlargest(max(k, 1), scores.items(), key=lambda item: item[1])
        return [self._chunks[idx] for idx, _ in best]


class VectorRetriever(Retriever):
    """FAISS similarity search. `embeddings` is any LangChain embeddings object."""

    def __init__(self, embeddings: Any) -> None:
        self._embeddings = embeddings
        self._store: Any = None
        self._size = 0

    @staticmethod
    def exists(directory: str) -> bool:
        return os.path.exists(os.path.join(directory, "index.faiss"))

    @property
    def size(self) -> int:
        return self._size

    @staticmethod
    def _faiss():
        from langchain_community.vectorstores import FAISS

        return FAISS

    def build(self, chunks: list[Document], directory: str) -> None:
        self._store = self._faiss().from_documents(chunks, self._embeddings)
        self._store.save_local(directory)
        self._size = len(chunks)
        # Kept alongside the vectors so keyword mode can reuse the same folder.
        _dump_chunks(directory, chunks)

    def load(self, directory: str) -> None:
        self._store = self._faiss().load_local(
            directory, self._embeddings, allow_dangerous_deserialization=True
        )
        self._size = getattr(self._store.index, "ntotal", 0)

    def retrieve(self, query: str, k: int) -> list[Document]:
        if self._store is None:
            raise IndexNotReadyError("Vector store not loaded. Build or load it first.")
        return self._store.similarity_search(query, k=max(k, 1))


class HybridRetriever(Retriever):
    """Fuses dense (FAISS) and lexical (inverted index) results.

    Dense search handles paraphrases; lexical search nails the exact identifiers
    — "Customer Profile #4", "ACC55491338" — that embeddings routinely miss.
    Results are merged with reciprocal rank fusion, which needs no score
    calibration between the two very different scales.
    """

    RRF_K = 60  # standard damping constant; larger = flatter rank weighting

    def __init__(self, embeddings: Any) -> None:
        self._vector = VectorRetriever(embeddings)
        self._keyword = KeywordRetriever()

    @staticmethod
    def exists(directory: str) -> bool:
        return VectorRetriever.exists(directory) and KeywordRetriever.exists(directory)

    @property
    def size(self) -> int:
        return self._vector.size

    def build(self, chunks: list[Document], directory: str) -> None:
        # VectorRetriever.build already persists chunks.jsonl, so the keyword
        # side only needs its in-memory index.
        self._vector.build(chunks, directory)
        self._keyword.index_chunks(chunks)

    def load(self, directory: str) -> None:
        self._vector.load(directory)
        self._keyword.load(directory)

    def retrieve(self, query: str, k: int) -> list[Document]:
        ranked: dict[str, float] = defaultdict(float)
        docs_by_key: dict[str, Document] = {}

        for retriever in (self._vector, self._keyword):
            for rank, doc in enumerate(retriever.retrieve(query, k)):
                key = doc.page_content
                docs_by_key.setdefault(key, doc)
                ranked[key] += 1.0 / (self.RRF_K + rank)

        best = heapq.nlargest(2 * k, ranked.items(), key=lambda item: item[1])
        return [docs_by_key[key] for key, _ in best]


def _embeddings_for(settings: Settings) -> Any:
    """Lazily construct the embeddings client for the active backend."""
    if settings.backend == "vertexai":
        from langchain_google_vertexai import VertexAIEmbeddings

        return VertexAIEmbeddings(
            model=settings.embedding_model,
            project=settings.gcp_project_id,
            location=settings.gcp_region,
        )

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def build_retriever(settings: Settings) -> Retriever:
    """Pick the retriever that matches the configured backend."""
    if settings.uses_vector_store:
        return HybridRetriever(_embeddings_for(settings))
    return KeywordRetriever()
