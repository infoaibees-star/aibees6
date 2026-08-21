"""
RAG orchestration.

Wires a retriever (see `retrieval.py`) to a generator (see `generation.py`) and
manages the on-disk index. Swapping backends is a config change, not a code
change — `RAG_BACKEND` selects vertexai / ollama / keyword.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from langchain_core.documents import Document

from config import Settings, settings as default_settings
from generation import NO_ANSWER, Generator, build_generator
from pdf_processor import file_fingerprint, load_and_process_document
from retrieval import Retriever, build_retriever

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Base class for RAG failures surfaced to the UI."""


class RAGConfigError(RAGError):
    """Configuration is incomplete or inconsistent."""


class RAGNotReadyError(RAGError):
    """A question was asked before an index was prepared."""


@dataclass(frozen=True)
class Answer:
    """A generated answer plus the chunks it was grounded in."""

    text: str
    sources: list[Document] = field(default_factory=list)


class RAGSystem:
    """Retrieval-augmented question answering over a single document."""

    def __init__(self, config: Settings = default_settings) -> None:
        problems = config.validate()
        if problems:
            raise RAGConfigError(" ".join(problems))

        self.config = config
        self._retriever: Retriever = build_retriever(config)
        self._generator: Generator = build_generator(config)
        self._index_dir: str | None = None

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._index_dir is not None

    @property
    def backend_name(self) -> str:
        return self._generator.name

    @property
    def chunk_count(self) -> int:
        return self._retriever.size

    def index_dir_for(self, source_path: str) -> str:
        """Where the index for `source_path` lives, keyed by file content."""
        return os.path.join(
            self.config.index_root,
            f"{self.config.backend}-{file_fingerprint(source_path)}",
        )

    def prepare(self, source_path: str, force_rebuild: bool = False) -> bool:
        """Load the index for `source_path`, building it only when necessary.

        Returns True if the index had to be built, False if a cached one was reused.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"File not found: {source_path}")

        directory = self.index_dir_for(source_path)

        if not force_rebuild and type(self._retriever).exists(directory):
            try:
                self._retriever.load(directory)
                self._index_dir = directory
                logger.info("Reused index at '%s' (%d chunks).", directory, self.chunk_count)
                return False
            except Exception:
                logger.warning("Cached index at '%s' is unusable; rebuilding.", directory)

        chunks = load_and_process_document(source_path)
        if not chunks:
            raise RAGError(f"No text could be extracted from '{source_path}'.")

        os.makedirs(directory, exist_ok=True)
        self._retriever.build(chunks, directory)
        self._index_dir = directory
        logger.info("Built index at '%s' (%d chunks).", directory, len(chunks))
        return True

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def ask(self, question: str) -> Answer:
        """Retrieve relevant chunks and generate a grounded answer."""
        if not self.is_ready:
            raise RAGNotReadyError("No index prepared. Call prepare() first.")

        docs = self._retriever.retrieve(question, k=self.config.retriever_top_k)
        if not docs:
            return Answer(text=NO_ANSWER)

        return Answer(text=self._generator.generate(question, docs), sources=docs)
