"""
Document loading and chunking.

Supports PDF and plain-text sources. PDFs are read with PyMuPDF when it is
available (noticeably faster than pypdf) and fall back to pypdf otherwise.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")

_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB reads keep large PDFs off the heap.


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension has no registered loader."""


def _pdf_loader(path: str):
    """Prefer PyMuPDF; fall back to pypdf if it is not installed."""
    try:
        from langchain_community.document_loaders import PyMuPDFLoader

        return PyMuPDFLoader(path)
    except ImportError:
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(path)


def _text_loader(path: str):
    from langchain_community.document_loaders import TextLoader

    return TextLoader(path, encoding="utf-8")


_LOADERS: dict[str, Callable[[str], object]] = {
    ".pdf": _pdf_loader,
    ".txt": _text_loader,
}


def extension_of(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def is_supported(path: str) -> bool:
    return extension_of(path) in _LOADERS


def file_fingerprint(path: str) -> str:
    """Short, stable digest of a file's *contents*.

    Content-based (rather than path-based) so re-uploading the same document
    reuses its existing index instead of re-embedding it.
    """
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def load_and_process_document(file_path: Optional[str] = None) -> list[Document]:
    """Load a document from disk and split it into overlapping chunks.

    Raises:
        FileNotFoundError: the path does not exist.
        UnsupportedFileTypeError: the extension has no loader.
        ValueError: the file could not be parsed.
    """
    target = file_path or settings.default_document

    if not os.path.exists(target):
        raise FileNotFoundError(f"File not found: {target}")

    ext = extension_of(target)
    build_loader = _LOADERS.get(ext)
    if build_loader is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(_LOADERS)}"
        )

    try:
        pages = build_loader(target).load()
    except Exception as exc:  # loader-specific errors vary wildly; normalise them
        raise ValueError(f"Could not read '{target}': {exc}") from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(pages)
