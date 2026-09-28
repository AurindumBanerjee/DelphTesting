"""In-memory vector store.

The index lives for the life of one user session and is held in Gradio state.
There is deliberately no persist_directory: this app is a demo, and the old
on-disk Chroma was the source of the "embed succeeded, retrieval empty" bug
when it ran across multiple serverless containers.
"""

import uuid
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from src.models import get_embeddings


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Embed chunks and return a queryable in-memory collection."""
    if not chunks:
        raise ValueError("No chunks to index")

    texts = [c.page_content for c in chunks]
    metadatas = [_clean(c.metadata) for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    store = Chroma.from_texts(
        texts=texts,
        embedding=get_embeddings(),
        metadatas=metadatas,
        ids=ids,
        collection_name="pdf_chunks",
    )
    logger.info(f"Indexed {len(texts)} chunks in memory")
    return store


def _clean(metadata: dict) -> dict:
    """Chroma only accepts str/int/float/bool values; drop or coerce the rest."""
    out = {}
    for key, value in (metadata or {}).items():
        if value is None or value == "":
            continue
        out[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return out
