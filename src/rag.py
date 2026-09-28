"""The RAG loop: retrieve -> build a cited prompt -> generate."""

from typing import List, Tuple

from langchain_core.documents import Document
from loguru import logger

from src.config import DEFAULT_SYSTEM_PROMPT, settings
from src.models import chat


def retrieve(store, question: str, k: int) -> List[Document]:
    return store.similarity_search(question, k=k)


def format_context(docs: List[Document]) -> str:
    """Number each passage so the model can cite it as [n]."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{i}] ({src}, p.{page})\n{doc.page_content}")
    return "\n\n".join(blocks)


def format_sources(docs: List[Document]) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        preview = " ".join(doc.page_content.split())[:180]
        lines.append(f"**[{i}] {src} — page {page}**\n\n> {preview}…")
    return "\n\n".join(lines) if lines else "_No passages retrieved._"


def answer(store, question: str, k: int = None, system_prompt: str = None) -> Tuple[str, str]:
    """Return (answer_markdown, sources_markdown)."""
    k = k or settings.default_k
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    if store is None:
        return "Upload a PDF first — there is nothing indexed yet.", ""

    docs = retrieve(store, question, k)
    if not docs:
        return "No relevant passages were found in the indexed documents.", ""

    if not settings.llm_configured:
        return (
            "**No LLM endpoint configured.** Retrieval worked — the passages are shown "
            "below — but generation needs `LLM_API_KEY` (and optionally `LLM_BASE_URL` "
            "and `LLM_MODEL`) to be set.",
            format_sources(docs),
        )

    user_prompt = f"Question: {question}\n\nContext passages:\n{format_context(docs)}"
    try:
        reply = chat(system_prompt, user_prompt)
    except Exception as exc:
        logger.error(f"Generation failed: {exc}")
        reply = f"Generation failed: {exc}"

    return reply, format_sources(docs)
