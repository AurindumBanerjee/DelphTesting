"""Model providers.

Two seams, so switching vendors is a config change and never a code change:
  get_embeddings() -> a LangChain Embeddings object (local, CPU, free)
  chat()           -> one call against any OpenAI-compatible chat endpoint
"""

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings

from src.config import settings


class FastEmbedEmbeddings(Embeddings):
    """Runs an ONNX sentence-embedding model in-process.

    Chosen over sentence-transformers/torch deliberately: ~5x smaller install
    and a much faster cold start, which matters on a free 2-vCPU Space.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Cached: the model is loaded once per process, not once per upload."""
    return FastEmbedEmbeddings(settings.embed_model)


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    return OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "not-needed",
    )


def chat(system_prompt: str, user_prompt: str) -> str:
    """Single-turn completion against the configured endpoint."""
    completion = _client().chat.completions.create(
        model=settings.llm_model,
        temperature=settings.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content
