from __future__ import annotations
from citeguard.config import Settings
from citeguard.retrieval.base import RetrieverBase


def build_retriever(settings: Settings) -> RetrieverBase:
    """Instantiate the configured retrieval backend."""
    backend = settings.retrieval_backend
    if backend == "bm25":
        from citeguard.retrieval.bm25 import BM25Retriever
        return BM25Retriever()
    if backend == "local_embeddings":
        from citeguard.retrieval.local_embeddings import LocalEmbeddingsRetriever
        return LocalEmbeddingsRetriever(model_name=settings.local_embeddings.model)
    if backend == "api_embeddings":
        from citeguard.retrieval.api_embeddings import APIEmbeddingsRetriever
        return APIEmbeddingsRetriever(
            api_key=settings.embedding_api_key,
            base_url=settings.openrouter_base_url,
        )
    raise ValueError(f"Unknown retrieval_backend: {backend!r}")
