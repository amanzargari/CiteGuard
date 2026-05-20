from __future__ import annotations
import pickle
import numpy as np
from pathlib import Path
import litellm
from citeguard.models import ChunkRecord
from citeguard.retrieval.base import RetrieverBase


class APIEmbeddingsRetriever(RetrieverBase):
    """Retriever using OpenRouter/API embedding endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "openai/text-embedding-3-small",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._chunks: list[ChunkRecord] = []
        self._embeddings: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = litellm.embedding(
            model=self._model,
            input=texts,
            api_key=self._api_key,
            api_base=self._base_url,
        )
        vecs = np.array(
            [d["embedding"] for d in response.data], dtype=np.float32
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)

    def index(self, chunks: list[ChunkRecord]) -> None:
        self._chunks = chunks
        if not chunks:
            return
        self._embeddings = self._embed([c.text for c in chunks])

    def query(self, text: str, top_k: int) -> list[ChunkRecord]:
        if self._embeddings is None or not self._chunks:
            return []
        q_emb = self._embed([text])[0]
        scores = self._embeddings @ q_emb
        k = min(top_k, len(self._chunks))
        indices = np.argsort(scores)[::-1][:k]
        results = []
        for i in indices:
            chunk = self._chunks[int(i)].model_copy()
            chunk.score = float(scores[i])
            results.append(chunk)
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self._chunks,
                "embeddings": self._embeddings,
                "model": self._model,
            }, f)

    @classmethod
    def load(cls, path: Path) -> APIEmbeddingsRetriever:
        with open(path, "rb") as f:
            data = pickle.load(f)
        r = cls(api_key="", base_url="", model=data["model"])
        r._chunks = data["chunks"]
        r._embeddings = data["embeddings"]
        return r
