from __future__ import annotations
import pickle
import numpy as np
from pathlib import Path
from citeguard.models import ChunkRecord
from citeguard.retrieval.base import RetrieverBase


class LocalEmbeddingsRetriever(RetrieverBase):
    """Retriever using sentence-transformers for local embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._chunks: list[ChunkRecord] = []
        self._embeddings: np.ndarray | None = None
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def index(self, chunks: list[ChunkRecord]) -> None:
        self._chunks = chunks
        if not chunks:
            return
        model = self._get_model()
        texts = [c.text for c in chunks]
        self._embeddings = model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ).astype(np.float32)

    def query(self, text: str, top_k: int) -> list[ChunkRecord]:
        if self._embeddings is None or not self._chunks:
            return []
        model = self._get_model()
        q_emb = model.encode(
            [text], show_progress_bar=False, normalize_embeddings=True
        )[0].astype(np.float32)
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
                "model_name": self._model_name,
            }, f)

    @classmethod
    def load(cls, path: Path) -> LocalEmbeddingsRetriever:
        with open(path, "rb") as f:
            data = pickle.load(f)
        r = cls(model_name=data["model_name"])
        r._chunks = data["chunks"]
        r._embeddings = data["embeddings"]
        return r
