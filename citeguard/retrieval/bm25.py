from __future__ import annotations
import pickle
from pathlib import Path
from rank_bm25 import BM25Plus
from citeguard.models import ChunkRecord
from citeguard.retrieval.base import RetrieverBase


class BM25Retriever(RetrieverBase):
    def __init__(self) -> None:
        self._chunks: list[ChunkRecord] = []
        self._bm25: BM25Plus | None = None

    def index(self, chunks: list[ChunkRecord]) -> None:
        self._chunks = chunks
        if not chunks:
            return
        tokenized = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Plus(tokenized)

    def query(self, text: str, top_k: int) -> list[ChunkRecord]:
        if not self._bm25 or not self._chunks:
            return []
        tokens = text.lower().split()
        scores = self._bm25.get_scores(tokens)
        k = min(top_k, len(self._chunks))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for i in indices:
            chunk = self._chunks[i].model_copy()
            chunk.score = float(scores[i])
            results.append(chunk)
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, f)

    @classmethod
    def load(cls, path: Path) -> BM25Retriever:
        with open(path, "rb") as f:
            data = pickle.load(f)
        r = cls()
        r._chunks = data["chunks"]
        r._bm25 = data["bm25"]
        return r
