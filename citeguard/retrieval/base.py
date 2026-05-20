from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from citeguard.models import ChunkRecord


class RetrieverBase(ABC):
    @abstractmethod
    def index(self, chunks: list[ChunkRecord]) -> None: ...

    @abstractmethod
    def query(self, text: str, top_k: int) -> list[ChunkRecord]: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> RetrieverBase: ...
