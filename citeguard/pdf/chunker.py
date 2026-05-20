from __future__ import annotations
import hashlib
from citeguard.models import ChunkRecord


class TextChunker:
    """Sliding-window text chunker that operates on word counts."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, pages: list[dict], pdf_path: str) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for page_data in pages:
            words = page_data["text"].split()
            page_num = page_data["page"]
            step = max(1, self._chunk_size - self._overlap)
            start = 0
            while start < len(words):
                end = min(start + self._chunk_size, len(words))
                text = " ".join(words[start:end])
                chunk_id = hashlib.md5(
                    f"{pdf_path}:{page_num}:{start}".encode()
                ).hexdigest()[:12]
                chunks.append(ChunkRecord(
                    pdf_path=pdf_path,
                    chunk_id=chunk_id,
                    text=text,
                    page=page_num,
                ))
                if end == len(words):
                    break
                start += step
        return chunks
