from __future__ import annotations
from pathlib import Path
from difflib import SequenceMatcher

from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
)

from citeguard.config import Settings
from citeguard.models import CitationRecord
from citeguard.pdf.extractor import PDFExtractor
from citeguard.pdf.chunker import TextChunker
from citeguard.retrieval.base import RetrieverBase
from citeguard.retrieval.factory import build_retriever
from citeguard.cache.manager import CacheManager


def _similarity(a: str, b: str) -> float:
    """Normalized SequenceMatcher similarity between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class PDFIndexer:
    """Extracts, chunks, and indexes PDFs. Matches citations to PDFs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._extractor = PDFExtractor()
        self._chunker = TextChunker(
            chunk_size=settings.retrieval.chunk_size,
            overlap=settings.retrieval.chunk_overlap,
        )
        self._retrievers: dict[str, RetrieverBase] = {}

    def index_all(
        self,
        pdf_paths: list[Path],
        cache: CacheManager,
        show_progress: bool = True,
    ) -> None:
        """Index all PDFs, using cache for already-indexed ones."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Indexing PDFs"),
            BarColumn(),
            TaskProgressColumn(),
            disable=not show_progress,
        ) as progress:
            task = progress.add_task("", total=len(pdf_paths))
            for pdf_path in pdf_paths:
                self._index_one(pdf_path, cache)
                progress.advance(task)

    def _index_one(self, pdf_path: Path, cache: CacheManager) -> None:
        """Index a single PDF, loading from cache if available."""
        cache_path = cache.index_cache_path(str(pdf_path))

        if cache_path.exists():
            try:
                retriever = build_retriever(self._settings)
                self._retrievers[str(pdf_path)] = type(retriever).load(cache_path)
                return
            except Exception:
                pass  # fall through to re-index

        try:
            pages = self._extractor.extract(pdf_path)
        except RuntimeError:
            return  # skip unreadable PDFs silently

        chunks = self._chunker.chunk(pages, str(pdf_path))
        retriever = build_retriever(self._settings)
        retriever.index(chunks)
        try:
            retriever.save(cache_path)
        except Exception:
            pass  # non-fatal if cache write fails
        self._retrievers[str(pdf_path)] = retriever

    def get_retriever(self, pdf_path: str) -> RetrieverBase | None:
        """Get the retriever for a specific PDF, or None if not indexed."""
        return self._retrievers.get(pdf_path)

    def match_citations_to_pdfs(
        self,
        citations: list[CitationRecord],
        reference_entries: dict[str, str],
        pdf_paths: list[Path],
    ) -> list[CitationRecord]:
        """Assign matched_pdf to each citation via multi-strategy fuzzy matching."""
        updated = []
        for citation in citations:
            ref_text = reference_entries.get(citation.raw_marker, "")
            best_pdf: str | None = None
            best_score = 0.0

            for pdf_path in pdf_paths:
                score = self._match_score(ref_text, pdf_path)
                if score > best_score:
                    best_score = score
                    best_pdf = str(pdf_path)

            matched = best_pdf if best_score >= 0.35 else None
            updated.append(citation.model_copy(update={
                "matched_pdf": matched,
                "reference_text": ref_text,
            }))
        return updated

    def _match_score(self, ref_text: str, pdf_path: Path) -> float:
        """Compute best match score between reference text and PDF using multiple strategies."""
        scores: list[float] = []

        # Strategy 1: Filename similarity
        scores.append(_similarity(ref_text, pdf_path.stem))

        # Strategy 2: PDF metadata (title, author, first-page text)
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            meta_title = doc.metadata.get("title", "")
            meta_author = doc.metadata.get("author", "")
            first_page = doc[0].get_text("text")[:500] if len(doc) > 0 else ""
            doc.close()

            if meta_title:
                scores.append(_similarity(ref_text, meta_title))
            if meta_author:
                scores.append(_similarity(ref_text, meta_author))
            if first_page:
                scores.append(_similarity(ref_text[:200], first_page[:200]))
        except Exception:
            pass

        return max(scores) if scores else 0.0
