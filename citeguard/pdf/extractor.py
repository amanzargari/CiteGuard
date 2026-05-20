from __future__ import annotations
from pathlib import Path
import fitz  # pymupdf


class PDFExtractor:
    """Extract text from PDF files using pymupdf."""

    def extract(self, path: Path) -> list[dict]:
        """Return list of {text: str, page: int} dicts, one per page with content."""
        if not path.exists():
            raise RuntimeError(f"PDF not found: {path}")
        try:
            doc = fitz.open(str(path))
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"text": text, "page": i + 1})
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to extract PDF {path}: {e}") from e
        return pages
