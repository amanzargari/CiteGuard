from __future__ import annotations
from pathlib import Path


class ManuscriptParser:
    """Convert any supported manuscript format to plain text."""

    def parse(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            return path.read_text(encoding="utf-8")
        if suffix == ".docx":
            return self._parse_docx(path)
        if suffix == ".pdf":
            return self._parse_pdf(path)
        raise ValueError(f"Unsupported manuscript format: {suffix}")

    def _parse_docx(self, path: Path) -> str:
        from docx import Document
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    def _parse_pdf(self, path: Path) -> str:
        import fitz
        doc = fitz.open(str(path))
        try:
            pages = [doc[i].get_text("text") for i in range(len(doc))]
        finally:
            doc.close()
        return "\n\n".join(pages)
