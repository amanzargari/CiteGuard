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
        raise ValueError(f"Unsupported manuscript format: {suffix}")

    def _parse_docx(self, path: Path) -> str:
        from docx import Document
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
