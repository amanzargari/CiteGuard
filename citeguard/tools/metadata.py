from __future__ import annotations
from pathlib import Path
import fitz  # pymupdf


def make_metadata_tools(pdf_paths: list[str]):
    """Return (get_paper_metadata,) tool function."""

    def get_paper_metadata(pdf_path: str) -> dict:
        """Get title, author, and first-page excerpt from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            dict with title, author, first_page_excerpt, or error.
        """
        path = Path(pdf_path)
        if not path.exists():
            return {"error": f"File not found: {pdf_path}"}
        try:
            doc = fitz.open(str(path))
            meta = doc.metadata
            first_page_text = ""
            if len(doc) > 0:
                first_page_text = doc[0].get_text("text")[:1000]
            doc.close()
            return {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "first_page_excerpt": first_page_text,
                "pdf_path": pdf_path,
            }
        except Exception as e:
            return {"error": str(e)}

    return (get_paper_metadata,)
