import pytest
from pathlib import Path
import fitz  # pymupdf


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    """Create a minimal PDF with known content for testing."""
    tmp = tmp_path_factory.mktemp("pdfs")
    path = tmp / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 72),
        "Abstract\n\n"
        "This paper presents a novel approach to citation verification. "
        "We achieve 94.3% accuracy on the benchmark dataset. "
        "Our method outperforms all prior baselines by a significant margin.\n\n"
        "Introduction\n\n"
        "Citation accuracy is a critical problem in academic publishing. "
        "Many manuscripts misrepresent their cited sources.",
        fontsize=11,
    )
    doc.save(str(path))
    doc.close()
    return path
