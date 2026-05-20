import pytest
from pathlib import Path
from citeguard.parsing.manuscript import ManuscriptParser

def test_parse_plain_text(tmp_path):
    f = tmp_path / "paper.txt"
    f.write_text("This is a test manuscript with [1] a citation.")
    parser = ManuscriptParser()
    text = parser.parse(f)
    assert "test manuscript" in text
    assert "[1]" in text

def test_parse_markdown(tmp_path):
    f = tmp_path / "paper.md"
    f.write_text("# Title\n\nThis method [2] outperforms baselines.\n\n## References\n")
    parser = ManuscriptParser()
    text = parser.parse(f)
    assert "outperforms baselines" in text

def test_parse_unsupported_format_raises(tmp_path):
    f = tmp_path / "paper.xyz"
    f.write_text("data")
    parser = ManuscriptParser()
    with pytest.raises(ValueError, match="Unsupported"):
        parser.parse(f)

def test_parse_docx(tmp_path):
    from docx import Document
    doc = Document()
    doc.add_paragraph("Smith et al. (2023) showed that transformers work well.")
    path = tmp_path / "paper.docx"
    doc.save(str(path))
    parser = ManuscriptParser()
    text = parser.parse(path)
    assert "transformers work well" in text
