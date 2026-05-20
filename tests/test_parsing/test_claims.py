import pytest
from citeguard.parsing.claims import ClaimSegmenter, BibliographyParser
from citeguard.models import CitationRecord, CitationFormat


def make_citation(cid: str, marker: str, pos: int) -> CitationRecord:
    return CitationRecord(
        id=cid, raw_marker=marker,
        format=CitationFormat.NUMERIC, position=pos,
    )


TEXT = (
    "Neural networks have revolutionized NLP. "
    "Transformers [1] achieved 95% accuracy on GLUE. "
    "This surpassed all previous methods. "
    "Later, BERT [2] set new records across many benchmarks."
)


def test_claim_contains_citation_context():
    seg = ClaimSegmenter(window=2)
    c = make_citation("ref_001", "[1]", TEXT.index("[1]"))
    result = seg.segment(TEXT, c)
    # The claim should contain the sentence with the citation
    assert "Transformers" in result.claim_text or "[1]" in result.claim_text
    assert result.citation_id == "ref_001"


def test_claim_has_context_fields():
    seg = ClaimSegmenter(window=2)
    c = make_citation("ref_001", "[1]", TEXT.index("[1]"))
    result = seg.segment(TEXT, c)
    assert isinstance(result.context_before, str)
    assert isinstance(result.context_after, str)


def test_claim_citation_id_matches():
    seg = ClaimSegmenter(window=2)
    c = make_citation("ref_001", "[1]", TEXT.index("[1]"))
    result = seg.segment(TEXT, c)
    assert result.citation_id == "ref_001"


def test_bibliography_numeric_entries():
    bib = "Some text.\n\nReferences\n\n[1] Smith, J. et al. Nature, 2023.\n[2] Jones, A. ICML, 2022.\n"
    parser = BibliographyParser()
    entries = parser.parse(bib)
    assert "[1]" in entries
    assert "Smith" in entries["[1]"]
    assert "[2]" in entries
    assert "Jones" in entries["[2]"]


def test_bibliography_returns_empty_if_no_section():
    parser = BibliographyParser()
    entries = parser.parse("No references here at all.")
    assert isinstance(entries, dict)
    assert len(entries) == 0


def test_bibliography_named_key_entries():
    bib = "Text.\n\nReferences\n\n[smith2023] Smith, J. Nature 2023.\n"
    parser = BibliographyParser()
    entries = parser.parse(bib)
    assert "[smith2023]" in entries
