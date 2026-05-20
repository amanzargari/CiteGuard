import pytest
from citeguard.parsing.citations import CitationExtractor
from citeguard.models import CitationFormat


def ext():
    return CitationExtractor()


def test_numeric_single():
    hits = ext().extract("This method [42] is state-of-the-art.")
    assert len(hits) == 1
    assert hits[0].raw_marker == "[42]"
    assert hits[0].format == CitationFormat.NUMERIC


def test_numeric_multiple_in_bracket():
    hits = ext().extract("See prior work [1,2,3].")
    assert any(h.format == CitationFormat.NUMERIC for h in hits)
    assert len(hits) >= 1


def test_named_key():
    hits = ext().extract("As in [smith2023nature], we observe the same trend.")
    assert len(hits) == 1
    assert hits[0].format == CitationFormat.NAMED_KEY
    assert hits[0].raw_marker == "[smith2023nature]"


def test_author_year_paren():
    hits = ext().extract("Accuracy improved (Smith et al., 2023).")
    assert len(hits) == 1
    assert hits[0].format == CitationFormat.AUTHOR_YEAR
    assert "Smith" in hits[0].raw_marker


def test_author_year_inline():
    hits = ext().extract("Smith (2023) demonstrated this effect.")
    assert any(h.format == CitationFormat.AUTHOR_YEAR for h in hits)


def test_superscript_footnote():
    hits = ext().extract("This was shown¹² in multiple studies.")
    assert any(h.format == CitationFormat.FOOTNOTE for h in hits)


def test_position_recorded():
    text = "Intro text. Then [7] appears."
    hits = ext().extract(text)
    assert hits[0].position > 0


def test_citation_ids_unique():
    hits = ext().extract("Refs [1], [2], and [3] matter.")
    ids = [h.id for h in hits]
    assert len(ids) == len(set(ids))


def test_no_false_positives_in_plain_text():
    hits = ext().extract("The year 2023 was productive. We used 3 methods.")
    # Should not detect bare numbers as citations
    numeric_hits = [h for h in hits if h.format == CitationFormat.NUMERIC]
    assert len(numeric_hits) == 0


def test_named_key_not_confused_with_numeric():
    hits = ext().extract("See [smith2023] for details and [42] for data.")
    formats = {h.format for h in hits}
    assert CitationFormat.NAMED_KEY in formats
    assert CitationFormat.NUMERIC in formats
