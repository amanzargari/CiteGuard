from citeguard.models import (
    CitationFormat, Verdict, Severity,
    CitationRecord, ClaimRecord, ChunkRecord,
    TokenUsage, VerificationResult,
)

def test_citation_record_defaults():
    r = CitationRecord(id="ref_001", raw_marker="[1]",
                       format=CitationFormat.NUMERIC, position=100)
    assert r.matched_pdf is None
    assert r.reference_text == ""

def test_verification_result_defaults():
    claim = ClaimRecord(citation_id="ref_001", claim_text="test",
                        context_before="", context_after="")
    result = VerificationResult(citation_id="ref_001",
                                timestamp="2026-01-01T00:00:00Z", claim=claim)
    assert result.status == "completed"
    assert result.confidence == 0.0
    assert result.token_usage.total == 0
    assert result.verdict == Verdict.AMBIGUOUS

def test_token_usage_accumulation():
    u = TokenUsage(prompt=100, completion=50, total=150)
    assert u.total == 150

def test_token_usage_add():
    a = TokenUsage(prompt=100, completion=50, total=150)
    b = TokenUsage(prompt=200, completion=80, total=280)
    c = a.add(b)
    assert c.prompt == 300
    assert c.completion == 130
    assert c.total == 430

def test_verdict_enum_values():
    assert Verdict.SUPPORTED == "SUPPORTED"
    assert Verdict.FABRICATED == "FABRICATED"
    assert Verdict.UNVERIFIABLE == "UNVERIFIABLE"

def test_severity_enum_values():
    assert Severity.LOW == "LOW"
    assert Severity.CRITICAL == "CRITICAL"

def test_chunk_record_score_default():
    c = ChunkRecord(pdf_path="a.pdf", chunk_id="c1", text="hello", page=1)
    assert c.score == 0.0

def test_verification_result_evidence_chunks_default_empty():
    claim = ClaimRecord(citation_id="ref_001", claim_text="test")
    result = VerificationResult(citation_id="ref_001", timestamp="2026-01-01T00:00:00Z", claim=claim)
    assert result.evidence_chunks == []
    assert result.issues == []

def test_citation_format_all_values():
    assert CitationFormat.NUMERIC == "numeric"
    assert CitationFormat.AUTHOR_YEAR == "author_year"
    assert CitationFormat.FOOTNOTE == "footnote"
    assert CitationFormat.NAMED_KEY == "named_key"
