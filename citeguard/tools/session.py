from __future__ import annotations
from citeguard.models import (
    CitationRecord, ClaimRecord, ChunkRecord,
    Verdict, Severity, TokenUsage,
)


class VerificationSession:
    """Mutable state for a single citation verification run, shared across tools."""

    def __init__(self, citation: CitationRecord, claim: ClaimRecord) -> None:
        self.citation = citation
        self.claim = claim
        self.evidence_chunks: list[ChunkRecord] = []
        self.verdict: Verdict | None = None
        self.confidence: float = 0.0
        self.severity: Severity = Severity.MEDIUM
        self.reasoning: str = ""
        self.issues: list[str] = []
        self.re_query_count: int = 0
        self.token_usage: TokenUsage = TokenUsage()
