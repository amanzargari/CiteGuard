from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class CitationFormat(str, Enum):
    NUMERIC = "numeric"
    AUTHOR_YEAR = "author_year"
    FOOTNOTE = "footnote"
    NAMED_KEY = "named_key"


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    EXAGGERATED = "EXAGGERATED"
    FABRICATED = "FABRICATED"
    AMBIGUOUS = "AMBIGUOUS"
    UNVERIFIABLE = "UNVERIFIABLE"
    ERROR = "ERROR"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CitationRecord(BaseModel):
    id: str
    raw_marker: str
    format: CitationFormat
    position: int
    matched_pdf: str | None = None
    reference_text: str = ""


class ClaimRecord(BaseModel):
    citation_id: str
    claim_text: str
    context_before: str = ""
    context_after: str = ""


class ChunkRecord(BaseModel):
    pdf_path: str
    chunk_id: str
    text: str
    page: int
    score: float = 0.0


class TokenUsage(BaseModel):
    prompt: int = 0
    completion: int = 0
    total: int = 0

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt=self.prompt + other.prompt,
            completion=self.completion + other.completion,
            total=self.total + other.total,
        )


class VerificationResult(BaseModel):
    citation_id: str
    status: str = "completed"
    timestamp: str
    claim: ClaimRecord
    matched_pdf: str | None = None
    evidence_chunks: list[ChunkRecord] = Field(default_factory=list)
    verdict: Verdict = Verdict.AMBIGUOUS
    confidence: float = 0.0
    severity: Severity = Severity.MEDIUM
    reasoning: str = ""
    issues: list[str] = Field(default_factory=list)
    re_query_count: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
