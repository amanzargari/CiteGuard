from __future__ import annotations
from datetime import datetime, timezone
from citeguard.models import Verdict, Severity, VerificationResult
from citeguard.tools.session import VerificationSession
from citeguard.cache.manager import CacheManager


def make_verdict_tools(session: VerificationSession, cache: CacheManager):
    """Return (mark_verdict, save_checkpoint) tool functions."""

    def mark_verdict(
        verdict: str,
        confidence: float,
        reasoning: str,
        issues: list[str] | None = None,
        severity: str = "MEDIUM",
    ) -> dict:
        """Record the final verification verdict.

        Args:
            verdict: One of SUPPORTED, PARTIAL, UNSUPPORTED, EXAGGERATED,
                     FABRICATED, AMBIGUOUS, UNVERIFIABLE.
            confidence: Float 0.0-1.0 indicating certainty.
            reasoning: Explanation of the verdict.
            issues: List of specific issues found (empty if SUPPORTED).
            severity: LOW, MEDIUM, HIGH, or CRITICAL.

        Returns:
            dict confirming verdict was recorded.
        """
        try:
            session.verdict = Verdict(verdict.upper())
        except ValueError:
            session.verdict = Verdict.AMBIGUOUS
        session.confidence = max(0.0, min(1.0, float(confidence)))
        session.reasoning = reasoning
        session.issues = issues or []
        try:
            session.severity = Severity(severity.upper())
        except ValueError:
            session.severity = Severity.MEDIUM
        return {"status": "verdict_recorded", "verdict": session.verdict.value}

    def save_checkpoint() -> dict:
        """Persist the current verification result to disk.

        Returns:
            dict confirming the checkpoint was saved.
        """
        if session.verdict is None:
            session.verdict = Verdict.AMBIGUOUS

        result = VerificationResult(
            citation_id=session.citation.id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            claim=session.claim,
            matched_pdf=session.citation.matched_pdf,
            evidence_chunks=session.evidence_chunks,
            verdict=session.verdict,
            confidence=session.confidence,
            severity=session.severity,
            reasoning=session.reasoning,
            issues=session.issues,
            re_query_count=session.re_query_count,
            token_usage=session.token_usage,
        )
        cache.save_checkpoint(result)
        return {"status": "saved", "citation_id": session.citation.id}

    return mark_verdict, save_checkpoint
