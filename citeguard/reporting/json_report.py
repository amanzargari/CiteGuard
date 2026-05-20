# citeguard/reporting/json_report.py
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from citeguard.config import Settings
from citeguard.models import VerificationResult, Verdict, Severity, TokenUsage


def generate_json_report(
    results: list[VerificationResult],
    settings: Settings,
    manuscript_path: str,
    pdf_folder: str,
    output_path: Path,
) -> dict:
    verdict_counts = Counter(r.verdict.value for r in results)
    total_usage = TokenUsage()
    for r in results:
        total_usage = total_usage.add(r.token_usage)

    confidence_buckets = {"high": 0, "medium": 0, "low": 0}
    for r in results:
        if r.confidence >= 0.75:
            confidence_buckets["high"] += 1
        elif r.confidence >= 0.40:
            confidence_buckets["medium"] += 1
        else:
            confidence_buckets["low"] += 1

    unverifiable = sum(1 for r in results if r.verdict == Verdict.UNVERIFIABLE)
    verified = len(results) - unverifiable

    report = {
        "metadata": {
            "manuscript": manuscript_path,
            "pdf_folder": pdf_folder,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strictness": settings.strictness,
            "retrieval_backend": settings.retrieval_backend,
            "models": settings.models.model_dump(),
        },
        "summary": {
            "total_citations": len(results),
            "verified": verified,
            "unverifiable": unverifiable,
            "verdicts": dict(verdict_counts),
            "confidence_distribution": confidence_buckets,
            "token_usage": total_usage.model_dump(),
        },
        "results": [r.model_dump() for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
