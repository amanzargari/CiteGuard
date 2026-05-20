from __future__ import annotations
import re
from citeguard.models import CitationRecord, ClaimRecord

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

_BIB_HEADERS = re.compile(
    r'(?:^|\n)(?:References|Bibliography|Works Cited|Literature Cited)\s*\n',
    re.IGNORECASE,
)
_NUMERIC_ENTRY = re.compile(r'^\[(\d+)\]\s+(.+)$', re.MULTILINE)
_NAMED_ENTRY = re.compile(
    r'^\[([a-zA-Z][a-zA-Z0-9]+\d{4}[a-zA-Z0-9]*)\]\s+(.+)$', re.MULTILINE
)
_AUTHORYEAR_ENTRY = re.compile(
    r'^([A-Z][a-zA-Z\-\']+(?:,\s[A-Z]\.?)+(?:\s+et\s+al\.?)?)\s+\((\d{4})\)\.\s+(.+)$',
    re.MULTILINE,
)


def _split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Return list of (sentence_text, start, end) tuples."""
    results = []
    pos = 0
    for match in _SENT_SPLIT.finditer(text):
        sent = text[pos:match.start()].strip()
        if sent:
            results.append((sent, pos, match.start()))
        pos = match.end()
    tail = text[pos:].strip()
    if tail:
        results.append((tail, pos, len(text)))
    return results


class ClaimSegmenter:
    def __init__(self, window: int = 2) -> None:
        self._window = window

    def segment(self, text: str, citation: CitationRecord) -> ClaimRecord:
        spans = _split_sentences_with_spans(text)
        if not spans:
            return ClaimRecord(citation_id=citation.id, claim_text="")

        pos = citation.position
        target_idx = len(spans) - 1  # default to last sentence

        for i, (sent, start, end) in enumerate(spans):
            if start <= pos <= end:
                target_idx = i
                break

        before_start = max(0, target_idx - self._window)
        after_end = min(len(spans), target_idx + self._window + 1)

        claim_text = spans[target_idx][0]
        context_before = " ".join(s for s, _, _ in spans[before_start:target_idx])
        context_after = " ".join(s for s, _, _ in spans[target_idx + 1:after_end])

        return ClaimRecord(
            citation_id=citation.id,
            claim_text=claim_text,
            context_before=context_before,
            context_after=context_after,
        )


class BibliographyParser:
    """Extract the reference list and build a marker → entry text dict."""

    def parse(self, text: str) -> dict[str, str]:
        match = _BIB_HEADERS.search(text)
        if not match:
            return {}
        bib_section = text[match.end():]
        entries: dict[str, str] = {}

        for m in _NUMERIC_ENTRY.finditer(bib_section):
            entries[f"[{m.group(1)}]"] = m.group(2).strip()

        for m in _NAMED_ENTRY.finditer(bib_section):
            entries[f"[{m.group(1)}]"] = m.group(2).strip()

        for m in _AUTHORYEAR_ENTRY.finditer(bib_section):
            key = f"({m.group(1)}, {m.group(2)})"
            entries[key] = m.group(3).strip()

        return entries
