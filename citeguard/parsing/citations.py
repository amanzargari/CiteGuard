from __future__ import annotations
import re
from citeguard.models import CitationRecord, CitationFormat

# Order matters: named_key FIRST to avoid [smith2023] being consumed by numeric pattern
_PATTERNS: list[tuple[re.Pattern, CitationFormat]] = [
    # Named key: [smith2023nature] — must come before numeric to avoid [1] eating it
    (re.compile(r'\[([a-zA-Z][a-zA-Z0-9]{2,}\d{4}[a-zA-Z0-9]*)\]'), CitationFormat.NAMED_KEY),
    # Numeric range: [1-3] or [1–3]
    (re.compile(r'\[(\d+[-–]\d+)\]'), CitationFormat.NUMERIC),
    # Numeric multiple: [1,2,3] or [1, 2]
    (re.compile(r'\[(\d+(?:,\s*\d+)+)\]'), CitationFormat.NUMERIC),
    # Numeric single: [42]
    (re.compile(r'\[(\d+)\]'), CitationFormat.NUMERIC),
    # Author-year paren: (Smith et al., 2023) or (Smith & Jones, 2020; Lee, 2021)
    (re.compile(
        r'\(([A-Z][a-zA-Z\-\']+(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][a-zA-Z\-\']+)?'
        r',\s*\d{4}[a-z]?(?:;\s*[A-Z][a-zA-Z\-\']+(?:\s+et\s+al\.?)?'
        r'(?:\s+(?:and|&)\s+[A-Z][a-zA-Z\-\']+)?,\s*\d{4}[a-z]?)*)\)'
    ), CitationFormat.AUTHOR_YEAR),
    # Author-year inline: Smith et al. (2023) or Jones (2020)
    (re.compile(
        r'([A-Z][a-zA-Z\-\']+(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][a-zA-Z\-\']+)?)'
        r'\s+\((\d{4}[a-z]?)\)'
    ), CitationFormat.AUTHOR_YEAR),
    # Unicode superscripts ¹²³
    (re.compile(r'([¹²³⁴-⁹⁰]+)'), CitationFormat.FOOTNOTE),
    # ASCII footnote symbols
    (re.compile(r'([†‡\xa7\xb6]+)'), CitationFormat.FOOTNOTE),
]


class CitationExtractor:
    def extract(self, text: str) -> list[CitationRecord]:
        found: list[CitationRecord] = []
        seen_spans: set[tuple[int, int]] = set()
        counter = 0

        for pattern, fmt in _PATTERNS:
            for match in pattern.finditer(text):
                span = match.span()
                # Skip if overlaps with already-found citation
                if any(span[0] < e and span[1] > s for s, e in seen_spans):
                    continue
                seen_spans.add(span)

                # Author-year inline has two capture groups: author + year
                if fmt == CitationFormat.AUTHOR_YEAR and len(match.groups()) == 2:
                    raw = f"{match.group(1)} ({match.group(2)})"
                else:
                    raw = match.group(0)

                counter += 1
                found.append(CitationRecord(
                    id=f"ref_{counter:04d}",
                    raw_marker=raw,
                    format=fmt,
                    position=span[0],
                ))

        found.sort(key=lambda c: c.position)
        return found
