# citeguard/reporting/pdf_report.py
from __future__ import annotations
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

from citeguard.models import VerificationResult, Verdict, Severity

_VERDICT_COLOR = {
    Verdict.SUPPORTED: colors.HexColor("#27ae60"),
    Verdict.PARTIAL: colors.HexColor("#f39c12"),
    Verdict.UNSUPPORTED: colors.HexColor("#e74c3c"),
    Verdict.EXAGGERATED: colors.HexColor("#e67e22"),
    Verdict.FABRICATED: colors.HexColor("#c0392b"),
    Verdict.AMBIGUOUS: colors.HexColor("#95a5a6"),
    Verdict.UNVERIFIABLE: colors.HexColor("#3498db"),
    Verdict.ERROR: colors.HexColor("#8e44ad"),
}

_SEVERITY_LABEL = {
    Severity.LOW: "Low",
    Severity.MEDIUM: "Medium",
    Severity.HIGH: "High",
    Severity.CRITICAL: "CRITICAL",
}


def _build_summary_chart(verdict_counts: Counter) -> Drawing:
    verdicts = list(verdict_counts.keys())
    counts = [verdict_counts[v] for v in verdicts]
    d = Drawing(400, 160)
    chart = VerticalBarChart()
    chart.x = 40
    chart.y = 20
    chart.width = 340
    chart.height = 120
    chart.data = [counts]
    chart.categoryAxis.categoryNames = verdicts
    chart.categoryAxis.labels.angle = 30
    chart.categoryAxis.labels.dy = -10
    chart.bars[0].fillColor = colors.HexColor("#2980b9")
    d.add(chart)
    return d


def generate_pdf_report(
    results: list[VerificationResult],
    output_path: Path,
    manuscript_name: str,
    strictness: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=20, spaceAfter=12)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceAfter=4)
    body = styles["BodyText"]
    small = ParagraphStyle("Small", parent=body, fontSize=8)

    story = []

    # Title page
    story.append(Paragraph("CiteGuard Verification Report", title_style))
    story.append(Paragraph(f"Manuscript: {manuscript_name}", body))
    story.append(Paragraph(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", body))
    story.append(Paragraph(f"Strictness: {strictness}", body))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%"))
    story.append(Spacer(1, 0.5 * cm))

    # Summary statistics
    verdict_counts = Counter(r.verdict for r in results)
    story.append(Paragraph("Summary Statistics", h1))

    summary_data = [["Metric", "Count"]]
    summary_data.append(["Total citations", str(len(results))])
    for verdict in Verdict:
        n = verdict_counts.get(verdict, 0)
        if n > 0:
            summary_data.append([verdict.value, str(n)])
    t = Table(summary_data, colWidths=[10 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    # Bar chart
    story.append(_build_summary_chart(Counter(r.verdict.value for r in results)))
    story.append(PageBreak())

    # Issues requiring attention
    priority_verdicts = [Verdict.FABRICATED, Verdict.UNSUPPORTED, Verdict.EXAGGERATED, Verdict.PARTIAL]
    for verdict in priority_verdicts:
        group = [r for r in results if r.verdict == verdict]
        if not group:
            continue
        story.append(Paragraph(f"{verdict.value} ({len(group)})", h1))
        for r in group:
            story.append(Paragraph(
                f"<b>{r.citation_id}</b> — {r.claim.claim_text[:120]}...",
                body,
            ))
            story.append(Paragraph(
                f"Confidence: {r.confidence:.0%}  |  Severity: {_SEVERITY_LABEL[r.severity]}  |  "
                f"PDF: {r.matched_pdf or 'N/A'}",
                small,
            ))
            if r.reasoning:
                story.append(Paragraph(f"<i>{r.reasoning[:300]}</i>", small))
            if r.issues:
                story.append(Paragraph("Issues: " + "; ".join(r.issues[:3]), small))
            story.append(Spacer(1, 0.3 * cm))
        story.append(Spacer(1, 0.2 * cm))

    # Full results table
    story.append(PageBreak())
    story.append(Paragraph("All Results", h1))

    table_data = [["ID", "Verdict", "Confidence", "Re-queries", "PDF"]]
    for r in results:
        table_data.append([
            r.citation_id,
            r.verdict.value,
            f"{r.confidence:.0%}",
            str(r.re_query_count),
            (Path(r.matched_pdf).name[:30] if r.matched_pdf else "—"),
        ])

    tbl = Table(table_data, colWidths=[3 * cm, 4 * cm, 2.5 * cm, 2.5 * cm, 5 * cm])
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
    ])
    for i, r in enumerate(results, start=1):
        bg = _VERDICT_COLOR.get(r.verdict, colors.white)
        style.add("BACKGROUND", (1, i), (1, i), bg)
        style.add("TEXTCOLOR", (1, i), (1, i), colors.white)
    tbl.setStyle(style)
    story.append(tbl)

    doc.build(story)
