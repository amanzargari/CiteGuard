from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn,
)

from citeguard.config import Settings
from citeguard.models import Verdict, VerificationResult, Severity
from citeguard.parsing.manuscript import ManuscriptParser
from citeguard.parsing.citations import CitationExtractor
from citeguard.parsing.claims import ClaimSegmenter, BibliographyParser
from citeguard.agents.pdf_indexer import PDFIndexer
from citeguard.agents.verifier import VerificationAgent
from citeguard.cache.manager import CacheManager

console = Console()

_VERDICT_STYLE: dict[str, str] = {
    Verdict.SUPPORTED: "green",
    Verdict.PARTIAL: "yellow",
    Verdict.UNSUPPORTED: "red",
    Verdict.EXAGGERATED: "orange3",
    Verdict.FABRICATED: "bold red",
    Verdict.AMBIGUOUS: "dim",
    Verdict.UNVERIFIABLE: "blue",
    Verdict.ERROR: "bold magenta",
}


class CiteGuardRunner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache = CacheManager(
            checkpoints_dir=Path(settings.output.checkpoints_dir),
            index_cache_dir=Path(settings.output.index_cache_dir),
        )
        self._indexer = PDFIndexer(settings)
        self._verifier = VerificationAgent(settings)
        self._parser = ManuscriptParser()
        self._citation_extractor = CitationExtractor()
        self._claim_segmenter = ClaimSegmenter(window=2)
        self._bib_parser = BibliographyParser()

    def run(
        self,
        manuscript_path: Path,
        pdf_folder: Path,
        resume: bool = True,
    ) -> list:
        console.rule("[bold blue]CiteGuard")
        console.print(
            f"Manuscript: [cyan]{manuscript_path}[/cyan]  "
            f"PDFs: [cyan]{pdf_folder}[/cyan]  "
            f"Strictness: [yellow]{self._settings.strictness}[/yellow]  "
            f"Backend: [yellow]{self._settings.retrieval_backend}[/yellow]"
        )

        # 1. Parse manuscript
        console.print("\n[bold]Step 1/4:[/bold] Parsing manuscript...")
        text = self._parser.parse(manuscript_path)
        citations = self._citation_extractor.extract(text)
        bib_entries = self._bib_parser.parse(text)
        console.print(f"  Found [green]{len(citations)}[/green] citations")

        # 2. Discover PDFs
        pdf_paths = sorted(pdf_folder.glob("*.pdf"))
        console.print(f"  Found [green]{len(pdf_paths)}[/green] PDFs in {pdf_folder}")

        # 3. Match citations to PDFs
        console.print("\n[bold]Step 2/4:[/bold] Matching citations to PDFs...")
        citations = self._indexer.match_citations_to_pdfs(citations, bib_entries, pdf_paths)
        matched = sum(1 for c in citations if c.matched_pdf)
        console.print(f"  Matched [green]{matched}[/green] / {len(citations)} citations to PDFs")

        # 4. Index PDFs
        console.print("\n[bold]Step 3/4:[/bold] Indexing PDFs...")
        self._indexer.index_all(pdf_paths, self._cache)

        # 5. Determine which citations to verify
        done_ids = self._cache.completed_citation_ids() if resume else set()
        pending = [c for c in citations if c.id not in done_ids]

        if done_ids:
            console.print(
                f"\n[bold]Resuming:[/bold] {len(done_ids)} already verified, "
                f"{len(pending)} remaining"
            )

        # 6. Verify pending citations
        console.print(f"\n[bold]Step 4/4:[/bold] Verifying {len(pending)} citations...\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Verifying", total=len(pending))

            for citation in pending:
                progress.update(
                    task,
                    description=f"Verifying [cyan]{citation.id}[/cyan] {citation.raw_marker[:30]}",
                )
                claim = self._claim_segmenter.segment(text, citation)
                retriever = (
                    self._indexer.get_retriever(citation.matched_pdf)
                    if citation.matched_pdf else None
                )
                try:
                    session = self._verifier.verify(
                        citation=citation,
                        claim=claim,
                        retriever=retriever,
                        cache=self._cache,
                        pdf_paths=[str(p) for p in pdf_paths],
                    )
                    verdict = session.verdict or Verdict.ERROR
                except Exception as e:
                    console.print(f"  [red]ERROR[/red] {citation.id}: {e}")
                    verdict = Verdict.ERROR
                    error_result = VerificationResult(
                        citation_id=citation.id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        claim=claim,
                        matched_pdf=citation.matched_pdf,
                        verdict=Verdict.ERROR,
                        confidence=0.0,
                        severity=Severity.HIGH,
                        reasoning=str(e),
                        issues=[str(e)],
                        re_query_count=0,
                    )
                    self._cache.save_checkpoint(error_result)

                style = _VERDICT_STYLE.get(verdict, "white")
                console.print(
                    f"  [{style}]{verdict.value:<14}[/{style}] "
                    f"[dim]{citation.id}[/dim] {citation.raw_marker[:40]}"
                )
                progress.advance(task)

        return self._cache.all_results()
