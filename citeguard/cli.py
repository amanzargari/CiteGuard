# citeguard/cli.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

console = Console()


@click.group()
def cli():
    """CiteGuard — Academic citation verification agent."""


@cli.command()
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.argument("pdf_folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "-c", type=click.Path(path_type=Path), default=None,
              help="Path to config.yaml")
@click.option("--strictness", type=click.Choice(["lenient", "balanced", "strict"]),
              default=None, help="Override strictness mode")
@click.option("--retrieval-backend",
              type=click.Choice(["bm25", "local_embeddings", "api_embeddings"]),
              default=None, help="Override retrieval backend")
@click.option("--resume/--no-resume", default=True,
              help="Resume from existing checkpoints (default: true)")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Override output directory")
def verify(
    manuscript: Path,
    pdf_folder: Path,
    config: Optional[Path],
    strictness: Optional[str],
    retrieval_backend: Optional[str],
    resume: bool,
    output_dir: Optional[Path],
):
    """Verify citation claims in MANUSCRIPT against PDFs in PDF_FOLDER."""
    from citeguard.config import load_settings
    from citeguard.agents.orchestrator import CiteGuardRunner
    from citeguard.reporting.json_report import generate_json_report
    from citeguard.reporting.pdf_report import generate_pdf_report

    settings = load_settings(config)
    if strictness:
        settings = settings.model_copy(update={"strictness": strictness})
    if retrieval_backend:
        settings = settings.model_copy(update={"retrieval_backend": retrieval_backend})
    if output_dir:
        settings = settings.model_copy(update={
            "output": settings.output.model_copy(update={
                "dir": str(output_dir),
                "checkpoints_dir": str(output_dir / "checkpoints"),
                "index_cache_dir": str(output_dir / "index_cache"),
            })
        })

    runner = CiteGuardRunner(settings)
    results = runner.run(manuscript, pdf_folder, resume=resume)

    out_dir = Path(settings.output.dir)
    json_path = out_dir / "citeguard_results.json"
    pdf_path = out_dir / "citeguard_report.pdf"

    report = generate_json_report(
        results=results,
        settings=settings,
        manuscript_path=str(manuscript),
        pdf_folder=str(pdf_folder),
        output_path=json_path,
    )

    generate_pdf_report(
        results=results,
        output_path=pdf_path,
        manuscript_name=manuscript.name,
        strictness=settings.strictness,
    )

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  JSON report: [cyan]{json_path}[/cyan]")
    console.print(f"  PDF report:  [cyan]{pdf_path}[/cyan]")
    summary = report["summary"]
    console.print(
        f"  Verdicts: "
        + "  ".join(f"{k}: {v}" for k, v in summary["verdicts"].items())
    )


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "fmt", type=click.Choice(["pdf", "json", "both"]),
              default="both")
def report(output_dir: Path, fmt: str):
    """Re-generate reports from existing checkpoints in OUTPUT_DIR."""
    from citeguard.config import load_settings
    from citeguard.cache.manager import CacheManager
    from citeguard.reporting.json_report import generate_json_report
    from citeguard.reporting.pdf_report import generate_pdf_report

    settings = load_settings()
    cache = CacheManager(
        checkpoints_dir=output_dir / "checkpoints",
        index_cache_dir=output_dir / "index_cache",
    )
    results = cache.all_results()
    console.print(f"Loaded {len(results)} results from {output_dir / 'checkpoints'}")

    if fmt in ("json", "both"):
        json_path = output_dir / "citeguard_results.json"
        generate_json_report(results, settings, "unknown", "unknown", json_path)
        console.print(f"JSON report: [cyan]{json_path}[/cyan]")

    if fmt in ("pdf", "both"):
        pdf_path = output_dir / "citeguard_report.pdf"
        generate_pdf_report(results, pdf_path, "report", settings.strictness)
        console.print(f"PDF report:  [cyan]{pdf_path}[/cyan]")


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
def status(output_dir: Path):
    """Show verification progress for a run in OUTPUT_DIR."""
    from citeguard.cache.manager import CacheManager
    from citeguard.models import Verdict
    from collections import Counter
    from rich.table import Table

    cache = CacheManager(
        checkpoints_dir=output_dir / "checkpoints",
        index_cache_dir=output_dir / "index_cache",
    )
    results = cache.all_results()
    if not results:
        console.print("[yellow]No checkpoints found.[/yellow]")
        return

    counts = Counter(r.verdict for r in results)
    table = Table(title=f"Status — {len(results)} citations verified")
    table.add_column("Verdict")
    table.add_column("Count", justify="right")
    for verdict, count in sorted(counts.items(), key=lambda x: -x[1]):
        table.add_row(verdict.value, str(count))
    console.print(table)
