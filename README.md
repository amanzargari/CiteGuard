# CiteGuard

A general-purpose agentic academic citation verification system.

CiteGuard checks whether claims made in a manuscript about cited papers are
accurate, semantically consistent, and supported by the referenced sources.

## Features

- Verifies citations across any academic domain and manuscript format
- Handles all citation styles: numeric `[1]`, author-year `(Smith, 2023)`, footnotes, named keys
- Three strictness modes: `lenient`, `balanced`, `strict`
- Three retrieval backends: BM25 (default, zero setup), local embeddings, API embeddings
- Resumable: interrupted runs resume from the last completed citation
- PDF report (ReportLab) + machine-readable JSON
- Powered by Google ADK + OpenRouter (any model)

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env: add your OPENROUTER_API_KEY

cp examples/config.example.yaml config.yaml
# Edit config.yaml: set strictness, models, retrieval_backend
```

### 3. Run

```bash
citeguard verify paper.md pdfs/ --config config.yaml
```

Output is written to `output/`:
- `output/citeguard_report.pdf` — human-readable PDF report
- `output/citeguard_results.json` — machine-readable results

### Resume an interrupted run

```bash
citeguard verify paper.md pdfs/ --resume
```

### Check progress

```bash
citeguard status output/
```

### Re-generate reports only

```bash
citeguard report output/ --format pdf
```

## Configuration

| Option | Values | Default | Description |
|---|---|---|---|
| `strictness` | `lenient`, `balanced`, `strict` | `balanced` | How strictly claims are evaluated |
| `retrieval_backend` | `bm25`, `local_embeddings`, `api_embeddings` | `bm25` | Chunk retrieval method |
| `models.strong` | Any OpenRouter model slug | `anthropic/claude-sonnet-4-5` | Model for final verdict |
| `models.cheap` | Any OpenRouter model slug | `openai/gpt-4o-mini` | Model for pre-filtering |
| `retrieval.top_k` | integer | `5` | Chunks retrieved per query |
| `retrieval.max_requery_rounds` | integer | `3` | Max re-query attempts per citation |
| `token_budget.per_citation` | integer | `4000` | Hard token cap per citation |

Full reference: `examples/config.example.yaml`

## Environment Variables (`.env`)

```
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_API_KEY=...   # only for api_embeddings backend
```

## Architecture

```
CiteGuardRunner (orchestrator)
├── ManuscriptParser     → extracts text from MD/TXT/DOCX
├── CitationExtractor    → detects all citation formats
├── BibliographyParser   → maps markers to reference entries
├── PDFIndexer           → chunks + indexes PDFs; matches citations to PDFs
└── VerificationAgent    → Google ADK LlmAgent with tool-use loop
    ├── retrieve_chunks  → queries the PDF retrieval index
    ├── re_query         → refined search if evidence is weak
    ├── mark_verdict     → records SUPPORTED/PARTIAL/UNSUPPORTED/etc.
    └── save_checkpoint  → persists result to JSON
```

## Verdicts

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | Claim accurately reflects the cited paper |
| `PARTIAL` | Claim is partially correct; important caveats omitted |
| `UNSUPPORTED` | Claim not found in the paper |
| `EXAGGERATED` | Claim overstates the paper's findings |
| `FABRICATED` | Claim describes something not in the paper |
| `AMBIGUOUS` | Insufficient evidence to determine |
| `UNVERIFIABLE` | No matching PDF available |

## Retrieval Backends

| Backend | Setup | Best For |
|---|---|---|
| `bm25` | None (default) | Technical text, exact term matching |
| `local_embeddings` | Downloads ~500 MB model on first use | Semantic paraphrase detection |
| `api_embeddings` | Requires `EMBEDDING_API_KEY` | Highest quality, adds cost |

## Requirements

- Python 3.11+
- OpenRouter API key
- PDFs named to facilitate matching (author+year+keyword recommended)
