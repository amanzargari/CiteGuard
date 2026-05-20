# CiteGuard — Design Specification

**Date:** 2026-05-20
**Status:** Approved
**Project:** CiteGuard — General-Purpose Academic Citation Verification Agent

---

## 1. Overview

CiteGuard is an agentic system that verifies whether claims made in academic manuscripts about cited papers are accurate and supported by the referenced sources. It is designed to work across any academic domain, citation style, and manuscript format.

### Target Use Cases
- Literature reviews and survey papers
- Review articles and meta-analyses
- Related work sections
- Comparative analysis sections

### Scale Target
100–200+ citations per manuscript without performance degradation.

---

## 2. Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Agent Framework | Google ADK (`google-adk`) |
| LLM Access | OpenRouter API (OpenAI-compatible) |
| PDF Extraction | PyMuPDF (`pymupdf`) |
| Retrieval (BM25) | `rank_bm25` |
| Retrieval (local embeddings) | `sentence-transformers` |
| Retrieval (API embeddings) | OpenRouter embedding endpoint |
| DOCX parsing | `python-docx` |
| Progress UI | `rich` |
| PDF Report | `reportlab` (Platypus layout engine) |
| Config validation | `pydantic-settings` |
| CLI | `click` |

### Environment Variables (`.env` only)
```
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_API_KEY=...        # only if retrieval_backend = api_embeddings
```

All other settings live in `config.yaml`.

---

## 3. Architecture

### Approach
**Orchestrator + Specialist Sub-Agents** (Approach B).

A root `OrchestratorAgent` coordinates three specialist agents. The VerificationAgent runs an iterative reasoning loop per citation with tool use. All state is persisted to JSON checkpoint files for resumability.

### Component Map

```
CLI Entry Point
    └── Config Manager
            └── Orchestrator Agent (ADK root)
                    ├── ManuscriptParserAgent  (one-shot)
                    ├── PDFIndexerAgent        (one-shot, skips cached)
                    ├── VerificationAgent      (per-citation loop)
                    │       ├── tool: retrieve_chunks
                    │       ├── tool: re_query
                    │       ├── tool: get_paper_metadata
                    │       ├── tool: mark_verdict
                    │       └── tool: save_checkpoint
                    └── ReportGenerator        (one-shot, post-verification)
```

### Supporting Infrastructure
- **Cache Layer** — JSON checkpoint files + pickled PDF index files
- **Progress UI** — `rich` live display with per-citation status
- **OpenRouter Client** — retry, rate-limit, token tracking

---

## 4. Directory Structure

```
citeguard/
├── citeguard/
│   ├── __init__.py
│   ├── cli.py                    # Click CLI entry point
│   ├── config.py                 # Pydantic Settings — merges config.yaml + env
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # Root ADK OrchestratorAgent
│   │   ├── manuscript_parser.py  # ManuscriptParserAgent + tools
│   │   ├── pdf_indexer.py        # PDFIndexerAgent + tools
│   │   ├── verifier.py           # VerificationAgent + per-citation loop
│   │   └── report_generator.py   # ReportGeneratorAgent
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── retrieve.py           # retrieve_chunks, re_query
│   │   ├── verdict.py            # mark_verdict, save_checkpoint
│   │   └── metadata.py           # get_paper_metadata
│   │
│   ├── parsing/
│   │   ├── __init__.py
│   │   ├── manuscript.py         # MD/TXT/DOCX → plain text
│   │   ├── citations.py          # Multi-format citation extractor
│   │   └── claims.py             # Claim segmenter (sentence window)
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract RetrieverBase
│   │   ├── bm25.py               # BM25 backend (default)
│   │   ├── local_embeddings.py   # sentence-transformers backend
│   │   └── api_embeddings.py     # OpenRouter/API embedding backend
│   │
│   ├── pdf/
│   │   ├── __init__.py
│   │   ├── extractor.py          # PDF text extraction via pymupdf
│   │   └── chunker.py            # Sliding window chunker
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   └── manager.py            # Checkpoint r/w, index cache, dedup
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── openrouter.py         # OpenRouter client, retry, rate-limit, token tracking
│   │
│   └── reporting/
│       ├── __init__.py
│       ├── json_report.py        # Structured JSON aggregation
│       └── pdf_report.py         # ReportLab PDF — tables, charts, color verdicts
│
├── tests/
│   ├── __init__.py
│   ├── test_parsing/
│   ├── test_retrieval/
│   ├── test_verification/
│   └── fixtures/
├── docs/
│   └── superpowers/specs/
├── examples/
│   ├── config.example.yaml
│   └── sample_output/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 5. Data Models

### Core Types

```python
class CitationFormat(str, Enum):
    NUMERIC = "numeric"           # [1], [23], (1)
    AUTHOR_YEAR = "author_year"   # (Smith et al., 2023)
    FOOTNOTE = "footnote"         # ¹, †, *
    NAMED_KEY = "named_key"       # [smith2023]

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
    id: str                        # e.g. "ref_042" or "smith2023"
    raw_marker: str                # e.g. "[42]" or "(Smith et al., 2023)"
    format: CitationFormat
    position: int                  # character offset in manuscript
    matched_pdf: str | None        # path to matched PDF, None if unmatched

class ClaimRecord(BaseModel):
    citation_id: str
    claim_text: str                # 2-3 sentence window around citation
    context_before: str
    context_after: str

class ChunkRecord(BaseModel):
    pdf_path: str
    chunk_id: str
    text: str
    page: int
    score: float

class TokenUsage(BaseModel):
    prompt: int
    completion: int
    total: int

class VerificationResult(BaseModel):
    citation_id: str
    status: str                    # "completed" | "error" | "skipped"
    timestamp: str
    claim: ClaimRecord
    matched_pdf: str | None
    evidence_chunks: list[ChunkRecord]
    verdict: Verdict
    confidence: float              # 0.0–1.0
    severity: Severity
    reasoning: str
    issues: list[str]
    re_query_count: int
    token_usage: TokenUsage
```

---

## 6. Agent Interaction & Tool Protocol

### Orchestrator Startup Sequence

1. Load config, validate env vars
2. Scan `output/checkpoints/` — collect completed citation IDs
3. Run `ManuscriptParserAgent` → get all citations + claims
4. Cross-reference with checkpoints → compute `remaining = all - completed`
5. Run `PDFIndexerAgent` → index all PDFs (skip already-cached)
6. For each citation in `remaining`: run `VerificationAgent`
7. Run `ReportGenerator` → write `citeguard_results.json` + `citeguard_report.pdf`

### VerificationAgent Tool Loop

```
Round 1:
  retrieve_chunks(citation_id, query=claim_text, top_k=5)
  → list[ChunkRecord]
  → agent evaluates evidence quality

If sufficient evidence:
  mark_verdict(...)
  save_checkpoint(citation_id)
  → DONE

If evidence weak/ambiguous (up to 3 rounds total):
  re_query(citation_id, refined_query, reason)
  → list[ChunkRecord] (top_k=8)
  → agent reasons again

If Round 3 reached with no verdict:
  mark_verdict(verdict=AMBIGUOUS, confidence=low)
  save_checkpoint(citation_id)
  → DONE
```

### Citation-to-PDF Matching (layered)

1. Parse bibliography section from manuscript → map marker to entry text
2. Fuzzy filename match (normalized title/author vs. PDF filename)
3. PDF metadata match (title/author fields in PDF)
4. First-page text match (title string comparison)
5. Fallback: `matched_pdf=null`, verdict=`UNVERIFIABLE`

---

## 7. Retrieval Backends

Configured via `config.yaml` → `retrieval_backend: bm25 | local_embeddings | api_embeddings`

| Backend | Setup | Best For |
|---|---|---|
| `bm25` | Zero setup, default | Technical text, exact term overlap |
| `local_embeddings` | Downloads ~500MB model once | Semantic paraphrase detection |
| `api_embeddings` | Requires embedding API key | Highest quality, adds cost |

All backends implement `RetrieverBase`:
```python
class RetrieverBase(ABC):
    def index(self, chunks: list[ChunkRecord]) -> None: ...
    def query(self, text: str, top_k: int) -> list[ChunkRecord]: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

---

## 8. LLM Model Strategy

### Two-Tier Routing

| Task | Tier | Default Model |
|---|---|---|
| Chunk relevance pre-filter | cheap | `openai/gpt-4o-mini` |
| Citation format detection | cheap | `openai/gpt-4o-mini` |
| Final verdict reasoning | strong | `anthropic/claude-sonnet-4-5` |
| Report summarization | medium | `openai/gpt-4o` |

Configured in `config.yaml`:
```yaml
models:
  cheap: "openai/gpt-4o-mini"
  medium: "openai/gpt-4o"
  strong: "anthropic/claude-sonnet-4-5"
```

### Token Optimization

1. Only top-k retrieved chunks (≤500 tokens each, k=5 default) enter the verification prompt — never full PDFs
2. PDF indexing is done once; cached index is reused across runs
3. System prompt is constant per run — eligible for OpenRouter cache-control
4. Cheap model pre-filters chunks by relevance score before strong model sees them
5. Claim extracted as 2–3 sentence window (~80–150 tokens)
6. Configurable token budget per citation (default: 4000 tokens); forces verdict on current evidence if exceeded
7. Checkpointed citations cost zero tokens on resume

### Rate Limit Handling

- Exponential backoff with jitter: 2s → 4s → 8s → 16s → 32s on 429/503
- Per-minute sliding window token counter
- Configurable `requests_per_minute` and `tokens_per_minute` caps
- 5 consecutive failures → mark citation as `ERROR`, continue

---

## 9. Strictness Modes

| Mode | Behavior |
|---|---|
| `lenient` | Acceptable paraphrase counts as SUPPORTED; only clear factual errors flagged |
| `balanced` | Distinguishes acceptable paraphrase from interpretation drift; flags exaggeration |
| `strict` | Any claim not directly evidenced by verbatim/near-verbatim text is flagged |

Strictness injects a different instruction block into the VerificationAgent system prompt. It does not change which model is used.

---

## 10. Output

### Files Written

```
output/
├── checkpoints/              # one JSON per citation (live during run)
│   ├── ref_001.json
│   └── ...
├── index_cache/              # pickled retrieval indexes per PDF
│   └── ...
├── citeguard_results.json    # aggregated machine-readable results
└── citeguard_report.pdf      # human-readable PDF report (ReportLab)
```

### Checkpoint JSON (per citation)
```json
{
  "citation_id": "ref_042",
  "status": "completed",
  "timestamp": "2026-05-20T14:32:11Z",
  "claim": "...",
  "matched_pdf": "pdfs/smith2023.pdf",
  "evidence_chunks": [...],
  "verdict": "SUPPORTED",
  "confidence": 0.87,
  "severity": "LOW",
  "reasoning": "...",
  "issues": [],
  "re_query_count": 1,
  "token_usage": {"prompt": 1240, "completion": 380, "total": 1620}
}
```

### Aggregated JSON Report
```json
{
  "metadata": { "manuscript": "...", "timestamp": "...", "strictness": "...", ... },
  "summary": {
    "total_citations": 148,
    "verified": 142,
    "unverifiable": 6,
    "verdicts": { "SUPPORTED": 98, "PARTIAL": 22, ... },
    "confidence_distribution": { "high": 87, "medium": 41, "low": 20 },
    "token_usage": { "prompt": 284120, "completion": 94300, "total": 378420 }
  },
  "results": [ ...VerificationResult objects... ]
}
```

### PDF Report Structure (ReportLab)
- Page 1: Title, metadata, summary statistics, verdict distribution bar chart
- Page 2+: Color-coded verdict table (green=SUPPORTED, yellow=PARTIAL, orange=UNSUPPORTED/EXAGGERATED, red=FABRICATED/CRITICAL)
- Issues section: grouped by severity (CRITICAL → HIGH → MEDIUM → LOW)
- Ambiguous findings section
- Unverifiable citations section
- Page footer: run metadata, token usage, estimated cost

---

## 11. Error Handling

| Failure | Response |
|---|---|
| PDF can't be parsed | Warn, mark citation UNVERIFIABLE, continue |
| No PDF match for citation | `matched_pdf=null`, verdict UNVERIFIABLE, continue |
| LLM returns malformed output | Retry up to 3× with schema reminder, then mark ERROR |
| Rate limit | Exponential backoff, then continue |
| 5 consecutive LLM failures | Save checkpoint as ERROR, skip, log |
| Retrieval returns 0 chunks | Re-query with broader terms; if empty → UNVERIFIABLE |
| Process killed | Completed checkpoints survive; restart resumes automatically |
| Token budget exceeded | Force verdict on current evidence, flag LOW_EVIDENCE |
| Missing PDF folder | Hard fail at startup before any work |
| Invalid config.yaml | Hard fail at startup with pydantic field-level error |

---

## 12. Progress UI

```
CiteGuard v0.1.0  |  balanced  |  bm25

 Indexing PDFs  ████████████████████░░░░  82/100  [00:34]

 Verifying citations
 ├─ ✅ ref_001  SUPPORTED      (0.91)  1 round   320 tok
 ├─ ⚠️  ref_002  PARTIAL        (0.74)  2 rounds  580 tok
 ├─ ❌ ref_003  UNSUPPORTED    (0.88)  1 round   290 tok
 ├─ 🔄 ref_004  [verifying...]
 └─ ░░ ref_005..148  pending

 Progress: 3/148  ETA: ~47 min  |  Tokens: 1,190  |  Est. cost: $0.001
```

Resume detection at startup:
```
Found 148 citations.
Found 67 existing checkpoints — resuming from ref_068.
Starting verification for 75 remaining citations...
```

---

## 13. Configuration Reference (`config.yaml`)

```yaml
strictness: balanced          # lenient | balanced | strict
retrieval_backend: bm25       # bm25 | local_embeddings | api_embeddings

models:
  cheap: "openai/gpt-4o-mini"
  medium: "openai/gpt-4o"
  strong: "anthropic/claude-sonnet-4-5"

retrieval:
  top_k: 5
  top_k_requery: 8
  max_requery_rounds: 3
  chunk_size: 500             # tokens per chunk
  chunk_overlap: 50

token_budget:
  per_citation: 4000          # hard cap; forces verdict on current evidence

rate_limits:
  requests_per_minute: 60
  tokens_per_minute: 100000

output:
  dir: "output/"
  checkpoints_dir: "output/checkpoints/"
  index_cache_dir: "output/index_cache/"

local_embeddings:
  model: "all-MiniLM-L6-v2"  # sentence-transformers model name
```

---

## 14. CLI Interface

```bash
# Basic usage
citeguard verify manuscript.md pdfs/ --config config.yaml

# Override strictness
citeguard verify manuscript.md pdfs/ --strictness strict

# Override retrieval backend
citeguard verify manuscript.md pdfs/ --retrieval-backend local_embeddings

# Resume a previous run
citeguard verify manuscript.md pdfs/ --resume

# Generate report only (from existing checkpoints)
citeguard report output/ --format pdf

# Show progress of a running job
citeguard status output/
```

---

## 15. Design Decisions

| Decision | Rationale |
|---|---|
| Google ADK as framework | Requested; provides native tool-use loop, session management, agent coordination |
| Approach B (Orchestrator + Specialists) | Maps cleanly to ADK patterns; each agent has one purpose; adaptive re-querying is natural |
| JSON checkpoints (not SQLite) | Simpler, human-inspectable, crash-safe, zero infrastructure |
| BM25 as default retrieval | Zero setup, fast, works well for technical text; semantic backends optional |
| PDF report via ReportLab | User preference; richer formatting than Markdown for researcher consumption |
| Two-tier model routing | Cheap model for classification saves significant cost at 150+ citation scale |
| `.env` for keys only | Clean separation: secrets in `.env`, behavior in `config.yaml` |
| Max 3 re-query rounds | Balances thoroughness against token cost; ambiguous cases are marked not retried infinitely |
