import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from citeguard.agents.verifier import VerificationAgent, _SYSTEM_PROMPTS
from citeguard.config import Settings
from citeguard.models import (
    CitationRecord, ClaimRecord, CitationFormat, Verdict, Severity
)
from citeguard.cache.manager import CacheManager
from citeguard.retrieval.bm25 import BM25Retriever
from citeguard.models import ChunkRecord


def make_settings() -> Settings:
    return Settings(openrouter_api_key="test-key")


def make_citation(matched_pdf: str | None = "test.pdf") -> CitationRecord:
    return CitationRecord(
        id="ref_001", raw_marker="[1]",
        format=CitationFormat.NUMERIC, position=0,
        matched_pdf=matched_pdf,
    )


def make_claim() -> ClaimRecord:
    return ClaimRecord(
        citation_id="ref_001",
        claim_text="The method achieves 94% accuracy on the benchmark.",
    )


def test_system_prompts_all_three_modes():
    assert "lenient" in _SYSTEM_PROMPTS
    assert "balanced" in _SYSTEM_PROMPTS
    assert "strict" in _SYSTEM_PROMPTS
    for prompt in _SYSTEM_PROMPTS.values():
        assert "mark_verdict" in prompt
        assert "save_checkpoint" in prompt


def test_unverifiable_no_pdf(tmp_path):
    """Citations without a matched PDF should be marked UNVERIFIABLE without LLM call."""
    settings = make_settings()
    agent = VerificationAgent(settings)
    cache = CacheManager(checkpoints_dir=tmp_path / "ckpts", index_cache_dir=tmp_path / "idx")
    citation = make_citation(matched_pdf=None)
    claim = make_claim()

    session = agent.verify(citation, claim, retriever=None, cache=cache, pdf_paths=[])

    assert session.verdict == Verdict.UNVERIFIABLE
    assert session.confidence == 1.0
    # Checkpoint should be written
    loaded = cache.load_checkpoint("ref_001")
    assert loaded is not None
    assert loaded.verdict == Verdict.UNVERIFIABLE


def test_build_prompt_contains_key_fields():
    settings = make_settings()
    agent = VerificationAgent(settings)
    citation = make_citation()
    claim = make_claim()
    prompt = agent._build_prompt(citation, claim)
    assert "[1]" in prompt
    assert "test.pdf" in prompt
    assert "94% accuracy" in prompt


def test_verify_adk_error_fallback(tmp_path):
    """When ADK raises an exception, verify falls back to ERROR verdict and saves checkpoint."""
    settings = make_settings()
    agent = VerificationAgent(settings)
    cache = CacheManager(checkpoints_dir=tmp_path / "ckpts", index_cache_dir=tmp_path / "idx")
    citation = make_citation()
    claim = make_claim()

    retriever = BM25Retriever()
    retriever.index([
        ChunkRecord(pdf_path="test.pdf", chunk_id="c1",
                    text="the method achieves high accuracy on benchmarks", page=1)
    ])

    # Patch LlmAgent to raise an error, simulating an ADK failure
    with patch("google.adk.agents.LlmAgent", side_effect=RuntimeError("ADK unavailable")):
        session = agent.verify(citation, claim, retriever, cache, [])

    # Should fall back to ERROR verdict
    assert session.verdict == Verdict.ERROR
    assert "ADK unavailable" in session.reasoning
    # Checkpoint should be written despite the error
    loaded = cache.load_checkpoint("ref_001")
    assert loaded is not None
    assert loaded.verdict == Verdict.ERROR


def test_verify_agent_no_verdict_fallback(tmp_path):
    """When agent finishes without calling mark_verdict, session gets AMBIGUOUS verdict."""
    settings = make_settings()
    agent = VerificationAgent(settings)
    cache = CacheManager(checkpoints_dir=tmp_path / "ckpts", index_cache_dir=tmp_path / "idx")
    citation = make_citation()
    claim = make_claim()

    # Simulate ADK runner that iterates without calling any tools
    async def fake_run_async(*args, **kwargs):
        return
        yield  # make it an async generator

    mock_runner = MagicMock()
    mock_runner.run_async = fake_run_async

    with patch("google.adk.agents.LlmAgent"), \
         patch("google.adk.runners.Runner", return_value=mock_runner), \
         patch("google.adk.sessions.InMemorySessionService"), \
         patch("google.adk.models.lite_llm.LiteLlm"):
        session = agent.verify(citation, claim, retriever=None, cache=cache, pdf_paths=[])

    # Should fall back to AMBIGUOUS when no verdict was set
    assert session.verdict == Verdict.AMBIGUOUS
    loaded = cache.load_checkpoint("ref_001")
    assert loaded is not None
