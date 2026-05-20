import pytest
from pathlib import Path
import yaml
from citeguard.config import load_settings, Settings


def test_default_settings():
    s = Settings()
    assert s.strictness == "balanced"
    assert s.retrieval_backend == "bm25"
    assert s.retrieval.top_k == 5
    assert s.models.cheap == "openai/gpt-4o-mini"
    assert s.models.strong == "anthropic/claude-sonnet-4-5"


def test_load_from_yaml(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"strictness": "strict", "retrieval_backend": "local_embeddings"}))
    s = load_settings(cfg)
    assert s.strictness == "strict"
    assert s.retrieval_backend == "local_embeddings"
    assert s.retrieval.top_k == 5  # default preserved


def test_invalid_strictness_raises(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"strictness": "extreme"}))
    with pytest.raises(Exception):
        load_settings(cfg)


def test_load_nonexistent_config_uses_defaults():
    s = load_settings(Path("nonexistent.yaml"))
    assert s.strictness == "balanced"


def test_output_config_defaults():
    s = Settings()
    assert s.output.dir == "output/"
    assert s.output.checkpoints_dir == "output/checkpoints/"
    assert s.output.index_cache_dir == "output/index_cache/"


def test_rate_limits_defaults():
    s = Settings()
    assert s.rate_limits.requests_per_minute == 60
    assert s.rate_limits.tokens_per_minute == 100_000


def test_token_budget_default():
    s = Settings()
    assert s.token_budget.per_citation == 4000
