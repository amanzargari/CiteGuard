from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    cheap: str = "openai/gpt-4o-mini"
    medium: str = "openai/gpt-4o"
    strong: str = "anthropic/claude-sonnet-4-5"


class RetrievalConfig(BaseModel):
    top_k: int = 5
    top_k_requery: int = 8
    max_requery_rounds: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50


class TokenBudgetConfig(BaseModel):
    per_citation: int = 4000


class RateLimitsConfig(BaseModel):
    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000


class OutputConfig(BaseModel):
    dir: str = "output/"
    checkpoints_dir: str = "output/checkpoints/"
    index_cache_dir: str = "output/index_cache/"


class LocalEmbeddingsConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # .env only
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_api_key: str = ""

    # config.yaml
    strictness: Literal["lenient", "balanced", "strict"] = "balanced"
    retrieval_backend: Literal["bm25", "local_embeddings", "api_embeddings"] = "bm25"
    models: ModelConfig = ModelConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    token_budget: TokenBudgetConfig = TokenBudgetConfig()
    rate_limits: RateLimitsConfig = RateLimitsConfig()
    output: OutputConfig = OutputConfig()
    local_embeddings: LocalEmbeddingsConfig = LocalEmbeddingsConfig()


def load_settings(config_path: Path | None = None) -> Settings:
    yaml_data: dict = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}
    return Settings(**yaml_data)
