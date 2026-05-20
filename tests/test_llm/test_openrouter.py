import pytest
from unittest.mock import patch, MagicMock
from citeguard.llm.openrouter import OpenRouterClient
from citeguard.config import Settings
from citeguard.models import TokenUsage


def make_client():
    s = Settings(openrouter_api_key="test-key",
                 openrouter_base_url="https://openrouter.ai/api/v1")
    return OpenRouterClient(s)


def test_total_usage_starts_zero():
    c = make_client()
    assert c.total_usage.total == 0


def test_complete_tracks_token_usage():
    c = make_client()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "answer"
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150

    with patch("citeguard.llm.openrouter.litellm.completion", return_value=mock_response):
        result = c.complete("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}])

    assert result == "answer"
    assert c.total_usage.prompt == 100
    assert c.total_usage.total == 150


def test_complete_accumulates_across_calls():
    c = make_client()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 25
    mock_response.usage.total_tokens = 75

    with patch("citeguard.llm.openrouter.litellm.completion", return_value=mock_response):
        c.complete("openai/gpt-4o-mini", [{"role": "user", "content": "a"}])
        c.complete("openai/gpt-4o-mini", [{"role": "user", "content": "b"}])

    assert c.total_usage.total == 150


def test_total_usage_returns_copy():
    c = make_client()
    u1 = c.total_usage
    u2 = c.total_usage
    assert u1 is not u2
