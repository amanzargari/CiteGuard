from __future__ import annotations
import time
import litellm
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)
from citeguard.config import Settings
from citeguard.models import TokenUsage

litellm.suppress_debug_info = True


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._usage = TokenUsage()
        self._request_timestamps: list[float] = []

    def _enforce_rate_limit(self) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        self._request_timestamps = [t for t in self._request_timestamps if t > cutoff]
        limit = self._settings.rate_limits.requests_per_minute
        if len(self._request_timestamps) >= limit:
            sleep_for = 60.0 - (now - self._request_timestamps[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=32),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.1,
    ) -> str:
        self._enforce_rate_limit()
        self._request_timestamps.append(time.monotonic())

        response = litellm.completion(
            model=f"openrouter/{model}",
            messages=messages,
            temperature=temperature,
            api_base=self._settings.openrouter_base_url,
            api_key=self._settings.openrouter_api_key,
        )

        usage = response.usage
        self._usage = TokenUsage(
            prompt=self._usage.prompt + usage.prompt_tokens,
            completion=self._usage.completion + usage.completion_tokens,
            total=self._usage.total + usage.total_tokens,
        )
        return response.choices[0].message.content

    @property
    def total_usage(self) -> TokenUsage:
        return self._usage.model_copy()
