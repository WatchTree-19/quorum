"""Minimal REST clients for the frontier model providers.

Deliberately written against the raw HTTP APIs using the standard library
alone, rather than against three vendor SDKs. The harness should run on a
machine with numpy and scikit-learn and nothing else, because every extra
dependency is one more reason a result cannot be reproduced.

API KEYS ARE READ FROM THE ENVIRONMENT AND NEVER FROM A FILE IN THE REPOSITORY,
never logged, and never written into the cache. Set them in the shell that runs
the harness:

    setx ANTHROPIC_API_KEY  ...        (Windows, new shell after)
    export ANTHROPIC_API_KEY=...       (macOS, Linux)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

TIMEOUT_S = 90
MAX_RETRIES = 5


class ProviderError(RuntimeError):
    pass


@dataclass
class Provider:
    """A named model behind a callable that turns (system, prompt) into text."""

    name: str
    model: str

    def complete(self, system: str, prompt: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    # Shared HTTP with retry on rate limits and transient server errors.
    def _post(self, url: str, payload: dict, headers: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                if exc.code in (429, 500, 502, 503, 529):
                    last = ProviderError(f"{self.name} HTTP {exc.code}: {detail}")
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise ProviderError(f"{self.name} HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                last = ProviderError(f"{self.name} network error: {exc}")
                time.sleep(min(2 ** attempt, 30))
        raise ProviderError(f"{self.name}: giving up after {MAX_RETRIES} attempts ({last})")


def _key(var: str, provider: str) -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        raise ProviderError(
            f"{provider}: environment variable {var} is not set. "
            f"The harness never reads keys from disk; set it in your shell and re-run."
        )
    return value


@dataclass
class AnthropicProvider(Provider):
    name: str = "anthropic"
    model: str = "claude-sonnet-4-5"
    max_tokens: int = 300

    def complete(self, system: str, prompt: str) -> str:
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            {
                "content-type": "application/json",
                "x-api-key": _key("ANTHROPIC_API_KEY", "anthropic"),
                "anthropic-version": "2023-06-01",
            },
        )
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        return "".join(parts)


@dataclass
class OpenAIProvider(Provider):
    name: str = "openai"
    model: str = "gpt-4.1"
    max_tokens: int = 300

    def complete(self, system: str, prompt: str) -> str:
        data = self._post(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": self.model,
                "max_completion_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            {
                "content-type": "application/json",
                "authorization": "Bearer " + _key("OPENAI_API_KEY", "openai"),
            },
        )
        choices = data.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""


@dataclass
class GeminiProvider(Provider):
    name: str = "gemini"
    model: str = "gemini-2.5-flash"
    max_tokens: int = 300

    def complete(self, system: str, prompt: str) -> str:
        key = _key("GEMINI_API_KEY", "gemini")
        data = self._post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": self.max_tokens},
            },
            {"content-type": "application/json", "x-goog-api-key": key},
        )
        cands = data.get("candidates") or [{}]
        parts = (cands[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts)


@dataclass
class FakeProvider(Provider):
    """Deterministic offline provider, for tests and for --dry-run.

    Answers from the fundamentals with a crude rule so that the harness can be
    exercised end to end, and every scorecard path taken, without a network or
    a key. It is not a baseline and is not reported as one.
    """

    name: str = "fake"
    model: str = "rule"
    calls: int = 0

    def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        gdp = 0.0
        for line in prompt.splitlines():
            if "GDP per capita" in line and "not reported" not in line:
                digits = line.split(":", 1)[1].replace(",", "").replace("USD", "").strip()
                try:
                    gdp = float(digits)
                except ValueError:
                    gdp = 0.0
        if 8_000 <= gdp <= 16_000:
            return '{"answer": "abstain", "probability_speculative": 0.5, "reason": "boundary"}'
        spec = gdp < 12_000
        p = 0.85 if spec else 0.1
        answer = "speculative" if spec else "investment_grade"
        return f'{{"answer": "{answer}", "probability_speculative": {p}, "reason": "rule"}}'


_REGISTRY = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "fake": FakeProvider,
}


def get_provider(name: str, model: str | None = None) -> Provider:
    """Build a provider by name, optionally overriding the default model id."""
    try:
        cls = _REGISTRY[name.lower()]
    except KeyError:
        raise ProviderError(
            f"Unknown provider {name!r}. Known: {', '.join(sorted(_REGISTRY))}."
        ) from None
    return cls(model=model) if model else cls()
