"""The LLM harness: score a language model on Quorum items.

Quorum's premise is that the hard financial questions have contested expert
answers. Until a language model has actually been put through the harness, that
premise is a design argument rather than a measurement. This package is what
turns it into one.

Public API:
    Provider, AnthropicProvider, OpenAIProvider, GeminiProvider, FakeProvider
    PromptStyle, build_prompt
    run_model    - items in, Predictions out, with disk caching
"""

from quorum.llm.providers import (
    AnthropicProvider,
    FakeProvider,
    GeminiProvider,
    OpenAIProvider,
    Provider,
    get_provider,
)
from quorum.llm.prompt import PromptStyle, build_prompt, parse_response
from quorum.llm.runner import RunStats, run_model

__all__ = [
    "Provider", "AnthropicProvider", "OpenAIProvider", "GeminiProvider",
    "FakeProvider", "get_provider",
    "PromptStyle", "build_prompt", "parse_response",
    "run_model", "RunStats",
]
