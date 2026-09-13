"""Run a provider over a set of Quorum items and collect Predictions.

Three things matter here beyond the obvious loop.

CACHING. Every call costs money, and a run that has to be repeated because the
scorecard code changed should not cost it twice. Responses are cached on disk
under a key derived from the provider, the model id, the prompt style and the
prompt text itself, so changing the prompt correctly invalidates the cache
while re-running an unchanged experiment is free.

FAILURES ARE RECORDED, NOT HIDDEN. A call that errors after its retries, or a
response the parser cannot read, becomes an abstention and is counted in the
run statistics. Quorum never scores abstention as error, so a broken run would
otherwise look like a thoughtful one.

NOTHING SENSITIVE IS WRITTEN. The cache holds prompts and completions. API keys
live only in the environment of the process.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from quorum.llm.prompt import SYSTEM, PromptStyle, build_prompt, parse_response
from quorum.llm.providers import Provider, ProviderError
from quorum.schema import Item, Prediction


@dataclass
class RunStats:
    """What actually happened during a run, reported alongside the scorecard."""

    n_items: int = 0
    from_cache: int = 0
    called: int = 0
    errors: int = 0
    unparseable: int = 0
    abstained: int = 0
    seconds: float = 0.0
    error_examples: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "RUN STATISTICS",
            f"  items                 : {self.n_items:,}",
            f"  served from cache     : {self.from_cache:,}",
            f"  API calls made        : {self.called:,}",
            f"  call failures         : {self.errors:,}",
            f"  unparseable responses : {self.unparseable:,}",
            f"  model abstentions     : {self.abstained:,}",
            f"  wall clock            : {self.seconds:,.1f}s",
        ]
        for e in self.error_examples[:3]:
            lines.append(f"  ! {e}")
        if self.errors or self.unparseable:
            lines.append("  NOTE: failures and unparseable responses were recorded as")
            lines.append("  abstentions. Quorum never scores an abstention as an error, so")
            lines.append("  check these counts before reading the abstain rate as behaviour.")
        return "\n".join(lines)


def _cache_key(provider: Provider, style: PromptStyle, prompt: str) -> str:
    blob = f"{provider.name}|{provider.model}|{style.value}|{prompt}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def run_model(
    items: list[Item],
    provider: Provider,
    *,
    style: PromptStyle = PromptStyle.BLIND,
    cache_dir: str | Path | None = "llm_cache",
    limit: int | None = None,
    sleep_s: float = 0.0,
    progress: bool = True,
) -> tuple[list[Prediction], RunStats]:
    """Query `provider` for every item and return (predictions, stats).

    Args:
        items: Quorum items to score.
        provider: Any Provider; FakeProvider runs offline.
        style: BLIND withholds the country name, NAMED supplies it. See
            quorum.llm.prompt for why that choice is the measurement.
        cache_dir: Where to cache raw completions. None disables caching.
        limit: Score only the first N items. Use it to price a run before
            committing to it.
        sleep_s: Pause between live calls, for rate limits.
        progress: Print a one-line counter to stderr.
    """
    work = items[:limit] if limit else items
    stats = RunStats(n_items=len(work))
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    predictions: list[Prediction] = []
    started = time.time()

    for i, item in enumerate(work, 1):
        prompt = build_prompt(item, style)
        text: str | None = None
        path = cache / f"{_cache_key(provider, style, prompt)}.json" if cache else None

        if path is not None and path.exists():
            try:
                text = json.loads(path.read_text(encoding="utf-8"))["completion"]
                stats.from_cache += 1
            except (json.JSONDecodeError, KeyError, OSError):
                text = None

        if text is None:
            try:
                text = provider.complete(SYSTEM, prompt)
                stats.called += 1
                if path is not None:
                    path.write_text(
                        json.dumps({"model": provider.model, "style": style.value,
                                    "prompt": prompt, "completion": text}, indent=1),
                        encoding="utf-8",
                    )
            except ProviderError as exc:
                stats.errors += 1
                if len(stats.error_examples) < 3:
                    stats.error_examples.append(f"{item.item_id}: {exc}")
                text = ""
            if sleep_s:
                time.sleep(sleep_s)

        pred = parse_response(item.item_id, text)
        if pred.label is None and pred.confidence is None and text.strip():
            stats.unparseable += 1
        if pred.label is None:
            stats.abstained += 1
        predictions.append(pred)

        if progress and (i % 10 == 0 or i == len(work)):
            print(f"\r  {i}/{len(work)} items", end="", file=sys.stderr, flush=True)

    if progress:
        print("", file=sys.stderr)
    stats.seconds = time.time() - started
    return predictions, stats
