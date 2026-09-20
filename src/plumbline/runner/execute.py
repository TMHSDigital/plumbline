"""Run an adapter over a dataset: threads, cache, cost guard, retries, artifact.

Four things this does that a plain loop would not.

It refuses to start a run it cannot afford. ``max_cost_usd`` and ``max_cases``
are checked against an estimate before the first call goes out, and the run
aborts with the arithmetic shown rather than discovering the problem halfway
through someone's budget.

It records failures as failures. A case that errors after its retries is a row
in the artifact with the error text on it, not a missing row. Silently dropping
failures makes an unreliable adapter look accurate.

It separates cache hits from live calls everywhere. A hit measures disk, so it
contributes to neither cost nor latency, and the artifact says how many there
were.

It writes down what actually answered. The requested model, the model the API
said it used, the pinned revision, the dataset hash, the prompt hash per case,
and the full config, with anything that looks like a credential redacted.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plumbline.adapters.base import Adapter
from plumbline.metrics.cost import Pricing, PricingTable, cost_of, estimate_case_cost, pricing_for
from plumbline.runner.cache import Cache, cache_key, prompt_fingerprint, to_jsonable
from plumbline.types import (
    Case,
    CaseRefusedError,
    ConfidenceSeries,
    PlumblineError,
    Prediction,
    ProbabilitySeries,
)

DEFAULT_WORKERS = 8
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 0.5

#: Config keys whose values never reach a results file or a log line.
_SECRET_MARKERS = ("key", "token", "secret", "password", "credential", "authorization")


class CostGuardError(PlumblineError):
    """The run would exceed a limit the caller set, so nothing was sent."""


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff around a transient failure.

    ``sleep`` is injected so tests can prove the backoff schedule without
    spending the wall-clock time it describes.
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")

    def delay_before(self, attempt: int) -> float:
        """Seconds to wait before ``attempt``, which is 1-based."""
        return self.backoff_seconds * (2 ** (attempt - 2)) if attempt > 1 else 0.0


@dataclass(frozen=True)
class CostGuard:
    """Limits checked before the run, not discovered during it."""

    max_cost_usd: float | None = None
    max_cases: int | None = None


@dataclass(frozen=True)
class CaseRecord:
    """One row of the results artifact."""

    case_id: str
    labels: tuple[str, ...]
    gold_label: str
    prompt_hash: str
    prediction: Prediction | None
    cost_usd: float | None
    from_cache: bool
    attempts: int
    error: str | None = None
    refused: bool = False

    @property
    def ok(self) -> bool:
        return self.prediction is not None

    @property
    def correct(self) -> bool | None:
        if self.prediction is None:
            return None
        return self.prediction.label == self.gold_label


@dataclass(frozen=True)
class RunResult:
    """Everything a run produced, and everything needed to reproduce it."""

    adapter_name: str
    probability_semantics: str
    model_requested: str
    model_reported: str | None
    revision: str | None
    timestamp: str
    dataset_hash: str
    pricing_key: str | None
    config: dict[str, Any]
    records: list[CaseRecord]
    cache_stats: dict[str, int] = field(default_factory=dict)

    @property
    def successes(self) -> list[CaseRecord]:
        return [record for record in self.records if record.ok]

    @property
    def failures(self) -> list[CaseRecord]:
        return [record for record in self.records if not record.ok]

    @property
    def live_calls(self) -> list[CaseRecord]:
        """Successful rows that actually went out, so cost and latency mean something."""
        return [record for record in self.records if record.ok and not record.from_cache]

    @property
    def outcomes(self) -> list[bool]:
        return [bool(record.correct) for record in self.successes]

    @property
    def accuracy(self) -> float | None:
        successes = self.successes
        return sum(self.outcomes) / len(successes) if successes else None

    def probabilities(self) -> ProbabilitySeries:
        """The calibratable column, over successful rows, tagged with its semantics."""
        return ProbabilitySeries(
            values=tuple(
                record.prediction.prob_selected
                for record in self.successes
                if record.prediction is not None
            ),
            semantics=self.probability_semantics,  # type: ignore[arg-type]
        )

    def confidences(self) -> ConfidenceSeries:
        return ConfidenceSeries(
            values=tuple(
                record.prediction.confidence
                for record in self.successes
                if record.prediction is not None
            )
        )

    def distributions(self) -> list[dict[str, float] | None]:
        return [
            record.prediction.distribution
            for record in self.successes
            if record.prediction is not None
        ]

    def gold_labels(self) -> list[str]:
        return [record.gold_label for record in self.successes]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "probability_semantics": self.probability_semantics,
            "model_requested": self.model_requested,
            "model_reported": self.model_reported,
            "revision": self.revision,
            "timestamp": self.timestamp,
            "dataset_hash": self.dataset_hash,
            "pricing_key": self.pricing_key,
            "config": self.config,
            "cache_stats": self.cache_stats,
            "records": [
                {
                    "case_id": record.case_id,
                    "labels": list(record.labels),
                    "gold_label": record.gold_label,
                    "prompt_hash": record.prompt_hash,
                    "from_cache": record.from_cache,
                    "attempts": record.attempts,
                    "error": record.error,
                    "refused": record.refused,
                    "cost_usd": record.cost_usd,
                    "prediction": (
                        to_jsonable(record.prediction) if record.prediction is not None else None
                    ),
                }
                for record in self.records
            ],
        }

    def write(self, directory: Path | str = "results") -> Path:
        """Write the artifact. ``results/`` is gitignored; user data stays local."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.timestamp.replace(":", "").replace("-", "")
        path = directory / f"{stamp}-{self.adapter_name}-{self.dataset_hash[:8]}.json"
        path.write_text(
            json.dumps(self.to_jsonable(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path


def dataset_hash(cases: Sequence[Case]) -> str:
    """Fingerprint the dataset, so a result can be tied to the rows that produced it."""
    encoded = json.dumps(
        [
            {
                "id": case.id,
                "text": case.text,
                "labels": sorted(case.labels),
                "gold": case.gold_label,
            }
            for case in cases
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(encoded.encode("utf-8"), digest_size=16).hexdigest()


def redact(config: Mapping[str, Any]) -> dict[str, Any]:
    """Strip anything that looks like a credential before it reaches disk.

    An API key must never appear in a results file. Matching on the key name is
    crude, and deliberately so: it errs toward redacting a harmless field rather
    than letting a secret through.
    """
    cleaned: dict[str, Any] = {}
    for key, value in config.items():
        if any(marker in key.lower() for marker in _SECRET_MARKERS):
            cleaned[key] = "[redacted]"
        elif isinstance(value, Mapping):
            cleaned[key] = redact(value)
        else:
            cleaned[key] = value
    return cleaned


def estimate_run_cost(
    cases: Sequence[Case],
    pricing: Pricing | None,
) -> float | None:
    """Rough total for the whole run, or None when the model is not in the table."""
    estimates = [estimate_case_cost(case.text, case.labels, pricing) for case in cases]
    return None if any(estimate is None for estimate in estimates) else sum(estimates)  # type: ignore[arg-type]


def check_guard(
    cases: Sequence[Case],
    guard: CostGuard,
    pricing: Pricing | None,
) -> float | None:
    """Abort before sending anything if the run would break a limit."""
    if guard.max_cases is not None and len(cases) > guard.max_cases:
        raise CostGuardError(
            f"this run covers {len(cases)} cases and max_cases is {guard.max_cases}. "
            "Nothing was sent. Raise the limit or pass fewer cases; plumbline will not "
            "silently truncate a dataset, because a partial run reported as a whole one "
            "is worse than no run."
        )

    estimate = estimate_run_cost(cases, pricing)
    if guard.max_cost_usd is None:
        return estimate

    if estimate is None:
        raise CostGuardError(
            f"max_cost_usd is set to {guard.max_cost_usd} but this model is not in the "
            "pricing table, so the run cannot be costed in advance. Nothing was sent. "
            "Add the model to the pricing table, or clear max_cost_usd to run without a "
            "spend limit."
        )
    if estimate > guard.max_cost_usd:
        raise CostGuardError(
            f"estimated cost {estimate:.4f} USD over {len(cases)} cases exceeds "
            f"max_cost_usd of {guard.max_cost_usd:.4f}. Nothing was sent. The estimate is "
            "rough and counts the case text, the option names, and a fixed overhead."
        )
    return estimate


def run(
    adapter: Adapter,
    cases: Sequence[Case],
    *,
    cache: Cache | None = None,
    guard: CostGuard | None = None,
    pricing_table: PricingTable | None = None,
    workers: int = DEFAULT_WORKERS,
    retry: RetryPolicy | None = None,
    extra_config: Mapping[str, Any] | None = None,
) -> RunResult:
    """Classify every case, in order, with the guard checked before anything is sent."""
    if not cases:
        raise ValueError("no cases to run")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")

    retry = retry or RetryPolicy()
    guard = guard or CostGuard()
    table: PricingTable = pricing_table or {}
    pricing, pricing_key = pricing_for(table, None, adapter.model_requested)
    estimate = check_guard(cases, guard, pricing)

    records: list[CaseRecord | None] = [None] * len(cases)

    def handle(index: int) -> None:
        records[index] = _one_case(cases[index], adapter, cache, retry, table, pricing_key)

    if workers == 1:
        for index in range(len(cases)):
            handle(index)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(handle, range(len(cases))))

    finished = [record for record in records if record is not None]
    reported = next(
        (
            record.prediction.model_reported
            for record in finished
            if record.prediction is not None and record.prediction.model_reported is not None
        ),
        None,
    )

    config: dict[str, Any] = {
        "workers": workers,
        "max_attempts": retry.max_attempts,
        "backoff_seconds": retry.backoff_seconds,
        "max_cost_usd": guard.max_cost_usd,
        "max_cases": guard.max_cases,
        "estimated_cost_usd": estimate,
        "cache_enabled": cache is not None and cache.enabled,
        "pricing_table": {
            name: {
                "input_usd_per_million": price.input_usd_per_million,
                "output_usd_per_million": price.output_usd_per_million,
            }
            for name, price in table.items()
        },
        **dict(extra_config or {}),
    }

    return RunResult(
        adapter_name=adapter.name,
        probability_semantics=adapter.probability_semantics,
        model_requested=adapter.model_requested,
        model_reported=reported,
        revision=adapter.revision,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        dataset_hash=dataset_hash(cases),
        pricing_key=pricing_key,
        config=redact(config),
        records=finished,
        cache_stats=cache.stats if cache is not None else {},
    )


def _one_case(
    case: Case,
    adapter: Adapter,
    cache: Cache | None,
    retry: RetryPolicy,
    table: PricingTable,
    pricing_key: str | None,
) -> CaseRecord:
    labels = list(case.labels)
    key = cache_key(adapter, case.text, labels) if cache is not None else None

    if cache is not None and key is not None:
        hit = cache.get(key)
        if hit is not None:
            return _record(case, hit, from_cache=True, attempts=0, table=table)

    last_error: Exception | None = None
    for attempt in range(1, retry.max_attempts + 1):
        delay = retry.delay_before(attempt)
        if delay:
            retry.sleep(delay)
        try:
            prediction = adapter.classify(case.text, labels)
        except CaseRefusedError as refusal:
            # A refusal is a decision, not a transient fault. Retrying it would
            # only produce the same refusal more slowly.
            return CaseRecord(
                case_id=case.id,
                labels=case.labels,
                gold_label=case.gold_label,
                prompt_hash=prompt_fingerprint(case.text, labels),
                prediction=None,
                cost_usd=None,
                from_cache=False,
                attempts=attempt,
                error=str(refusal),
                refused=True,
            )
        except Exception as error:  # an adapter may raise anything; record it as a failure
            last_error = error
            continue

        if cache is not None and key is not None:
            cache.put(key, prediction)
        return _record(case, prediction, from_cache=False, attempts=attempt, table=table)

    return CaseRecord(
        case_id=case.id,
        labels=case.labels,
        gold_label=case.gold_label,
        prompt_hash=prompt_fingerprint(case.text, labels),
        prediction=None,
        cost_usd=None,
        from_cache=False,
        attempts=retry.max_attempts,
        error=f"{type(last_error).__name__}: {last_error}",
    )


def _record(
    case: Case,
    prediction: Prediction,
    *,
    from_cache: bool,
    attempts: int,
    table: PricingTable,
) -> CaseRecord:
    if from_cache:
        # A hit measures disk, not the model. Charging for it would make a
        # re-run look cheaper than the run it repeats.
        cost = None
    elif prediction.cost_usd is not None:
        cost = prediction.cost_usd
    else:
        pricing, _ = pricing_for(table, prediction.model_reported, "")
        cost = cost_of(prediction.input_tokens, prediction.output_tokens, pricing)

    return CaseRecord(
        case_id=case.id,
        labels=case.labels,
        gold_label=case.gold_label,
        prompt_hash=str(prediction.raw.get("prompt_hash"))
        if prediction.raw.get("prompt_hash")
        else prompt_fingerprint(case.text, list(case.labels)),
        prediction=prediction,
        cost_usd=cost,
        from_cache=from_cache,
        attempts=attempts,
    )
