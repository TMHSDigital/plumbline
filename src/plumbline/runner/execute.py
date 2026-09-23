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
from plumbline.metrics.cost import (
    CostBasis,
    Pricing,
    PricingTable,
    cost_of,
    estimate_case_cost,
    pricing_for,
)
from plumbline.runner.cache import (
    Cache,
    cache_key,
    prompt_fingerprint,
    to_jsonable,
    to_prediction,
)
from plumbline.types import (
    ArtifactError,
    Case,
    CaseRefusedError,
    ConfidenceSeries,
    PlumblineError,
    Prediction,
    ProbabilitySeries,
    QuestionType,
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


#: HTTP statuses a second attempt can fix: the request timed out, conflicted,
#: came too early, or was rate limited. Any 5xx joins them. Every other status
#: is the service saying no, and asking again only pays for the same answer.
_RETRYABLE_STATUSES = frozenset({408, 409, 425, 429})

#: The longest a Retry-After header is allowed to hold a case, in seconds.
MAX_RETRY_AFTER_SECONDS = 60.0


def _status_of(error: BaseException) -> int | None:
    """The HTTP status an SDK error carries, whichever attribute it uses."""
    for attribute in ("status_code", "status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def is_transient(error: BaseException) -> bool:
    """Whether a second attempt could succeed where this one failed.

    Only transport failures are: a connection that dropped, a timeout, a rate
    limit, a server error. Everything plumbline raises on purpose is permanent
    (the service answered a different question, the checkpoint was not the one
    pinned, an optional dependency is missing), as is a 4xx and any error the
    runner does not recognise, because every retry is another billed call. The
    checks are by shape rather than by SDK class, so the runner imports no SDK.
    """
    if isinstance(error, PlumblineError):
        return False
    status = _status_of(error)
    if status is not None:
        return status in _RETRYABLE_STATUSES or 500 <= status < 600
    if isinstance(error, TimeoutError | ConnectionError):
        return True
    # SDKs that do not subclass the builtins still name their transport errors
    # this way (Anthropic's APIConnectionError and APITimeoutError).
    return any(
        cls.__name__.endswith(("ConnectionError", "TimeoutError")) for cls in type(error).__mro__
    )


def retry_after(error: BaseException) -> float | None:
    """Seconds a Retry-After header on the error asks for, capped, or None."""
    headers = getattr(error, "headers", None)
    if headers is None:
        headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None  # an HTTP-date, or garbage: fall back to the backoff schedule
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)


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
    question_type: QuestionType = "choice"
    asked_as: str = "choice"
    """How the adapter actually asked it, which is not always what the row is.

    A yes/no row answered as a two-option choice and one answered as a noul are
    different measurements. Both are recorded so the report can keep them apart
    rather than averaging across a difference nobody can see.
    """
    cost_basis: CostBasis = "no_prediction"
    pricing_key: str | None = None

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
    dataset_rows: int
    pricing_key: str | None
    config: dict[str, Any]
    records: list[CaseRecord]
    cache_stats: dict[str, int] = field(default_factory=dict)
    pricing: dict[str, Any] | None = None
    """Provenance of the pricing entry that was applied, or None when none was.

    Carries the source and the date that entry was read, so a result opened in a
    year is not silently re-scored against the prices of the day it is opened.
    """

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
            "dataset_rows": self.dataset_rows,
            "pricing_key": self.pricing_key,
            "pricing": self.pricing,
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
                    "question_type": record.question_type,
                    "asked_as": record.asked_as,
                    "cost_usd": record.cost_usd,
                    "cost_basis": record.cost_basis,
                    "pricing_key": record.pricing_key,
                    "prediction": (
                        to_jsonable(record.prediction) if record.prediction is not None else None
                    ),
                }
                for record in self.records
            ],
        }

    @classmethod
    def read(cls, path: Path | str) -> RunResult:
        """Rebuild a finished run from the artifact it wrote.

        A report months after the fact should not require paying for the run
        again, and a result re-read this way carries the pricing entry and date
        it was scored against rather than today's.

        A path that is not there, or a file that is not an artifact, is refused
        with a sentence rather than a traceback. Naming the wrong file is an
        ordinary mistake, and a stack trace is the right output for a bug and
        the wrong output for a typo.
        """
        path = Path(path)
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ArtifactError(
                f"no artifact at {path}. plumbline never guesses a results location, "
                "because a relative default resolves against whatever directory the "
                "process started in."
            ) from None
        except OSError as unreadable:
            raise ArtifactError(f"{path} could not be read: {unreadable.strerror}") from unreadable
        except json.JSONDecodeError as broken:
            raise ArtifactError(
                f"{path} is not valid JSON: {broken.msg} (line {broken.lineno}). An "
                "artifact is written by plumbline, so this is usually the wrong file "
                "rather than a damaged one."
            ) from broken

        if not isinstance(stored, dict):
            raise ArtifactError(f"{path} holds a {type(stored).__name__}, not an artifact object.")
        try:
            return cls._from_stored(stored, path)
        except (KeyError, TypeError, AttributeError) as wrong_shape:
            raise ArtifactError(
                f"{path} is valid JSON but not a plumbline artifact: {wrong_shape!r}. "
                "Artifacts are the files `plumbline run` writes into its results "
                "directory."
            ) from wrong_shape

    @classmethod
    def _from_stored(cls, stored: dict[str, Any], path: Path) -> RunResult:
        return cls(
            adapter_name=stored["adapter_name"],
            probability_semantics=stored["probability_semantics"],
            model_requested=stored["model_requested"],
            model_reported=stored.get("model_reported"),
            revision=stored.get("revision"),
            timestamp=stored["timestamp"],
            dataset_hash=stored["dataset_hash"],
            dataset_rows=stored.get("dataset_rows", len(stored["records"])),
            pricing_key=stored.get("pricing_key"),
            config=stored.get("config", {}),
            records=[_record_from_jsonable(row) for row in stored["records"]],
            cache_stats=stored.get("cache_stats", {}),
            pricing=stored.get("pricing"),
        )

    def write(self, directory: Path | str) -> Path:
        """Write the artifact to ``directory``, which the caller must name.

        There is deliberately no default. The artifact carries the user's
        per-case records, and a relative default resolves against the working
        directory, so where it lands depends on where the process happened to
        start. Safety that depends on cwd is not safety. ``Cache`` requires its
        directory for the same reason.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.timestamp.replace(":", "").replace("-", "")
        base = f"{stamp}-{self.adapter_name}-{self.dataset_hash[:8]}"
        # The stamp is only accurate to the second, so two runs of the same
        # adapter over the same dataset can land on one name. An artifact holds
        # the user's per-case records; a second run quietly replacing the first
        # would destroy a result nobody asked to delete.
        path = directory / f"{base}.json"
        suffix = 2
        while path.exists():
            path = directory / f"{base}-{suffix}.json"
            suffix += 1
        path.write_text(
            json.dumps(self.to_jsonable(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path


def _record_from_jsonable(row: dict[str, Any]) -> CaseRecord:
    """One stored row back into a record, with nothing invented for a gap."""
    prediction = row.get("prediction")
    return CaseRecord(
        case_id=row["case_id"],
        labels=tuple(row["labels"]),
        gold_label=row["gold_label"],
        prompt_hash=row["prompt_hash"],
        prediction=to_prediction(prediction) if prediction is not None else None,
        cost_usd=row.get("cost_usd"),
        from_cache=bool(row.get("from_cache", False)),
        attempts=int(row.get("attempts", 0)),
        error=row.get("error"),
        refused=bool(row.get("refused", False)),
        question_type=row.get("question_type", "choice"),
        asked_as=row.get("asked_as", "choice"),
        cost_basis=row.get("cost_basis", "no_prediction"),
        pricing_key=row.get("pricing_key"),
    )


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
    today = datetime.now(UTC).date()
    pricing, pricing_key = pricing_for(table, None, adapter.model_requested)
    estimate = check_guard(cases, guard, pricing)

    records: list[CaseRecord | None] = [None] * len(cases)

    def handle(index: int) -> None:
        records[index] = _one_case(cases[index], adapter, cache, retry, table)

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

    # Which entry actually priced the rows, which can differ from the one the
    # guard used: the guard runs before anything answers and can only look up
    # the requested model, while billing follows what the API said it used.
    applied_key = next(
        (record.pricing_key for record in finished if record.pricing_key is not None),
        pricing_key,
    )
    applied = table.get(applied_key) if applied_key is not None else None

    config: dict[str, Any] = {
        "workers": workers,
        "max_attempts": retry.max_attempts,
        "backoff_seconds": retry.backoff_seconds,
        "max_cost_usd": guard.max_cost_usd,
        "max_cases": guard.max_cases,
        "estimated_cost_usd": estimate,
        "cache_enabled": cache is not None and cache.enabled,
        "pricing_table": {name: price.provenance(today) for name, price in table.items()},
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
        # The hash says which rows; the count is what every calibration figure
        # has to be read against, so both travel with the result.
        dataset_rows=len(cases),
        pricing_key=applied_key,
        pricing=applied.provenance(today) if applied is not None else None,
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
) -> CaseRecord:
    labels = list(case.labels)
    key = cache_key(adapter, case.text, labels, case.question_type) if cache is not None else None

    if cache is not None and key is not None:
        hit = cache.get(key)
        if hit is not None:
            return _record(
                case,
                hit,
                from_cache=True,
                attempts=0,
                table=table,
                reports_tokens=adapter.reports_tokens,
            )

    last_error: Exception | None = None
    attempts = 0
    for attempt in range(1, retry.max_attempts + 1):
        delay = retry.delay_before(attempt)
        if last_error is not None:
            delay = max(delay, retry_after(last_error) or 0.0)
        if delay:
            retry.sleep(delay)
        attempts = attempt
        try:
            prediction = adapter.classify(case.text, labels, question_type=case.question_type)
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
                question_type=case.question_type,
                asked_as="refused",
            )
        except Exception as error:  # an adapter may raise anything; record it as a failure
            last_error = error
            if is_transient(error):
                continue
            # No retry fixes it, and each one would be another billed call.
            break

        if cache is not None and key is not None:
            cache.put(key, prediction)
        return _record(
            case,
            prediction,
            from_cache=False,
            attempts=attempt,
            table=table,
            reports_tokens=adapter.reports_tokens,
        )

    return CaseRecord(
        case_id=case.id,
        labels=case.labels,
        gold_label=case.gold_label,
        prompt_hash=prompt_fingerprint(case.text, labels),
        prediction=None,
        cost_usd=None,
        from_cache=False,
        attempts=attempts,
        error=f"{type(last_error).__name__}: {last_error}",
        question_type=case.question_type,
        asked_as="failed",
    )


def _record(
    case: Case,
    prediction: Prediction,
    *,
    from_cache: bool,
    attempts: int,
    table: PricingTable,
    reports_tokens: bool,
) -> CaseRecord:
    cost, basis, key = _cost_of_record(
        prediction, from_cache=from_cache, table=table, reports_tokens=reports_tokens
    )

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
        question_type=case.question_type,
        asked_as=str(prediction.raw.get("asked_as", "choice")),
        cost_basis=basis,
        pricing_key=key,
    )


def _cost_of_record(
    prediction: Prediction,
    *,
    from_cache: bool,
    table: PricingTable,
    reports_tokens: bool,
) -> tuple[float | None, CostBasis, str | None]:
    """The cost of one row, and the reason it is what it is.

    Every path that yields no cost names itself. A reader looking at a blank
    cost column can then tell a local model, which never reports tokens, from an
    API that reported none on this call, from a model that answered but is not
    in the pricing table. Those are three different things to go and fix.
    """
    if from_cache:
        # A hit measures disk, not the model. Charging for it would make a
        # re-run look cheaper than the run it repeats.
        return None, "cache_hit", None
    if prediction.cost_usd is not None:
        # An adapter that was handed a cost by the vendor. Nothing to derive.
        return prediction.cost_usd, "priced", None

    pricing, key = pricing_for(table, prediction.model_reported, "")
    cost = cost_of(prediction.input_tokens, prediction.output_tokens, pricing)
    if cost is not None:
        return cost, "priced", key
    if not reports_tokens:
        return None, "adapter_reports_no_tokens", key
    if prediction.input_tokens is None or prediction.output_tokens is None:
        return None, "tokens_not_reported", key
    return None, "model_not_priced", key
