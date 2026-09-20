"""The runner: cache, cost guard, retries, failure recording, artifact.

Tested against the mock only. Nothing here goes near a network.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from plumbline.adapters.base import Adapter
from plumbline.metrics.cost import Pricing
from plumbline.runner import execute
from plumbline.runner.cache import Cache, cache_key
from plumbline.runner.execute import CostGuard, CostGuardError, RetryPolicy
from plumbline.types import Case, CaseRefusedError, Prediction
from tests.helpers import gold_by_text, make_cases

LABELS = ("billing", "returns", "shipping", "other")
#: A fixture priced as of today, so "not stale" stays true as the clock moves.
#: Staleness itself is tested against an explicitly old entry below.
READ_ON = date.today()

PRICING = {
    "mock-1": Pricing(
        input_usd_per_million=1.0,
        output_usd_per_million=5.0,
        source="test fixture",
        as_of=READ_ON,
    )
}


def an_adapter(cases, **overrides):
    from plumbline.adapters.mock import MockAdapter

    config: dict = {"seed": 7, "accuracy": 0.8}
    config.update(overrides)
    return MockAdapter(gold_by_text(cases), **config)


class FlakyAdapter(Adapter):
    """Fails a fixed number of times per case, then answers. Counts its calls."""

    def __init__(self, inner: Adapter, failures_before_success: int) -> None:
        self.inner = inner
        self.name = "flaky"
        self.model_requested = inner.model_requested
        self.revision = None
        self.probability_semantics = inner.probability_semantics
        self.failures_before_success = failures_before_success
        self.calls: dict[str, int] = {}

    def classify(self, text: str, labels: list[str], **asked: object) -> Prediction:
        self.calls[text] = self.calls.get(text, 0) + 1
        if self.calls[text] <= self.failures_before_success:
            raise TimeoutError("upstream timed out")
        return self.inner.classify(text, labels, **asked)  # type: ignore[arg-type]


class RefusingAdapter(Adapter):
    """Declines every case, the way a local model does when an option is multi-token."""

    def __init__(self) -> None:
        self.name = "refuser"
        self.model_requested = "refuser-1"
        self.revision = "abc123"
        self.probability_semantics = "restricted_softmax"
        self.calls = 0

    def classify(self, text: str, labels: list[str], **asked: object) -> Prediction:
        self.calls += 1
        raise CaseRefusedError("option 'shipping' does not map to a single token")


# Basic execution


def test_a_run_returns_one_record_per_case_in_input_order() -> None:
    cases = make_cases(200, labels=LABELS)
    result = execute.run(an_adapter(cases), cases, workers=4)

    assert len(result.records) == 200
    assert [record.case_id for record in result.records] == [case.id for case in cases]


def test_threading_does_not_change_any_answer() -> None:
    """Seeds are per case, so worker count is an execution detail and nothing more."""
    cases = make_cases(300, labels=LABELS)
    serial = execute.run(an_adapter(cases), cases, workers=1)
    parallel = execute.run(an_adapter(cases), cases, workers=8)

    assert [record.prediction for record in serial.records] == [
        record.prediction for record in parallel.records
    ]


def test_the_run_reports_accuracy_over_successful_rows() -> None:
    cases = make_cases(1000, labels=LABELS)
    result = execute.run(an_adapter(cases, accuracy=0.8), cases, workers=4)
    assert result.accuracy == pytest.approx(0.8, abs=0.05)


def test_an_empty_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="no cases"):
        execute.run(an_adapter(make_cases(2)), [])


# The cache


def test_a_second_run_is_served_entirely_from_cache(tmp_path: Path) -> None:
    cases = make_cases(100, labels=LABELS)
    cache = Cache(tmp_path / "cache")

    first = execute.run(an_adapter(cases), cases, cache=cache, workers=4)
    assert cache.stats["writes"] == 100
    assert all(not record.from_cache for record in first.records)

    second = execute.run(an_adapter(cases), cases, cache=Cache(tmp_path / "cache"), workers=4)
    assert all(record.from_cache for record in second.records)
    assert second.cache_stats["hits"] == 100


def test_cached_answers_are_identical_to_the_originals(tmp_path: Path) -> None:
    cases = make_cases(60, labels=LABELS)
    cache = Cache(tmp_path / "cache")
    first = execute.run(an_adapter(cases), cases, cache=cache, workers=2)
    second = execute.run(an_adapter(cases), cases, cache=Cache(tmp_path / "cache"), workers=2)

    for before, after in zip(first.records, second.records, strict=True):
        assert before.prediction == after.prediction


def test_a_cache_hit_costs_nothing_and_is_excluded_from_live_calls(tmp_path: Path) -> None:
    """A hit measures disk. Charging for it would make a re-run look cheaper."""
    cases = make_cases(60, labels=LABELS)
    cache = Cache(tmp_path / "cache")
    execute.run(an_adapter(cases), cases, cache=cache, pricing_table=PRICING, workers=2)

    second = execute.run(
        an_adapter(cases),
        cases,
        cache=Cache(tmp_path / "cache"),
        pricing_table=PRICING,
        workers=2,
    )
    assert all(record.cost_usd is None for record in second.records)
    assert second.live_calls == []


def test_changing_a_call_parameter_invalidates_the_cache(tmp_path: Path) -> None:
    """The key covers call params, so a differently configured adapter is a different run."""
    cases = make_cases(40, labels=LABELS)
    cache = Cache(tmp_path / "cache")
    execute.run(an_adapter(cases, accuracy=0.8), cases, cache=cache, workers=2)

    changed = execute.run(
        an_adapter(cases, accuracy=0.6), cases, cache=Cache(tmp_path / "cache"), workers=2
    )
    assert all(not record.from_cache for record in changed.records)


def test_reordering_labels_does_not_invalidate_the_cache() -> None:
    cases = make_cases(4, labels=LABELS)
    adapter = an_adapter(cases)
    forward = cache_key(adapter, cases[0].text, ["a", "b", "c"])
    backward = cache_key(adapter, cases[0].text, ["c", "b", "a"])
    assert forward == backward


def test_a_disabled_cache_never_reads_or_writes(tmp_path: Path) -> None:
    cases = make_cases(20, labels=LABELS)
    cache = Cache(tmp_path / "cache", enabled=False)
    execute.run(an_adapter(cases), cases, cache=cache, workers=2)
    assert cache.stats == {"hits": 0, "misses": 0, "writes": 0}
    assert not (tmp_path / "cache").exists()


def test_a_corrupt_cache_entry_is_a_miss_rather_than_a_crash(tmp_path: Path) -> None:
    cases = make_cases(10, labels=LABELS)
    cache = Cache(tmp_path / "cache")
    execute.run(an_adapter(cases), cases, cache=cache, workers=1)

    key = cache_key(an_adapter(cases), cases[0].text, list(cases[0].labels))
    cache.path_for(key).write_text("{ this is not json", encoding="utf-8")

    reread = Cache(tmp_path / "cache")
    result = execute.run(an_adapter(cases), cases, cache=reread, workers=1)
    assert result.records[0].from_cache is False
    assert all(record.ok for record in result.records)


# The cost guard


def test_too_many_cases_aborts_before_anything_is_sent() -> None:
    cases = make_cases(500, labels=LABELS)
    with pytest.raises(CostGuardError, match="max_cases"):
        execute.run(an_adapter(cases), cases, guard=CostGuard(max_cases=100))


def test_the_case_guard_refuses_to_truncate_silently() -> None:
    cases = make_cases(500, labels=LABELS)
    with pytest.raises(CostGuardError, match=r"will not.*silently truncate"):
        execute.run(an_adapter(cases), cases, guard=CostGuard(max_cases=100))


def test_an_estimate_over_budget_aborts_before_anything_is_sent() -> None:
    cases = make_cases(2000, labels=LABELS)
    adapter = an_adapter(cases)
    with pytest.raises(CostGuardError, match="exceeds max_cost_usd"):
        execute.run(
            adapter,
            cases,
            guard=CostGuard(max_cost_usd=0.000001),
            pricing_table=PRICING,
        )


def test_a_budget_with_no_pricing_row_aborts_rather_than_guessing() -> None:
    cases = make_cases(50, labels=LABELS)
    with pytest.raises(CostGuardError, match="not in the pricing table"):
        execute.run(an_adapter(cases), cases, guard=CostGuard(max_cost_usd=1.0), pricing_table={})


def test_a_run_inside_budget_proceeds_and_records_the_estimate() -> None:
    cases = make_cases(100, labels=LABELS)
    result = execute.run(
        an_adapter(cases),
        cases,
        guard=CostGuard(max_cost_usd=100.0),
        pricing_table=PRICING,
        workers=4,
    )
    assert len(result.records) == 100
    assert result.config["estimated_cost_usd"] > 0


def test_the_guard_runs_before_any_call_is_made() -> None:
    cases = make_cases(500, labels=LABELS)
    adapter = FlakyAdapter(an_adapter(cases), failures_before_success=0)
    with pytest.raises(CostGuardError):
        execute.run(adapter, cases, guard=CostGuard(max_cases=10))
    assert adapter.calls == {}


# Retries and failures


def test_a_transient_failure_is_retried_and_then_succeeds() -> None:
    cases = make_cases(20, labels=LABELS)
    adapter = FlakyAdapter(an_adapter(cases), failures_before_success=2)
    slept: list[float] = []

    result = execute.run(
        adapter,
        cases,
        workers=1,
        retry=RetryPolicy(max_attempts=3, backoff_seconds=0.5, sleep=slept.append),
    )
    assert all(record.ok for record in result.records)
    assert all(record.attempts == 3 for record in result.records)
    assert slept[:2] == [0.5, 1.0]


def test_backoff_doubles_and_the_first_attempt_does_not_wait() -> None:
    policy = RetryPolicy(backoff_seconds=0.5)
    assert [policy.delay_before(attempt) for attempt in (1, 2, 3, 4)] == [0.0, 0.5, 1.0, 2.0]


def test_a_failure_that_exhausts_its_retries_is_recorded_not_dropped() -> None:
    """Dropping failures silently would make an unreliable adapter look accurate."""
    cases = make_cases(15, labels=LABELS)
    adapter = FlakyAdapter(an_adapter(cases), failures_before_success=99)

    result = execute.run(
        adapter, cases, workers=1, retry=RetryPolicy(max_attempts=2, sleep=lambda _: None)
    )
    assert len(result.records) == 15
    assert len(result.failures) == 15
    assert result.successes == []
    for record in result.failures:
        assert record.prediction is None
        assert "TimeoutError" in str(record.error)
        assert record.attempts == 2


def test_a_refusal_is_recorded_as_a_refusal_and_is_not_retried() -> None:
    """A refusal is a decision. Retrying it produces the same refusal, slower."""
    cases = make_cases(10, labels=LABELS)
    adapter = RefusingAdapter()

    result = execute.run(
        adapter, cases, workers=1, retry=RetryPolicy(max_attempts=3, sleep=lambda _: None)
    )
    assert adapter.calls == 10
    assert all(record.refused for record in result.records)
    assert all(record.attempts == 1 for record in result.records)
    assert all("single token" in str(record.error) for record in result.records)


def test_failures_are_excluded_from_accuracy_rather_than_scored_wrong() -> None:
    cases = make_cases(40, labels=LABELS)
    inner = an_adapter(cases)

    class HalfBroken(FlakyAdapter):
        def classify(self, text: str, labels: list[str], **asked: object) -> Prediction:
            if text.endswith(("0", "1", "2", "3", "4")):
                raise RuntimeError("upstream is unwell")
            return self.inner.classify(text, labels, **asked)  # type: ignore[arg-type]

    adapter = HalfBroken(inner, failures_before_success=0)
    result = execute.run(
        adapter, cases, workers=1, retry=RetryPolicy(max_attempts=1, sleep=lambda _: None)
    )
    assert result.failures
    assert len(result.outcomes) == len(result.successes)


# The artifact


def test_the_artifact_refuses_to_pick_its_own_directory() -> None:
    """No default. Where user records land must not depend on the cwd."""
    cases = make_cases(4, labels=LABELS)
    result = execute.run(an_adapter(cases), cases, workers=1)

    with pytest.raises(TypeError):
        result.write()  # type: ignore[call-arg]


def test_the_artifact_records_what_actually_answered(tmp_path: Path) -> None:
    cases = make_cases(30, labels=LABELS)
    adapter = an_adapter(cases, model_requested="mock-alias", model_reported="mock-1.13")
    result = execute.run(adapter, cases, pricing_table=PRICING, workers=2)

    path = result.write(tmp_path / "results")
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert stored["adapter_name"] == "mock"
    assert stored["probability_semantics"] == "calibrated_claim"
    assert stored["model_requested"] == "mock-alias"
    assert stored["model_reported"] == "mock-1.13"
    assert stored["revision"] is None
    assert stored["dataset_hash"]
    assert stored["timestamp"]
    assert len(stored["records"]) == 30


def test_every_case_record_carries_a_prompt_hash_and_its_distribution(tmp_path: Path) -> None:
    cases = make_cases(20, labels=LABELS)
    result = execute.run(an_adapter(cases), cases, workers=2)
    stored = json.loads(result.write(tmp_path / "results").read_text(encoding="utf-8"))

    for record in stored["records"]:
        assert len(record["prompt_hash"]) == 32
        assert record["prediction"]["distribution"] is not None
        assert set(record["prediction"]["distribution"]) == set(LABELS)


def test_an_adapter_supplied_prompt_hash_is_preferred() -> None:
    """A local model hashes the prompt it actually built, and the runner uses that."""
    cases = make_cases(5, labels=LABELS)
    inner = an_adapter(cases)

    class Hashing(FlakyAdapter):
        def classify(self, text: str, labels: list[str], **asked: object) -> Prediction:
            from dataclasses import replace

            prediction = self.inner.classify(text, labels, **asked)  # type: ignore[arg-type]
            return replace(prediction, raw={**prediction.raw, "prompt_hash": "deadbeef"})

    result = execute.run(Hashing(inner, 0), cases, workers=1)
    assert all(record.prompt_hash == "deadbeef" for record in result.records)


def test_the_dataset_hash_changes_with_the_data_and_not_with_order() -> None:
    cases = make_cases(50, labels=LABELS)
    assert execute.dataset_hash(cases) == execute.dataset_hash(list(cases))
    assert execute.dataset_hash(cases) != execute.dataset_hash(cases[:49])


def test_credentials_never_reach_the_artifact(tmp_path: Path) -> None:
    """The one thing that must never be written, checked on the written bytes."""
    cases = make_cases(5, labels=LABELS)
    result = execute.run(
        an_adapter(cases),
        cases,
        workers=1,
        extra_config={
            "api_key": "ts-secret-value",
            "base_url": "https://example.invalid",
            "nested": {"auth_token": "another-secret", "model": "mock-1"},
        },
    )
    raw = result.write(tmp_path / "results").read_text(encoding="utf-8")

    assert "ts-secret-value" not in raw
    assert "another-secret" not in raw
    assert "[redacted]" in raw
    assert "https://example.invalid" in raw


@pytest.mark.parametrize(
    "key", ["api_key", "TYPESAFE_API_KEY", "authToken", "client_secret", "Authorization"]
)
def test_anything_that_looks_like_a_credential_is_redacted(key: str) -> None:
    assert execute.redact({key: "sensitive"})[key] == "[redacted]"


def test_ordinary_config_survives_redaction() -> None:
    assert execute.redact({"workers": 8, "model": "jev-1.13"}) == {
        "workers": 8,
        "model": "jev-1.13",
    }


# Pricing provenance and the reasons a cost column is blank


class SilentUsageAdapter(Adapter):
    """Reports that it can count tokens, then returns a response with none.

    This is the second reason a cost column is blank, and it is a fact about the
    run rather than about the adapter. The Jev wire schema marks both counts
    required while the SDK types them optional, so it is reachable in practice.
    """

    reports_tokens = True

    def __init__(self, inner: Adapter) -> None:
        self.inner = inner
        self.name = "silent-usage"
        self.model_requested = inner.model_requested
        self.revision = None
        self.probability_semantics = inner.probability_semantics

    def classify(self, text: str, labels: list[str], **asked: object) -> Prediction:
        from dataclasses import replace

        return replace(
            self.inner.classify(text, labels, **asked),  # type: ignore[arg-type]
            input_tokens=None,
            output_tokens=None,
        )


def test_the_artifact_records_which_pricing_entry_it_used_and_when_it_was_read(
    tmp_path: Path,
) -> None:
    cases = make_cases(6, labels=LABELS)
    result = execute.run(an_adapter(cases), cases, pricing_table=PRICING, workers=1)
    stored = json.loads(result.write(tmp_path / "results").read_text(encoding="utf-8"))

    assert stored["pricing_key"] == "mock-1"
    assert stored["pricing"]["as_of"] == READ_ON.isoformat()
    assert stored["pricing"]["source"] == "test fixture"
    assert stored["pricing"]["stale"] is False


def test_an_old_pricing_entry_is_flagged_rather_than_presented_as_current(
    tmp_path: Path,
) -> None:
    """A price read years ago is not today's price, and the artifact says so."""
    cases = make_cases(4, labels=LABELS)
    table = {
        "mock-1": Pricing(
            input_usd_per_million=1.0,
            output_usd_per_million=5.0,
            source="an old price list",
            as_of=date(2019, 1, 1),
        )
    }
    result = execute.run(an_adapter(cases), cases, pricing_table=table, workers=1)
    stored = json.loads(result.write(tmp_path / "results").read_text(encoding="utf-8"))

    assert stored["pricing"]["stale"] is True
    assert "may be out of date" in stored["pricing"]["statement"]


def test_a_run_without_a_pricing_table_records_no_pricing_entry() -> None:
    cases = make_cases(4, labels=LABELS)
    result = execute.run(an_adapter(cases), cases, workers=1)

    assert result.pricing_key is None
    assert result.pricing is None


def test_a_priced_row_says_it_was_priced() -> None:
    cases = make_cases(4, labels=LABELS)
    result = execute.run(an_adapter(cases), cases, pricing_table=PRICING, workers=1)

    assert {record.cost_basis for record in result.records} == {"priced"}
    assert all(record.pricing_key == "mock-1" for record in result.records)


def test_an_adapter_that_cannot_report_tokens_says_that_and_not_that_they_are_missing() -> None:
    """A local model's blank column is a property of the adapter, not of the run."""
    cases = make_cases(4, labels=LABELS)
    result = execute.run(
        an_adapter(cases, report_tokens=False), cases, pricing_table=PRICING, workers=1
    )

    assert {record.cost_basis for record in result.records} == {"adapter_reports_no_tokens"}
    assert all(record.cost_usd is None for record in result.records)


def test_an_api_that_returned_no_tokens_is_distinguished_from_one_that_never_does() -> None:
    cases = make_cases(4, labels=LABELS)
    adapter = SilentUsageAdapter(an_adapter(cases))
    result = execute.run(adapter, cases, pricing_table=PRICING, workers=1)

    assert {record.cost_basis for record in result.records} == {"tokens_not_reported"}


def test_tokens_with_no_pricing_row_are_reported_as_unpriced_rather_than_free() -> None:
    cases = make_cases(4, labels=LABELS)
    result = execute.run(an_adapter(cases), cases, pricing_table={}, workers=1)

    assert {record.cost_basis for record in result.records} == {"model_not_priced"}


def test_a_cache_hit_is_not_charged_and_says_why(tmp_path: Path) -> None:
    cases = make_cases(4, labels=LABELS)
    cache = Cache(tmp_path / "cache")
    execute.run(an_adapter(cases), cases, cache=cache, pricing_table=PRICING, workers=1)
    result = execute.run(an_adapter(cases), cases, cache=cache, pricing_table=PRICING, workers=1)

    assert {record.cost_basis for record in result.records} == {"cache_hit"}
    assert all(record.cost_usd is None for record in result.records)


def test_a_failed_case_has_no_cost_basis_to_report() -> None:
    cases = make_cases(3, labels=LABELS)
    adapter = FlakyAdapter(an_adapter(cases), failures_before_success=99)
    result = execute.run(adapter, cases, retry=RetryPolicy(max_attempts=1), workers=1)

    assert {record.cost_basis for record in result.records} == {"no_prediction"}


def test_the_artifact_carries_the_reason_each_cost_is_blank(tmp_path: Path) -> None:
    cases = make_cases(4, labels=LABELS)
    result = execute.run(
        an_adapter(cases, report_tokens=False), cases, pricing_table=PRICING, workers=1
    )
    stored = json.loads(result.write(tmp_path / "results").read_text(encoding="utf-8"))

    assert {record["cost_basis"] for record in stored["records"]} == {"adapter_reports_no_tokens"}


def test_the_artifact_records_how_many_rows_the_run_covered(tmp_path: Path) -> None:
    """The dataset hash identifies the rows; the count is what ECE is read against."""
    cases = make_cases(12, labels=LABELS)
    result = execute.run(an_adapter(cases), cases, workers=1)

    stored = json.loads(result.write(tmp_path / "results").read_text(encoding="utf-8"))

    assert result.dataset_rows == 12
    assert stored["dataset_rows"] == 12
    assert stored["dataset_hash"]


# What the row asks, and how the adapter asked it


def a_noul_case(case_id: str = "yn-1") -> Case:
    return Case(
        id=case_id,
        text="should this dispute be decided in the cardholder's favour?",
        labels=("no", "yes"),
        gold_label="yes",
        question_type="noul",
    )


def test_the_artifact_records_what_a_row_asks_and_how_it_was_asked(tmp_path: Path) -> None:
    """The mock asks everything as a choice, including yes/no rows. Say so."""
    cases = [a_noul_case()]
    result = execute.run(an_adapter(cases), cases, workers=1)

    stored = json.loads(result.write(tmp_path / "results").read_text(encoding="utf-8"))
    record = stored["records"][0]

    assert result.records[0].question_type == "noul"
    assert result.records[0].asked_as == "choice"
    assert record["question_type"] == "noul"
    assert record["asked_as"] == "choice"


def test_a_choice_row_records_both_as_choice() -> None:
    cases = make_cases(2, labels=LABELS)
    result = execute.run(an_adapter(cases), cases, workers=1)

    assert {record.question_type for record in result.records} == {"choice"}
    assert {record.asked_as for record in result.records} == {"choice"}


def test_the_cache_does_not_confuse_a_yes_no_ask_with_a_two_option_choice() -> None:
    """Same text, same options, different question. Different answer, different key."""
    cases = [a_noul_case()]
    adapter = an_adapter(cases)

    as_choice = cache_key(adapter, cases[0].text, list(cases[0].labels))
    as_noul = cache_key(adapter, cases[0].text, list(cases[0].labels), question_type="noul")

    assert as_choice != as_noul
