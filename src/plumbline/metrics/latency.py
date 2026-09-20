"""Latency percentiles. There is deliberately no mean here.

A mean hides the tail, and the tail is what decides whether a decision model can
sit in a hot path. A model averaging 40ms with a p99 of 2s is not a 40ms model
to anyone who has to serve it.

Cache hits are excluded upstream by the runner rather than filtered here: a hit
measures disk, not the model, and mixing the two would quietly improve every
percentile on a re-run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_PERCENTILES = (50.0, 95.0, 99.0)


@dataclass(frozen=True)
class LatencySummary:
    """Tail latency in milliseconds, over the cases that actually called out."""

    p50: float
    p95: float
    p99: float
    n: int
    excluded_cache_hits: int

    def __str__(self) -> str:
        tail = (
            f", excluding {self.excluded_cache_hits} cache hits" if self.excluded_cache_hits else ""
        )
        return (
            f"p50 {self.p50:.1f}ms, p95 {self.p95:.1f}ms, p99 {self.p99:.1f}ms "
            f"over {self.n} live calls{tail}"
        )


def summarize(
    latencies_ms: Sequence[float],
    *,
    excluded_cache_hits: int = 0,
) -> LatencySummary:
    """p50, p95, and p99 by nearest rank, so every figure is a latency that happened.

    Nearest rank, the inverse-CDF definition: the reported p99 is the smallest
    observed latency at or below which 99 percent of calls completed. The
    alternative, linear interpolation between neighbouring order statistics, can
    report a p99 that no call ever took, by averaging one fast call with one slow
    one. For a tail statistic whose whole purpose is to describe real worst cases,
    an invented number is the wrong answer.

    One consequence worth knowing: with 100 samples and a single slow call, p99 is
    the fast value, because 99 percent of calls really did complete that quickly.
    The slow call is the maximum, not the p99. Small samples cannot see far into
    the tail, and this reports that honestly instead of interpolating past it.

    Raises on an empty input rather than returning zeros, because zero latency is
    a claim and "nothing ran" is not.
    """
    if not latencies_ms:
        raise ValueError(
            "no live calls to summarize. Every case was served from cache or failed, "
            "so there is no latency to report."
        )
    values = np.asarray(latencies_ms, dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("latencies must be non-negative")

    p50, p95, p99 = (
        float(value) for value in np.percentile(values, DEFAULT_PERCENTILES, method="inverted_cdf")
    )
    return LatencySummary(
        p50=p50,
        p95=p95,
        p99=p99,
        n=len(values),
        excluded_cache_hits=excluded_cache_hits,
    )
