"""On-disk prediction cache, so a metrics bugfix does not cost a second run.

Without this, re-running after correcting a metric means paying for every call
again, which in practice means the metric does not get corrected. The cache is
what makes the correctness work in Phase 2 and Phase 3 affordable.

The key covers everything that can change an answer: the adapter name, the model
requested, the pinned revision, the case text, the sorted label set, and the
adapter's own call parameters. Sorting the labels means reordering options does
not invalidate the cache, which is correct for every adapter here, since all of
them are asked to pick from a set.

A hit is recorded as a hit. It does not count toward cost or latency, because it
measures disk rather than the model.

Cases that share a key within one run are answered once: the runner holds a lock
per key across look up, call, and store, so the second waits and then hits. A
write that fails is counted and dropped rather than raised, because the answer
it was storing is already paid for.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from plumbline.adapters.base import Adapter
from plumbline.types import Prediction, QuestionType

CACHE_FORMAT_VERSION = 1


def cache_key(
    adapter: Adapter,
    text: str,
    labels: Sequence[str],
    question_type: QuestionType = "choice",
) -> str:
    """A stable fingerprint of everything that determines the answer.

    The question type is part of the key. The same text and the same two options
    asked as a yes/no and asked as a choice are different requests with
    different answers, and a shared key would serve one as the other.
    """
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "adapter": adapter.name,
        "model_requested": adapter.model_requested,
        "revision": adapter.revision,
        "text": text,
        "labels": sorted(labels),
        "question_type": question_type,
        "call_params": _canonical(dict(adapter.call_params)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(encoded.encode("utf-8"), digest_size=16).hexdigest()


def prompt_fingerprint(text: str, labels: Sequence[str]) -> str:
    """The request fingerprint recorded per case in the results artifact.

    Adapters that build a real prompt, such as the local logit readout, put their
    own ``prompt_hash`` in ``Prediction.raw`` and the runner prefers it. This is
    the fallback for adapters whose request is just the case and its options.
    """
    encoded = json.dumps(
        {"text": text, "labels": sorted(labels)}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.blake2b(encoded.encode("utf-8"), digest_size=16).hexdigest()


def _canonical(value: Any) -> Any:
    """Make call params comparable regardless of how they were constructed."""
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


class Cache:
    """A sharded directory of one JSON file per prediction.

    JSON files rather than a database, so a suspicious result can be read with a
    text editor and a bad entry removed with ``rm``. Sharded by the first two
    characters of the key, because a flat directory of a hundred thousand files
    is slow on Windows.
    """

    def __init__(self, directory: Path | str, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.write_errors = 0

    def path_for(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Prediction | None:
        if not self.enabled:
            return None
        path = self.path_for(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            prediction = _to_prediction(stored["prediction"])
        except (OSError, ValueError, KeyError, TypeError):
            # A corrupt or hand-edited entry is a miss, not a crash. The run
            # pays for that one case again and overwrites the bad file.
            self.misses += 1
            return None
        self.hits += 1
        return prediction

    def put(self, key: str, prediction: Prediction) -> None:
        """Store an answer, or count the failure to; never raise.

        Each write goes through a temporary file of its own, so two writers of
        one key cannot collide on a shared name. A write that still fails (a
        virus scanner holding the file on Windows, a full disk) is counted in
        ``stats`` and dropped: the answer is already in hand and paid for, and a
        missing entry costs one more call on a later run, not this run.
        """
        if not self.enabled:
            return
        path = self.path_for(key)
        payload = {"version": CACHE_FORMAT_VERSION, "prediction": to_jsonable(prediction)}
        temporary: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f"{key}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(json.dumps(payload, sort_keys=True, default=str))
            os.replace(temporary, path)
        except OSError:
            self.write_errors += 1
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink()
            return
        self.writes += 1

    @property
    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "write_errors": self.write_errors,
        }


def to_jsonable(prediction: Prediction) -> dict[str, Any]:
    """Flatten a prediction for storage, keeping the full distribution."""
    return dataclasses.asdict(prediction)


def to_prediction(stored: dict[str, Any]) -> Prediction:
    """Rebuild a prediction from its stored form, for the cache and the artifact."""
    return _to_prediction(stored)


def _to_prediction(stored: dict[str, Any]) -> Prediction:
    return Prediction(
        label=stored["label"],
        prob_selected=stored["prob_selected"],
        distribution=stored["distribution"],
        confidence=stored["confidence"],
        latency_ms=stored["latency_ms"],
        cost_usd=stored["cost_usd"],
        input_tokens=stored["input_tokens"],
        output_tokens=stored["output_tokens"],
        model_reported=stored["model_reported"],
        raw=stored.get("raw", {}),
    )
