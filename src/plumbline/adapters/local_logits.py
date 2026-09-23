"""Restricted-vocabulary logit readout from a pinned local checkpoint.

This arm is the null hypothesis of the whole tool. It asks an open checkpoint
one question, reads the logits of the next token, and takes a softmax over the
option tokens alone. Nobody claims those numbers are calibrated. If a hosted
model's calibrated claim does not beat this on your data, the claim is not worth
paying for, so this arm has to be easy to run and impossible to mistake for
something stronger.

Three things it refuses to do.

It refuses to run unpinned. ``revision`` is required, and a branch or tag is
rejected unless the caller asks for that explicitly, because a moving reference
means two runs a week apart can disagree for reasons that are not in the data.
What actually answered is recorded next to what was asked for.

It refuses a case it cannot ask cleanly. Every option must map to exactly one
token, and two options must not map to the same one. When they do not, the case
is refused. Truncating an option to its first token would silently change the
question: "shipping" and "shipped" can share a first token, and a softmax over
first tokens is a softmax over a different label set than the one the dataset
declares.

It reports no tokens and therefore no cost. A local checkpoint's real cost is
hardware and wall-clock, not billable tokens, so ``reports_tokens`` is False and
the runner records every row as ``adapter_reports_no_tokens`` rather than
leaving a blank column that could be read as free.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Protocol

from plumbline.adapters.base import Adapter
from plumbline.types import CaseRefusedError, PlumblineError, Prediction, QuestionType

DEFAULT_INSTRUCTIONS = "Which label best describes this text?"

#: The one prompt shape this adapter sends. It is in ``call_params``, so editing
#: it invalidates the cache rather than mixing two prompt shapes in one table.
DEFAULT_PROMPT_TEMPLATE = "{instructions}\n\nText: {text}\n\nOptions: {options}\n\nAnswer:"

#: Prepended to an option before tokenizing. Most byte-level tokenizers give a
#: word a different id at the start of a line than after a space, and the answer
#: position follows a space, so the leading space is part of the question.
DEFAULT_OPTION_PREFIX = " "

_COMMIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")


class CheckpointMismatchError(PlumblineError):
    """The checkpoint that answered is not the checkpoint that was pinned.

    Not a transport error and not retryable. Everything measured under the
    requested revision would be attributed to the wrong weights.
    """


class LogitReadout(Protocol):
    """The two things this adapter needs from a checkpoint.

    Deliberately narrow, so the adapter can be tested without loading a model
    and so a different runtime can be dropped in without touching the adapter.
    """

    @property
    def resolved_revision(self) -> str:
        """The revision that actually loaded, which may differ from the request."""

    def token_ids(self, text: str) -> Sequence[int]:
        """Tokenize ``text``, with no special tokens added."""

    def option_logits(self, prompt: str, token_ids: Sequence[int]) -> Sequence[float]:
        """Next-token logits at the end of ``prompt``, for those ids, in order."""


class LocalLogitsAdapter(Adapter):
    """One pinned open checkpoint, read as a restricted softmax.

    Args:
        model_requested: Checkpoint id, usually a HuggingFace repo name.
        revision: The commit to pin to. Required, and required to be a 40
            character commit sha unless ``allow_unpinned_revision`` is set.
        readout: The loaded checkpoint. Left as None, one is built from
            ``model_requested`` and ``revision`` with transformers, which is an
            optional dependency; tests inject a fake instead.
        instructions: The question text, sent with every case.
        prompt_template: Formatted with ``instructions``, ``text``, and
            ``options``. Answer-changing, so it is part of ``call_params``.
        option_prefix: Prepended to each option before tokenizing.
        allow_unpinned_revision: Accept a branch or tag. Recorded on every
            prediction, because a result produced under a moving reference
            cannot be reproduced from the artifact alone.
        device: Passed to the default readout. Not part of ``call_params``: it
            moves arithmetic, not the question.

    ``probability_semantics`` is fixed at ``"restricted_softmax"`` and is not a
    constructor argument. There is no configuration under which this arm becomes
    a calibrated claim, and the report groups on that field.
    """

    reports_tokens = False

    def __init__(
        self,
        *,
        model_requested: str,
        revision: str,
        readout: LogitReadout | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        option_prefix: str = DEFAULT_OPTION_PREFIX,
        allow_unpinned_revision: bool = False,
        device: str = "cpu",
    ) -> None:
        if not revision or not revision.strip():
            raise ValueError(
                "local_logits requires a revision. An unpinned checkpoint makes a result "
                "unreproducible: the weights behind the same name can change between runs "
                "and nothing in the artifact would say so."
            )
        if not allow_unpinned_revision and not _COMMIT_SHA.match(revision.strip()):
            raise ValueError(
                f"revision {revision!r} is not a 40 character commit sha. A branch or tag "
                "can move under a run, so plumbline pins to a commit. Pass the commit, or "
                "set allow_unpinned_revision=True to accept a moving reference, which is "
                "then recorded on every prediction."
            )

        self.name = "local_logits"
        self.model_requested = model_requested
        # The base interface allows no revision, because a hosted model has
        # none. This arm always has one, and keeps it under its own name so the
        # rest of the class can rely on that without re-checking.
        self.pinned_revision = revision.strip()
        self.revision = self.pinned_revision
        self.probability_semantics = "restricted_softmax"
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.option_prefix = option_prefix
        self.allow_unpinned_revision = allow_unpinned_revision
        self.device = device
        self._readout = readout

    @property
    def readout(self) -> LogitReadout:
        """The loaded checkpoint, built on first use so import stays cheap."""
        if self._readout is None:
            self._readout = TransformersReadout(
                model_id=self.model_requested,
                revision=self.pinned_revision,
                device=self.device,
            )
        return self._readout

    @property
    def call_params(self) -> Mapping[str, object]:
        """What changes the answer, and therefore the cache key.

        The base key already covers the adapter name, the model, the revision,
        the case text, and the sorted labels. The prompt shape is added here.
        """
        return {
            "instructions": self.instructions,
            "prompt_template": self.prompt_template,
            "option_prefix": self.option_prefix,
        }

    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        question_type: QuestionType = "choice",
    ) -> Prediction:
        """Read the option tokens, whatever kind of question the row is.

        A yes/no row is asked here as a two-option choice: there is no bare
        probability to read off a local checkpoint, only a softmax over the
        tokens "yes" and "no". That is a different question from the one a noul
        asks, so it is recorded as ``asked_as: choice`` and the report keeps the
        two apart rather than comparing them as though they matched.
        """
        if len(labels) < 2:
            raise ValueError(f"need at least 2 labels, got {len(labels)}")

        readout = self.readout
        token_ids = self._single_token_ids(readout, labels)
        self._check_revision(readout)

        prompt = self.prompt_template.format(
            instructions=self.instructions, text=text, options=", ".join(labels)
        )

        started = time.perf_counter()
        logits = list(readout.option_logits(prompt, token_ids))
        latency_ms = (time.perf_counter() - started) * 1000.0

        if len(logits) != len(labels):
            raise PlumblineError(
                f"the readout returned {len(logits)} logits for {len(labels)} options. "
                "A restricted softmax needs one logit per option, in order."
            )

        distribution = _restricted_softmax(labels, logits)
        selected = max(distribution, key=lambda label: distribution[label])

        return Prediction(
            label=selected,
            prob_selected=distribution[selected],
            distribution=distribution,
            # No vendor here, so no vendor summary statistic. Deriving one from
            # the distribution would put a number in the confidence column that
            # no system under test ever reported.
            confidence=None,
            latency_ms=latency_ms,
            cost_usd=None,
            input_tokens=None,
            output_tokens=None,
            model_reported=f"{self.model_requested}@{readout.resolved_revision}",
            raw={
                "asked_as": "choice",
                "question_type": question_type,
                "revision_requested": self.pinned_revision,
                "revision_resolved": readout.resolved_revision,
                "revision_pinned_by_commit": bool(_COMMIT_SHA.match(self.pinned_revision)),
                "option_token_ids": dict(zip(labels, token_ids, strict=True)),
                "option_logits": dict(zip(labels, logits, strict=True)),
                "probability_semantics": self.probability_semantics,
            },
        )

    def _single_token_ids(self, readout: LogitReadout, labels: list[str]) -> list[int]:
        """One token id per option, or a refusal naming what went wrong.

        Both failures here are the same mistake in different clothes: the
        question the checkpoint would actually be asked is not the question the
        dataset declares.
        """
        ids: list[int] = []
        multi: list[tuple[str, int]] = []
        for label in labels:
            encoded = list(readout.token_ids(f"{self.option_prefix}{label}"))
            if len(encoded) != 1:
                multi.append((label, len(encoded)))
            else:
                ids.append(encoded[0])

        if multi:
            named = ", ".join(f"{label!r} ({count} tokens)" for label, count in multi)
            raise CaseRefusedError(
                f"{self.model_requested} cannot ask this case cleanly: {named}. "
                "plumbline refuses the case rather than truncating an option to its "
                "first token, because a softmax over truncated options answers a "
                "different question than the one the dataset asks. Use options that are "
                "single tokens for this checkpoint, or run this case on another arm."
            )

        duplicates = [
            label for label, token in zip(labels, ids, strict=True) if ids.count(token) > 1
        ]
        if duplicates:
            raise CaseRefusedError(
                f"{self.model_requested} maps {sorted(duplicates)!r} to the same token, so "
                "the options cannot be told apart in a restricted softmax. The case is "
                "refused rather than scored against options the checkpoint cannot "
                "distinguish."
            )
        return ids

    def _check_revision(self, readout: LogitReadout) -> None:
        """Refuse when a pinned commit resolved to different weights."""
        resolved = readout.resolved_revision
        if _COMMIT_SHA.match(self.pinned_revision) and resolved != self.pinned_revision:
            raise CheckpointMismatchError(
                f"{self.model_requested} was pinned to {self.pinned_revision} and loaded "
                f"{resolved}. Nothing is measured against weights that were not asked "
                "for."
            )


def _restricted_softmax(labels: Sequence[str], logits: Sequence[float]) -> dict[str, float]:
    """Softmax over the option logits alone, which is all this arm ever claims.

    Conditional on the supplied options by construction: the mass the checkpoint
    puts on every other token in its vocabulary is discarded, not modeled. That
    is exactly why ``probability_semantics`` is ``"restricted_softmax"``.
    """
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return {label: weight / total for label, weight in zip(labels, weights, strict=True)}


class TransformersReadout:
    """The default readout: a causal LM loaded with transformers, pinned by revision.

    torch and transformers are an optional dependency (``plumbline[local]``), so
    they are imported here rather than at module import. plumbline installs and
    runs hosted-only on a machine with no GPU.
    """

    def __init__(self, *, model_id: str, revision: str, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as missing:  # pragma: no cover - exercised by installing extras
            raise PlumblineError(
                "the local_logits arm needs torch and transformers, which are an optional "
                "dependency. Install them with: uv sync --extra local (with pip, install "
                "from the repository: pip install "
                '"plumbline[local] @ git+https://github.com/TMHSDigital/plumbline"; the '
                "plumbline on PyPI is an unrelated project)"
            ) from missing

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
        self._model.to(device)
        self._model.eval()
        self._device = device
        # transformers records the commit it resolved to. When it does not, the
        # requested revision is the only thing known, and the adapter's own
        # check is what catches a mismatch it can see.
        self._resolved = str(getattr(self._model.config, "_commit_hash", None) or revision)

    @property
    def resolved_revision(self) -> str:
        return self._resolved

    def token_ids(self, text: str) -> Sequence[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def option_logits(self, prompt: str, token_ids: Sequence[int]) -> Sequence[float]:
        encoded = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            logits = self._model(**encoded).logits[0, -1, :]
        return [float(logits[token_id]) for token_id in token_ids]
