"""The text-generating control arm, via the Anthropic Messages API.

This arm reports no probability, and that is what it is for. It is the cost of
asking a text generator to classify: you get a label and nothing to threshold
on, so it is excluded from calibration entirely rather than given a probability
of 1.0 for whatever it wrote. Its accuracy is the floor a probability-reporting
system has to clear before its probabilities are worth discussing.

Three deliberate choices.

An answer that is not one of the options is refused, not repaired. Nearest-match
or first-word matching would turn the harness into the thing being measured, and
the rate at which a generator declines to answer the question as posed is a
finding worth reporting rather than hiding.

A refusal is not retried. plumbline sends one request per case. Retrying until
the generator lands on a valid label would measure the retry loop and report the
number as if one call had produced it.

No fallback model is configured. The Messages API can re-run a declined request
on another model inside the same call; for a measurement harness that would
quietly change the system under test partway through a dataset. A decline is
recorded as a refusal with its category, and the run stays about one model.
"""

from __future__ import annotations

import string
import time
from collections.abc import Mapping
from typing import Any, Literal, get_args

import anthropic
from anthropic.types import MessageParam, OutputConfigParam

from plumbline.adapters.base import Adapter
from plumbline.types import CaseRefusedError, Prediction, QuestionType

#: Default model. Recorded alongside whatever the API says actually answered.
DEFAULT_MODEL = "claude-opus-5"

DEFAULT_INSTRUCTIONS = (
    "You are classifying text. Answer with exactly one of the options you are "
    "given, copied exactly, and nothing else. No punctuation, no explanation."
)

DEFAULT_PROMPT_TEMPLATE = "Text:\n{text}\n\nOptions: {options}\n\nAnswer with one option."

#: Room for the answer plus whatever thinking the model does on the way to it.
#: The answer itself is a word; a cap tight enough to truncate would be measured
#: as this arm failing, which would be plumbline's fault rather than the model's.
DEFAULT_MAX_TOKENS = 1024

Effort = Literal["low", "medium", "high", "xhigh", "max"]

EFFORTS: tuple[Effort, ...] = get_args(Effort)

#: Effort is the one knob that still exists for depth. Sampling parameters were
#: removed from the current models and sending one returns a 400, so there is no
#: temperature here to set to zero: a classification prompt this narrow is
#: steered by the instructions and the effort level or not at all.
DEFAULT_EFFORT: Effort = "low"

#: Stripped from both ends of an answer before it is matched against the options.
_TRIM = string.whitespace + string.punctuation


class OffLabelAnswerError(CaseRefusedError):
    """The generator answered something that is not one of the options.

    A refusal rather than an error, so the runner records it once and does not
    retry it. It is a fact about this arm on this case, not a transient fault.
    """


class TruncatedAnswerError(CaseRefusedError):
    """The answer hit the token cap, so what came back is not a whole answer."""


class ModelRefusedError(CaseRefusedError):
    """The model declined the request. Recorded with its category, never scored."""


class GenerativeAdapter(Adapter):
    """One text generator, asked for a label and held to it.

    Args:
        model_requested: Model to ask for. What answered is recorded separately.
        instructions: The system prompt. Answer-changing, so it is part of
            ``call_params``.
        prompt_template: Formatted with ``text`` and ``options``.
        max_tokens: Response cap. Part of ``call_params`` because a cap that
            truncates changes what comes back.
        effort: ``low`` through ``max``. Depth of reasoning before the answer.
        api_key: Passed to the SDK. Omit it and the SDK resolves credentials
            from the environment, which is the intended path. It is never stored
            on the adapter, never placed in ``call_params``, and never reaches
            ``raw``.
        base_url: Overrides the endpoint. Answer-changing, so it is in
            ``call_params``.
        timeout: Per-request timeout in seconds.
        client: A pre-built client, mainly so tests can exercise this adapter
            without a network or a key.

    ``probability_semantics`` is fixed at ``"none"`` and is not a constructor
    argument. There is no probability here to relabel.
    """

    reports_tokens = True

    def __init__(
        self,
        *,
        model_requested: str = DEFAULT_MODEL,
        instructions: str = DEFAULT_INSTRUCTIONS,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: Effort = DEFAULT_EFFORT,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS!r}, got {effort!r}")

        self.name = "generative"
        self.model_requested = model_requested
        self.revision = None  # Hosted models are not pinned by checkpoint.
        self.probability_semantics = "none"
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.effort = effort
        self.base_url = base_url
        self.timeout = timeout
        self._api_key = api_key
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        """The SDK client, built on first use so no key is needed to import."""
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self._api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    @property
    def call_params(self) -> Mapping[str, object]:
        """What changes the answer, and therefore the cache key.

        The key is deliberately absent: it authenticates the caller, it does not
        change the answer, and a cache key is written to disk.
        """
        return {
            "instructions": self.instructions,
            "prompt_template": self.prompt_template,
            "max_tokens": self.max_tokens,
            "effort": self.effort,
            "base_url": self.base_url,
        }

    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        question_type: QuestionType = "choice",
    ) -> Prediction:
        """Ask for one option in text, whatever kind of question the row is.

        A generator has no probability to report under any question type, so a
        yes/no row is asked the same way as any other: pick one of these words.
        Recorded as ``asked_as: choice``, because that is what it is.
        """
        if len(labels) < 2:
            raise ValueError(f"need at least 2 labels, got {len(labels)}")

        prompt = self.prompt_template.format(text=text, options=", ".join(labels))

        started = time.perf_counter()
        # No `fallbacks` parameter, on purpose, and against the SDK's default
        # advice for this model family. Server-side fallback re-runs a declined
        # request on another model inside the same call, which is the right
        # default for an application and the wrong one for a measurement: the
        # dataset would be answered partly by one model and partly by another,
        # and the run would no longer be about the system under test. A decline
        # is raised as ModelRefusedError, recorded with its category, and
        # counted. If you are here to add a fallback, that is the tradeoff you
        # are making; docs/PLAN.md carries the decision.
        response = self.client.messages.create(
            model=self.model_requested,
            max_tokens=self.max_tokens,
            system=self.instructions,
            output_config=OutputConfigParam(effort=self.effort),
            messages=[MessageParam(role="user", content=prompt)],
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        written = self._answer_text(response)
        label = self._match(written, labels)
        usage = response.usage

        return Prediction(
            label=label,
            # No probability, no distribution, no vendor statistic. This arm is
            # excluded from calibration rather than given a probability of 1.0
            # for whatever it happened to write.
            prob_selected=None,
            distribution=None,
            confidence=None,
            latency_ms=latency_ms,
            cost_usd=None,  # Derived in metrics/cost.py from the tokens below.
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            model_reported=response.model,
            raw={
                "model": response.model,
                "asked_as": "choice",
                "question_type": question_type,
                "answer_text": written,
                "stop_reason": response.stop_reason,
                "effort": self.effort,
                "max_tokens": self.max_tokens,
            },
        )

    def _answer_text(self, response: Any) -> str:
        """The text the model wrote, or a refusal saying why there is none.

        ``stop_reason`` is checked before the content is read. A declined
        request and a truncated one both return a 200 with content in them, and
        reading either as a label would put an answer in the results that the
        model did not give.
        """
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise ModelRefusedError(
                f"the model declined this case (category: {category!r}). It is recorded "
                "as a refusal rather than scored, and no fallback model is substituted, "
                "because a run measures one model."
            )
        if response.stop_reason == "max_tokens":
            raise TruncatedAnswerError(
                f"the answer hit max_tokens ({self.max_tokens}) and is incomplete, so it "
                "is not read as a label. Raise max_tokens, or lower effort so less of the "
                "budget goes to reasoning."
            )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    def _match(self, written: str, labels: list[str]) -> str:
        """The option the generator wrote, or a refusal quoting what it wrote.

        Punctuation and case are stripped, because "Billing." is the same answer
        as "billing" written by a model that likes sentences. Nothing beyond
        that: no prefix matching, no nearest option, no first word. Each of
        those would decide the answer on the harness's behalf, and how often a
        generator fails to answer the question as posed is one of the things
        this arm exists to measure.
        """
        cleaned = written.strip().strip(_TRIM).strip()
        if not cleaned:
            raise OffLabelAnswerError(
                "the model returned no text, so there is no answer to record. The case is "
                "refused rather than scored."
            )

        for label in labels:
            if cleaned == label:
                return label
        folded = cleaned.casefold()
        for label in labels:
            if folded == label.casefold():
                return label

        raise OffLabelAnswerError(
            f"the model answered {cleaned!r}, which is not one of {sorted(labels)!r}. "
            "plumbline refuses the case rather than guessing which option was meant: "
            "repairing the answer would score the harness instead of the model, and how "
            "often this arm declines to answer the question as posed is itself a result."
        )
