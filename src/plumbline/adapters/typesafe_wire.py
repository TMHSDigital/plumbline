"""The Jev wire format over HTTP, via the TypeSafe Python SDK.

One classification case becomes one ``Choice`` question whose criteria are the
case's labels. The answer carries a selected label, a probability per label, and
a confidence, which is exactly the shape plumbline measures.

Two things this adapter deliberately does not do.

It does not compute cost. ``Prediction.cost_usd`` stays None and
``metrics/cost.py`` derives cost from the token counts and a pricing table, so
pricing is never hardcoded in an adapter and never silently goes stale.

It does not invent token counts. ``Usage.input_tokens`` and
``Usage.output_tokens`` are typed ``int | None`` by the SDK and default to None,
even though the wire schema marks both required. When the API reports no count,
this adapter reports None, and the cost path turns that into "not reported"
rather than into zero. A zero would read as free.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    SystemOneResponse,
    TypeSafeClient,
)

from plumbline.adapters.base import Adapter, check_probability_semantics
from plumbline.types import (
    CaseRefusedError,
    PlumblineError,
    Prediction,
    ProbabilitySemantics,
    QuestionType,
    yes_no_labels,
)

#: The question name sent to the API. One question per call, so the name is
#: internal, but it is recorded so a stored answer can be traced to its request.
QUESTION_NAME = "classification"

DEFAULT_INSTRUCTIONS = "Which label best describes this text?"

#: Sent with a yes/no case. The question itself is in the document, because
#: that is where a dataset's own wording lives; this only says how to answer.
DEFAULT_NOUL_INSTRUCTIONS = (
    "Answer the question stated in the document. Report the probability that the answer is yes."
)

#: The key the case text is filed under in the request state.
STATE_KEY = "document"


class WireContractError(PlumblineError):
    """The API answered, but not with the labels that were asked about.

    Separate from a transport error on purpose. A connection failure is the
    network's problem and the runner may retry it. This is the service
    answering a different question than the one posed, which no retry fixes and
    which must not be scored as though it were a prediction.
    """


class TypeSafeWireAdapter(Adapter):
    """Hosted Jev, or any Jev-compatible endpoint reached by ``base_url``.

    Args:
        model_requested: Model name or alias to ask for. Recorded alongside
            whatever the API says it actually used, which can differ.
        instructions: The question text sent with every case. Part of
            ``call_params``, because changing it changes the answer and must
            therefore invalidate the cache.
        api_key: Passed to the SDK. Omit it and the SDK reads
            ``TYPESAFE_API_KEY`` from the environment, which is the intended
            path. It is never stored on the adapter, never placed in
            ``call_params``, and never reaches ``raw``.
        base_url: Overrides the hosted endpoint. Answer-changing, so it is part
            of ``call_params``.
        timeout: Per-request timeout in seconds, passed to the SDK.
        probability_semantics: Defaults to ``"calibrated_claim"`` because that
            is what hosted Jev claims. Whether the claim survives contact with
            a dataset is the question plumbline exists to answer, so the label
            records the claim and never asserts it is true. A self-hosted
            endpoint making no such claim should say so here.
        client: A pre-built client, mainly so tests can exercise this adapter
            without a network or a key.
    """

    supported_question_types = ("choice", "noul")

    reports_tokens = True
    """This transport can report token counts, so a blank cost is about the run.

    When a row from this adapter has no cost, the runner records it as
    ``tokens_not_reported`` rather than ``adapter_reports_no_tokens``: the API
    could have said, and on that call it did not.
    """

    def __init__(
        self,
        *,
        model_requested: str = "jev-1",
        instructions: str = DEFAULT_INSTRUCTIONS,
        noul_instructions: str = DEFAULT_NOUL_INSTRUCTIONS,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        probability_semantics: ProbabilitySemantics = "calibrated_claim",
        client: TypeSafeClient | None = None,
    ) -> None:
        self.name = "typesafe_wire"
        self.model_requested = model_requested
        self.revision = None  # Hosted models are not pinned by checkpoint.
        self.probability_semantics = check_probability_semantics(probability_semantics)
        self.instructions = instructions
        self.noul_instructions = noul_instructions
        self.base_url = base_url
        self.timeout = timeout
        self._owns_client = client is None
        self._client = client or TypeSafeClient(
            api_key=api_key, model=model_requested, base_url=base_url, timeout=timeout
        )

    @property
    def call_params(self) -> Mapping[str, object]:
        """What changes the answer, and therefore the cache key.

        The base class already folds in the adapter name, ``model_requested``,
        ``revision``, the case text, and the sorted labels. Added here are the
        instructions and the endpoint. The key is deliberately absent: it
        authenticates the caller, it does not change the answer, and a cache key
        is written to disk.
        """
        return {
            "instructions": self.instructions,
            "noul_instructions": self.noul_instructions,
            "base_url": self.base_url,
            "question_name": QUESTION_NAME,
        }

    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        question_type: QuestionType = "choice",
    ) -> Prediction:
        """Ask one question of the kind the dataset says this row is."""
        if not labels:
            raise ValueError("labels must not be empty")

        if question_type == "noul":
            return self._classify_noul(text, labels)
        if question_type != "choice":
            raise CaseRefusedError(
                f"typesafe_wire does not ask {question_type!r} questions. A score question "
                "asks for an ordinal level, and asking it as a choice between unordered "
                "options throws the ordering away, so the case is refused rather than "
                "answered as something else."
            )

        question = Choice(
            instructions=self.instructions,
            criteria=dict.fromkeys(labels),
        )

        response, latency_ms = self._ask(text, question)
        return self._to_prediction(response, labels, latency_ms)

    def _classify_noul(self, text: str, labels: list[str]) -> Prediction:
        """Ask a yes/no question as a Noul: one probability, nothing derived.

        This is the cleanest calibration target the API offers. There is no
        distribution to renormalize and no confidence statistic computed from
        one, so ``prob_selected`` is exactly what the vendor reported, mapped
        onto whichever answer was given.
        """
        pair = yes_no_labels(labels)
        if pair is None:
            raise CaseRefusedError(
                f"this row is a yes/no question but its options {sorted(labels)!r} are not a "
                "yes/no pair. Which option is the yes is a fact about the dataset, and "
                "guessing it would invert every probability on the row, so the case is "
                "refused."
            )
        affirmative, negative = pair

        response, latency_ms = self._ask(text, Noul(instructions=self.noul_instructions))
        answer = self._noul_for(response)

        probability_of_yes = float(answer.noul)
        if not 0.0 <= probability_of_yes <= 1.0:
            raise WireContractError(
                f"the API reported a noul of {probability_of_yes!r}, which is not a "
                "probability. Nothing is clamped: a number outside [0, 1] is a contract "
                "failure, not a value to repair."
            )

        # noul is P(yes). The probability of the answer actually given is noul
        # for yes and 1 - noul for no; reporting noul for a "no" answer would
        # put the wrong number in the calibration column.
        said_yes = probability_of_yes >= 0.5
        label = affirmative if said_yes else negative
        prob_selected = probability_of_yes if said_yes else 1.0 - probability_of_yes

        usage = response.usage
        return Prediction(
            label=label,
            prob_selected=prob_selected,
            # A Noul has no distribution and therefore no statistic computed
            # from one. Both stay None rather than being synthesized from the
            # single number, which would invent a shape the vendor never sent.
            distribution=None,
            confidence=None,
            latency_ms=latency_ms,
            cost_usd=None,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model_reported=response.model,
            raw={
                "model": response.model,
                "question_name": QUESTION_NAME,
                "asked_as": "noul",
                "noul": probability_of_yes,
                "affirmative_label": affirmative,
                "negative_label": negative,
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                },
            },
        )

    def _ask(
        self, text: str, question: Choice | Noul
    ) -> tuple[SystemOneResponse, float]:
        """One request, one question, with the latency it took."""
        started = time.perf_counter()
        response = self._client.system_one(
            state={STATE_KEY: text},
            questions={QUESTION_NAME: question},
        )
        return response, (time.perf_counter() - started) * 1000.0

    def _noul_for(self, response: SystemOneResponse) -> NoulAnswer:
        """The one Noul answer, or a contract error naming what arrived."""
        try:
            return response.nouls[QUESTION_NAME]
        except KeyError:
            raise WireContractError(
                f"no noul answer named {QUESTION_NAME!r} in the response. A yes/no question "
                "was asked and something else came back, which is not a prediction to "
                f"score. Answers present: {sorted(response.answers)!r}"
            ) from None

    def _to_prediction(
        self, response: SystemOneResponse, labels: list[str], latency_ms: float
    ) -> Prediction:
        answer = self._answer_for(response)
        distribution = self._checked_distribution(answer.probabilities, labels)

        if answer.choice not in distribution:
            raise WireContractError(
                f"the API selected {answer.choice!r}, which is not one of the "
                f"labels that were asked about: {sorted(labels)!r}"
            )

        # Both counts stay exactly as reported. See the module docstring.
        #
        # TODO(first live call): the wire schema
        # (typesafe_sdk/_schemas/models.py) marks input_tokens and output_tokens
        # required, while the SDK response type
        # (typesafe_sdk/_core/response_types.py) widens both to `int | None` and
        # defaults them to None. Only a live call settles which is true in
        # practice. Whoever makes the first one: record whether usage came back
        # populated, on which model, and on what date, in docs/PLAN.md under
        # open questions. If the API does populate them, the None path here
        # stays anyway -- it costs nothing and it is the difference between a
        # blank cost and a false zero -- but the report can stop hedging about
        # how often it is taken.
        usage = response.usage

        return Prediction(
            label=answer.choice,
            prob_selected=distribution[answer.choice],
            distribution=distribution,
            confidence=answer.confidence,
            latency_ms=latency_ms,
            cost_usd=None,  # Derived in metrics/cost.py from the tokens below.
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model_reported=response.model,
            raw=self._raw(response, answer.choice, distribution),
        )

    def _answer_for(self, response: SystemOneResponse) -> ChoiceAnswer:
        """The one Choice answer, or a contract error naming what arrived."""
        try:
            return response.choices[QUESTION_NAME]
        except KeyError:
            raise WireContractError(
                f"no choice answer named {QUESTION_NAME!r} in the response. "
                f"Answers present: {sorted(response.answers)!r}"
            ) from None

    @staticmethod
    def _checked_distribution(
        probabilities: Mapping[str, float], labels: list[str]
    ) -> dict[str, float]:
        """Require the reported labels to be the labels that were asked about.

        Not renormalized and not filtered. A distribution over a different label
        set is not a rescaling problem, it is a different question, and silently
        reshaping it would put a number in the calibration table that no model
        ever reported.
        """
        reported = set(probabilities)
        asked = set(labels)
        if reported != asked:
            missing = sorted(asked - reported)
            unexpected = sorted(reported - asked)
            raise WireContractError(
                "the API reported probabilities over a different label set than "
                f"was asked about. Missing: {missing!r}. Unexpected: {unexpected!r}"
            )
        return dict(probabilities)

    @staticmethod
    def _raw(
        response: SystemOneResponse, choice: str, distribution: dict[str, float]
    ) -> dict[str, Any]:
        """What is kept for the artifact. No credentials, no headers, no key."""
        return {
            "model": response.model,
            "question_name": QUESTION_NAME,
            "asked_as": "choice",
            "choice": choice,
            "probabilities": distribution,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    def close(self) -> None:
        """Close the HTTP client, unless the caller supplied its own."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> TypeSafeWireAdapter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
