"""Reading a labeled dataset off disk, and refusing the rows that cannot be scored.

One rule matters more than the rest: a row whose gold label is not one of its
own options is refused, never loaded. Such a row marks every system wrong by
construction, so it arrives in a report looking like a model failure when it is
a typo in a file. plumbline would rather run 110 rows and say which one it
dropped than run 111 and hand out a silent penalty.

Every other refusal follows the same shape. The loader never repairs a row, and
it never drops one quietly: each refusal carries the line number, the id, and
what was wrong, and the report states how many rows were read, loaded, and
refused. A caller that needs the whole dataset calls ``require_complete`` and
finds out before the run rather than after the bill.

Two readers live here. ``load_jsonl`` reads plumbline's own record shape, which
is what ``datasets/private/`` holds. ``load_jevbench`` reads the JevBench public
file, which is a fixture for proving the pipeline composes, not a reproduction
of anyone's benchmark (see ``datasets/public/README.md``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from plumbline.types import (
    QUESTION_TYPES,
    Case,
    DatasetError,
    QuestionType,
)

#: Separates the question from the state in a composed case text.
_TEXT_JOIN = "\n\n"


@dataclass(frozen=True)
class RowRefusal:
    """One row that was not loaded, and why it was not."""

    row: int  # 1-based line number in the file
    case_id: str | None
    reason: str

    def __str__(self) -> str:
        named = f" ({self.case_id})" if self.case_id else ""
        return f"line {self.row}{named}: {self.reason}"


@dataclass(frozen=True)
class LoadSummary:
    """What a load found, without its cases: the part a report prints.

    An artifact keeps this, so a report rendered from artifacts later still says
    how many rows were read, loaded, and refused, as the one written at run time
    did. The cases themselves are already in the artifact's records.
    """

    source: str
    rows_read: int
    loaded: int
    refusals: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    unsupported_by_type: Mapping[str, int] = field(default_factory=dict)

    def statement(self) -> str:
        """One line naming all three counts, because two of them are not enough."""
        parts = [
            f"{self.rows_read} rows read from {self.source}, "
            f"{self.loaded} loaded, {len(self.refusals)} refused."
        ]
        parts.extend(self.notes)
        return " ".join(parts)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "rows_read": self.rows_read,
            "loaded": self.loaded,
            "refusals": list(self.refusals),
            "notes": list(self.notes),
            "unsupported_by_type": dict(self.unsupported_by_type),
        }

    @classmethod
    def from_jsonable(cls, stored: Mapping[str, Any]) -> LoadSummary:
        return cls(
            source=str(stored["source"]),
            rows_read=int(stored["rows_read"]),
            loaded=int(stored["loaded"]),
            refusals=tuple(str(refusal) for refusal in stored.get("refusals", ())),
            notes=tuple(str(note) for note in stored.get("notes", ())),
            unsupported_by_type={
                str(name): int(count)
                for name, count in dict(stored.get("unsupported_by_type", {})).items()
            },
        )


@dataclass(frozen=True)
class LoadReport:
    """What a file contained, what came out of it, and what was left behind."""

    source: str
    cases: tuple[Case, ...]
    rows_read: int
    refusals: tuple[RowRefusal, ...] = ()
    notes: tuple[str, ...] = ()
    question_types: Mapping[str, int] = field(default_factory=dict)
    normalized_gold_rows: int = 0

    @property
    def row_count(self) -> int:
        """Rows that became cases. The number the report prints beside ECE."""
        return len(self.cases)

    @property
    def scoreable(self) -> tuple[Case, ...]:
        """The cases v0.1 is willing to turn into numbers.

        An ordinal score row is loaded and kept in ``cases`` (nothing is lost
        quietly) and left out of here, because flattening its levels into
        unordered options discards the ordering that makes it a score. A run
        takes this set; the count that was held back is in the notes.
        """
        return tuple(case for case in self.cases if case.is_scoreable)

    @property
    def unsupported_by_type(self) -> dict[str, int]:
        """How many loaded rows are of a type v0.1 will not score, by type."""
        counts: dict[str, int] = {}
        for case in self.cases:
            if not case.is_scoreable:
                counts[case.question_type] = counts.get(case.question_type, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def is_complete(self) -> bool:
        return not self.refusals

    def summary(self) -> LoadSummary:
        """This load without its cases, as an artifact stores it."""
        return LoadSummary(
            source=self.source,
            rows_read=self.rows_read,
            loaded=self.row_count,
            refusals=tuple(str(refusal) for refusal in self.refusals),
            notes=self.notes,
            unsupported_by_type=self.unsupported_by_type,
        )

    def statement(self) -> str:
        """One line naming all three counts, because two of them are not enough."""
        return self.summary().statement()

    def require_complete(self) -> None:
        """Refuse to proceed on a partial dataset, naming every row dropped."""
        if self.refusals:
            listed = "; ".join(str(refusal) for refusal in self.refusals)
            raise DatasetError(
                f"{len(self.refusals)} of {self.rows_read} rows in {self.source} could not "
                f"be loaded: {listed}. Fix the file, or call load without "
                "require_complete and accept a run over the rows that survived."
            )


def load_jsonl(path: Path | str) -> LoadReport:
    """Read plumbline's own JSONL: id, text, labels, gold_label, descriptions."""
    path = Path(path)
    rows = _read_rows(path)

    cases: list[Case] = []
    refusals: list[RowRefusal] = []
    seen: set[str] = set()

    for number, raw in rows:
        if isinstance(raw, str):
            refusals.append(RowRefusal(number, None, raw))
            continue
        case_id = _string_or_none(raw.get("id"))
        try:
            case = _case_from_record(raw)
        except DatasetError as refused:
            refusals.append(RowRefusal(number, case_id, str(refused)))
            continue
        if case.id in seen:
            refusals.append(_duplicate(number, case.id))
            continue
        seen.add(case.id)
        cases.append(case)

    return LoadReport(
        source=str(path),
        cases=tuple(cases),
        rows_read=len(rows),
        refusals=tuple(refusals),
    )


def load_jevbench(path: Path | str) -> LoadReport:
    """Read the JevBench public file into plumbline cases.

    This is a translation, not a reproduction. JevBench asks its rows through
    its own harness; plumbline composes a case text from the row's question and
    state, asks each row as its question type, and scores with its own metrics.
    Numbers from here are not comparable with JevBench's published ones, and
    ``datasets/public/README.md`` says so in the same words.

    What the translation does, all of it recorded in the report:

    * ``state`` becomes the case text, under the row's ``question.instructions``.
      A structured state is rendered as sorted, indented JSON so the same file
      always produces the same prompt and therefore the same prompt hash.
    * ``expected`` written as a JSON number is matched to the string option of
      the same name. ``1`` and ``"1"`` are the same option; the count of rows
      that needed this is reported rather than absorbed.
    * ``question.criteria`` becomes ``label_descriptions`` only when its keys are
      exactly the options. On the yes/no rows the criteria describe the
      statement rather than the options, and mapping them across would be a
      guess about someone else's file, so they are dropped and counted.
    """
    path = Path(path)
    rows = _read_rows(path)

    cases: list[Case] = []
    refusals: list[RowRefusal] = []
    types: dict[str, int] = {}
    seen: set[str] = set()
    normalized = 0
    dropped_criteria = 0

    for number, raw in rows:
        if isinstance(raw, str):
            refusals.append(RowRefusal(number, None, raw))
            continue
        case_id = _string_or_none(raw.get("id"))
        question = raw.get("question")
        question = question if isinstance(question, Mapping) else {}
        kind = _string_or_none(question.get("type")) or "unknown"
        types[kind] = types.get(kind, 0) + 1

        try:
            record, was_normalized, lost_criteria = _jevbench_record(raw, question)
            case = _case_from_record(record)
        except DatasetError as refused:
            refusals.append(RowRefusal(number, case_id, str(refused)))
            continue

        if case.id in seen:
            refusals.append(_duplicate(number, case.id))
            continue
        seen.add(case.id)
        normalized += int(was_normalized)
        dropped_criteria += int(lost_criteria)
        cases.append(case)

    notes = [
        "Translated from JevBench: the case text is the row's question above its state, "
        "and each row is asked as the question type it states. plumbline's harness, "
        "prompts and scoring differ from JevBench's, so these numbers are not comparable "
        "with theirs."
    ]
    if normalized:
        notes.append(
            f"{normalized} rows wrote the gold label as a JSON number against string "
            "options; each was matched to the option of the same name."
        )
    if dropped_criteria:
        notes.append(
            f"{dropped_criteria} rows carried criteria that do not describe the options "
            "one for one, so their option descriptions were dropped rather than guessed."
        )

    unsupported = sum(1 for case in cases if not case.is_scoreable)
    if unsupported:
        notes.append(
            f"{unsupported} rows ask for an ordinal score. plumbline v0.1 has no ordinal "
            "support: flattening levels into unordered options discards the ordering, "
            "so they are loaded, marked, and excluded from scored results."
        )

    return LoadReport(
        source=str(path),
        cases=tuple(cases),
        rows_read=len(rows),
        refusals=tuple(refusals),
        notes=tuple(notes),
        question_types=dict(sorted(types.items())),
        normalized_gold_rows=normalized,
    )


def _read_rows(path: Path) -> list[tuple[int, Mapping[str, Any] | str]]:
    """Parse every non-blank line, keeping bad lines as their own refusal text."""
    if not path.is_file():
        raise DatasetError(
            f"no dataset at {path}. plumbline never guesses a dataset location, because "
            "a relative default resolves against whatever directory the process started in."
        )

    rows: list[tuple[int, Mapping[str, Any] | str]] = []
    # Decoded line by line, so a file that is not UTF-8 is named with the line
    # that is not. The first line is read as utf-8-sig, so a byte order mark
    # (which Windows editors and spreadsheet exports write) is read as nothing
    # rather than refusing the first row.
    for number, raw in enumerate(path.read_bytes().splitlines(keepends=True), start=1):
        try:
            line = raw.decode("utf-8-sig" if number == 1 else "utf-8")
        except UnicodeDecodeError as undecodable:
            raise DatasetError(
                f"{path} is not UTF-8 text: line {number} holds the byte "
                f"{raw[undecodable.start]:#04x}, which UTF-8 does not allow. Save the "
                "file as UTF-8 and load it again."
            ) from None
        _parse_line(rows, number, line)

    if not rows:
        raise DatasetError(
            f"{path} holds no rows. An empty dataset is refused rather than run, because "
            "a run over nothing reports as cleanly as a run over everything."
        )
    return rows


def _parse_line(rows: list[tuple[int, Mapping[str, Any] | str]], number: int, line: str) -> None:
    """One line of the file: a row, a refusal to record, or nothing if it is blank."""
    if not line.strip():
        return  # A blank line is formatting, not a row.
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as broken:
        rows.append((number, f"line is not valid JSON: {broken.msg}"))
        return
    if not isinstance(parsed, dict):
        rows.append((number, f"line is a {type(parsed).__name__}, expected an object"))
        return
    rows.append((number, parsed))


def _duplicate(number: int, case_id: str) -> RowRefusal:
    """The refusal for an id already seen, the same from either loader."""
    return RowRefusal(
        number,
        case_id,
        f"duplicate id {case_id!r}; results are keyed by id, so a repeat would "
        "overwrite an earlier row",
    )


def _case_from_record(record: Mapping[str, Any]) -> Case:
    """Build one Case, or refuse with the reason spelled out."""
    for name in ("id", "text", "labels", "gold_label"):
        if name not in record:
            raise DatasetError(f"missing required field {name!r}")

    case_id = _string_or_none(record["id"])
    if not case_id:
        raise DatasetError("field 'id' must be a non-empty string")

    text = record["text"]
    if not isinstance(text, str) or not text.strip():
        raise DatasetError("field 'text' must be a non-empty string")

    raw_labels = record["labels"]
    if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, str | bytes):
        raise DatasetError("field 'labels' must be a list of strings")
    # Each label is taken as written. str() would have turned null into "None"
    # and true into "True", and loaded an option nobody wrote.
    for label in raw_labels:
        if not isinstance(label, str) or not label.strip():
            raise DatasetError(
                f"every label must be a non-empty string, and {label!r} is not; labels are "
                "the options the model chooses between, as written"
            )
    labels = tuple(raw_labels)
    if len(labels) < 2:
        raise DatasetError(f"a case needs at least 2 labels, got {len(labels)}")
    if len(set(labels)) != len(labels):
        raise DatasetError(f"labels contain duplicate entries: {list(labels)!r}")

    gold = str(record["gold_label"])
    if gold not in labels:
        raise DatasetError(
            f"gold_label {gold!r} is not one of this row's options {list(labels)!r}. The "
            "row is refused rather than loaded: scoring it would mark every system wrong "
            "on it, which reads as a model failure and is a dataset error."
        )

    descriptions = record.get("label_descriptions")
    if descriptions is not None:
        if not isinstance(descriptions, Mapping):
            raise DatasetError("field 'label_descriptions' must be an object")
        unknown = sorted(str(key) for key in descriptions if key not in labels)
        if unknown:
            raise DatasetError(
                f"label_descriptions describes {unknown!r}, which are not among this row's "
                f"options {list(labels)!r}; a description has to belong to an option"
            )
        for option, description in descriptions.items():
            if not isinstance(description, str) or not description.strip():
                raise DatasetError(
                    f"label_descriptions gives {option!r} the description {description!r}; "
                    "each description is a non-empty string"
                )

    question_type = record.get("question_type", "choice")
    if question_type not in QUESTION_TYPES:
        raise DatasetError(
            f"question_type {question_type!r} is not one of {list(QUESTION_TYPES)!r}. An "
            "unrecognized type is refused rather than assumed to be a choice: asking a "
            "question the wrong way round is not something to guess at."
        )

    return Case(
        id=case_id,
        text=text,
        labels=labels,
        gold_label=gold,
        label_descriptions=(
            {str(key): str(value) for key, value in descriptions.items()} if descriptions else None
        ),
        question_type=cast("QuestionType", question_type),
    )


def _jevbench_record(
    raw: Mapping[str, Any], question: Mapping[str, Any]
) -> tuple[dict[str, Any], bool, bool]:
    """Map one JevBench row onto plumbline's record shape."""
    labels = raw.get("labels")
    labels = [str(label) for label in labels] if isinstance(labels, list) else labels

    expected = raw.get("expected")
    normalized = expected is not None and not isinstance(expected, str)

    criteria = question.get("criteria")
    descriptions: dict[str, str] | None = None
    if (
        isinstance(criteria, Mapping)
        and isinstance(labels, list)
        and {str(key) for key in criteria} == set(labels)
    ):
        descriptions = {str(key): str(value) for key, value in criteria.items()}

    instructions = _string_or_none(question.get("instructions")) or ""
    text = _TEXT_JOIN.join(part for part in (instructions, _render_state(raw.get("state"))) if part)

    return (
        {
            "id": raw.get("id"),
            "text": text,
            "labels": labels,
            "gold_label": expected,
            "label_descriptions": descriptions,
            "question_type": question.get("type", "choice"),
        },
        normalized,
        isinstance(criteria, Mapping) and bool(criteria) and descriptions is None,
    )


def _render_state(state: Any) -> str:
    """A string state as written; anything else as stable JSON.

    Sorted keys and a fixed indent, so the same row always produces the same
    prompt. A prompt that varied between loads would change the prompt hash and
    quietly invalidate the cache.
    """
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False)


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
