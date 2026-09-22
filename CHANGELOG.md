# Changelog

Notable changes per release. Dates are the release date, not the tag date when
those differ.

This project is pre-1.0. The measurement behaviour is the stable part; the
Python API and the CLI flags are not, and a minor version may change either.
Anything that changes what a number **means** is called out under Changed, not
buried under Fixed, because a figure that moved for a methodology reason is a
different event from one that moved because it was wrong.

## Unreleased

### Added

- A browser calculator for the ECE floor (`site/`), a JavaScript port of
  `synthetic_floor` that reproduces numpy's seeded random stream draw for draw.
  `scripts/floor_golden.py` exports golden values from the Python and
  `scripts/check_floor_parity.mjs` holds the port to them within 1e-9 in CI.
  The page is not deployed yet.
- The site page: the argument, a worked example that derives the example
  report's ECE line from its 105 rows in the browser, and a sample-size planner.
  `scripts/build_site.py` regenerates the example from the mock adapter at build
  time and refuses if `docs/example-report.md` no longer matches its own
  command.

### Changed

- Report bullets separate the label from the figure with a colon instead of an
  em dash (`- **Cost**: not reported.`), and the docs no longer use em dashes.
  A CI job now fails on an em dash in tracked markdown or `src/`. **This
  changes report text, not any number.**

### Fixed

- The inconclusive verdict read as a pass. "Not distinguishable from a
  perfectly calibrated model at this sample size" is what the arithmetic
  establishes and close to the opposite of what it means, and readers took it
  as a clean result. Every such figure now leads with `INCONCLUSIVE` and states
  that nothing was established in either direction. **This changes report text,
  not any number.**
- The package shipped no PEP 561 `py.typed` marker, so downstream type checkers
  ignored its annotations and consumers silently saw `Any`.
- `plumbline version` printed `0.1.0.dev0` from the v0.1.0 release, because the
  test asserting the version used a substring match that `0.1.0.dev0` satisfies.
- `plumbline report` on a missing or malformed artifact raised a bare
  `FileNotFoundError` traceback instead of refusing with a reason.
- An adapter built without a required setting raised a bare `TypeError` from
  `__init__` instead of naming the setting.
- CI ran every pull request twice, because both the `push` and `pull_request`
  triggers fired on a branch pushed to origin.

### Documentation

- README rewritten for a reader arriving from a link: what the tool is now
  precedes what it is not, and decision model, calibration, cascade, Noul,
  binning noise and the Jev wire format are each defined where they appear.
  Adds prerequisites, a bash quickstart, and a note that this is not on PyPI.
- The adapters table gained a "Run for real" column. Two of the three
  transports have never run outside the test suite.
- SECURITY.md distinguishes what secret scanning covers from what it does not,
  since the closest thing to a disclosure this project has had was a vendor's
  price, which no scanner recognises.

## v0.1.0 (2026-09-21)

First release.

### Added

- **Calibration measured against its own null.** ECE, MCE and Brier are each
  reported beside the floor a perfectly calibrated model would produce at the
  same row count, computed by simulation. A figure inside its floor is reported
  as unresolvable rather than as a result.
- **Choice and Noul question types.** A yes/no row is asked as a Noul where the
  transport has one, and every record carries both what the row asks and how it
  was asked, so the two are never averaged together.
- **Three adapter transports.** `typesafe_wire` for the Jev wire format over
  HTTP, `local_logits` for option-token logits from a pinned local checkpoint,
  and `generative` as a text-generating control arm. Plus a seeded `mock`.
- **Three probability semantics classes.** `calibrated_claim`,
  `restricted_softmax` and `none`. The report groups on this field and refuses
  to place figures from different classes side by side.
- **Temperature scaling with a refusal gate.** Fitted on a held-out split.
  When the residual says temperature is the wrong correction, or the split has
  fewer than 200 rows, the tool emits no temperature rather than one that does
  not fit.
- **Cascade threshold selection.** Given the cost of one escalation and one
  wrong answer, it states where to cut and what that buys. Without both
  numbers it refuses, because no benchmark can know them.
- **Cost from reported tokens against a dated pricing table.** Every entry
  carries its source and the date it was read. A blank cost column names which
  of four reasons made it blank.
- **Operator-supplied pricing** via `--pricing`, for vendors whose terms treat
  their rates as confidential. Those ship unpriced, naming the page to read.
- **A loader that refuses rather than repairs.** A row whose gold label is not
  among its own options is refused with its line number, because scoring it
  would mark every system wrong and read as a model failure.
- **Latency percentiles by nearest rank**, and a results artifact recording the
  requested model, the model that answered, the dataset hash and row count, and
  the pricing entry applied, with credentials redacted.
- Apache-2.0. The vendored JevBench fixture is MIT and attributed in
  `datasets/public/README.md`.

### Known limitations at release

Ordinal Score rows load but are excluded from every figure. One request per
case, so cost and latency are conservative relative to batched use. Temperature
scaling only. Verified on Windows and Ubuntu, Python 3.12 and 3.13.
