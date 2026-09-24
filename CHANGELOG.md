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
- The site page: the argument, a worked example that derives the example
  report's ECE line from its 105 rows in the browser, and a sample-size planner.
  `scripts/build_site.py` regenerates the example from the mock adapter at build
  time and refuses if `docs/example-report.md` no longer matches its own
  command.
- The site hosts the repository's docs (README, METHODOLOGY, PLAN, the example
  report, CHANGELOG, CONTRIBUTING, SECURITY, and the dataset README), rendered
  at build time from the commit being deployed by a vendored markdown-it under
  the runner's Node. Each page names the commit and build time it came from.
  The build fails on a broken link or anchor, on raw HTML outside a short
  allowlist, and on a modified vendored renderer.
- The site is live at <https://tmhsdigital.github.io/plumbline/>, deployed by
  `site.yml` from `main` only after the parity check passes and
  `scripts/check_site_links.mjs` finds every link, anchor, and meta tag in the
  assembled site resolving and nothing loading from another origin. Every page
  carries a canonical URL and Open Graph and Twitter card tags; the card image
  (`site/og.png`) is rendered from `scripts/og_image.html`. Missing paths get a
  404 page that links back.
- The site has a header on every page (calculator, docs, example report,
  GitHub), and the doc pages have a grouped docs sidebar, an "On this page"
  contents list that follows the section in view, a link on every h2 and h3
  that copies itself, previous and next links, an "Edit on GitHub" link, and
  copy buttons on code blocks. All of the navigation is HTML written at build
  time, so it works with scripts off; `site/site.js` adds only the conveniences
  and a light, dark, or automatic theme that is remembered between visits.
  Form borders and the histogram's bars now meet 3:1 contrast in both themes.
- Search across the explainer and every doc, from the header or with `/` or
  Ctrl+K. The index (`search-index.json`) is written at build time from the
  rendered sections and fetched from the site only when search opens; the
  ranking (`site/search.js`) runs in the browser with no library.
  `scripts/check_search.mjs` holds the ranking to its cases and every index
  entry to a page and id that exist.
- The explainer opens with the example report's result drawn as a card (the
  measured ECE against its floor's 95th percentile, and the verdict), written
  at build time from the report's own line so it reads the same with scripts
  off, and keeps its contents in a rail beside the text on wide screens.
- The calculator and planner each run in their own worker and can be
  cancelled; their inputs are kept in the address, so a link reproduces a
  result, and each result has "Copy link" and "Copy result" buttons.
- Every page carries a Content-Security-Policy that allows nothing but the
  site's own files (no inline script or style), and `check_site_links.mjs`
  fails a page that lacks it or carries anything it would block.
- The site is checked in a real browser before it deploys.
  `scripts/smoke_site.mjs` drives the runner's own Chrome over the DevTools
  protocol with Node's standard library (nothing is installed), runs the
  calculator, a shared link, and search, and fails on any console error or CSP
  violation on any page. After a deploy, a new job checks the live site's
  links, anchors, meta tags, policy, and 404.

### Changed

- Report bullets separate the label from the figure with a colon instead of an
  em dash (`- **Cost**: not reported.`), and the docs no longer use em dashes.
  A CI job now fails on an em dash in tracked markdown or `src/`. **This
  changes report text, not any number.** The job also covers `site/`,
  `scripts/`, and `.github/`, and the HTML entity and JavaScript escape
  spellings of the character.
- `--` is no longer used as a stand-in for a dash: METHODOLOGY, the source
  comments and docstrings, and one report note (the ordinal-score line in the
  load summary) now use commas, colons, or parentheses. A second check in the
  same CI job fails on a bare `--` between words; flags, git's end-of-options
  separator, and HTML comments are unaffected. **This changes report text, not
  any number.**

### Fixed

- The dependency floors in pyproject were never tested, and three were wrong:
  scipy 1.14.0 has no wheel for Python 3.13, anthropic before 0.77 lacks the
  structured output types the generative adapter uses, and typer before 0.16
  breaks against click 8.2 (#47). The floors are now the oldest releases the
  whole suite passes on: numpy 2.1, scipy 1.14.1, typer 0.16, typesafe-sdk
  0.5.7, and anthropic 0.77. The two SDKs are capped below their next breaking
  series, httpx is no longer declared (nothing imported it), and a new CI job
  runs the suite on the oldest direct dependencies pyproject allows.
- A broken or missing SDK took down every command, the mock included, because
  the registry imported all four adapters at startup (#47). An adapter is now
  imported the first time it is created, and one that cannot be says which
  module is missing; the other adapters, and the adapters list, are unaffected.
- `scripts/build_site.py --out` deleted whatever directory it was given unless
  it was the repository, `site/`, or above them, so a typo such as `--out docs`
  or `--out .git` deleted source or history (#50). It now deletes only a
  directory that is empty or carries the `.plumbline-site` marker a build
  writes, and refuses anything else before doing any work. A `_site` built
  before this change has no marker, so it is refused once; remove it by hand.
- A row whose response named no model was never priced, because pricing
  looked up only the model reported; the guard, which prices the requested
  model, and the report then disagreed about the same run (#36). Such a row is
  now priced by the requested model, and its pricing key says so. A response
  that names a model missing from the table is still unpriced: a newer version
  never inherits an older rate.
- The cache key sorted the options, but `generative` and `local_logits` list
  them in their prompts in the order given, so a cached answer could be served
  for a prompt that was never sent (#37). For those two the key keeps the order.
- Option descriptions were parsed and then dropped: `typesafe_wire` sent every
  criterion as empty, and the dataset hash ignored descriptions and question
  types (#39). `typesafe_wire` now sends them as the choice's criteria and keys
  its cache on them, the dataset hash covers both where a row carries them (a
  plain choice dataset keeps its hash), and the report says when an adapter
  did not send the descriptions its rows carried. **The public fixture's
  dataset hash changes, from `c18e9496` to `1b96dc91`, and the example report
  gains that note; no number changes.**
- `plumbline report` combined artifacts from different datasets under the first
  one's hash, as if their figures were comparable, and gave two runs of one
  adapter identical headings (#43). Different datasets are now refused unless
  `--allow-mixed` is passed, when every arm names its dataset and row count;
  arms that share an adapter name are told apart by model, then by time.
- A credential inside a list in the run config reached the artifact, because
  redaction walked only dictionaries; it now walks lists and tuples too (#46).
  The report and the artifact named the dataset by the path as typed, so an
  absolute path shared a username and a directory layout; a dataset inside the
  working directory is named relative to it and one outside by its file name,
  and the hash still says which rows they were. A backtick in a model name no
  longer breaks out of its code span.
- The JevBench loader skipped the duplicate-id check the JSONL loader makes, so
  repeated ids loaded silently (#44). Both loaders now share it. The dataset
  loader read a byte order mark as part of the first row and refused it, turned
  a label of `null`, `true` or `1` into the text "None", "True" or "1", accepted
  empty labels and descriptions of options that do not exist, and failed on a
  non-UTF-8 file without naming it (#45). It now reads the mark as nothing,
  refuses each of the others with the reason, and names the file and line that
  is not UTF-8. The pricing loader accepted NaN, infinite and negative prices,
  `as_of` values such as `20260901` or `2026-W36-1`, and dates in the future,
  which kept the report from ever calling a price stale; each is refused now,
  and a byte order mark is read as nothing there too.
- The CLI accepted options that misbehaved or crashed (#41): `--limit 0` ran
  every row and a negative limit sliced from the end, `--boot 0` failed with a
  traceback after the run had been paid for, an option the adapter does not
  take and a missing API key each printed a traceback, and a `--report` path
  that was a directory crashed after the run. Counts must now be positive, and
  each of the rest is one line, before anything is sent.
- `plumbline run` exited 0 when every case failed, and printed its status and
  its errors on stdout, so `run > report.md` captured them and a script could
  not tell a run with no figures from a good one (#42). Status and errors now
  go to stderr, stdout carries only the report, and a run with no figures
  exits 1 after writing the artifact and the report.
- When every case failed for the same reason (usually a missing extra or key),
  the report said only that the failures were in the artifact (#15). It now
  prints that reason, once; different reasons are still left to the artifact.
- The cascade's threshold was chosen and scored on the same rows, so the
  coverage and cost it printed were its best case rather than what it would
  do; with no temperature recommended it used every row and still called them
  held-out (#30). It is now chosen on the fit half and reported on the
  held-out half, and the 200-row minimum counts the held-out half, so a run
  needs at least 400 scored rows for a threshold. **This changes which runs
  print a threshold, and the cost and coverage printed with it.**
- Equal-count binning split tied predictions across a bin boundary by input
  order, so the same rows gave an ECE of 0.3 in one order and 0.2 in another,
  and bins holding the same values were labelled with different ranges (#38).
  A cut now never falls inside a tie, and each bin's edges come from the rows
  it holds. Equal-width binning, the default, is unchanged.
- The cascade never considered escalating every case, because the highest
  observed score always kept the rows that reached it covered; with every
  case wrong and errors dear it chose a threshold costing $201 where
  escalating all three cost $3 (#34). Escalating everything is now a
  candidate, and the report says so in words when it wins. Cost per correct
  answer divided the priced total by correct answers from unpriced rows too,
  understating it (#35); it now counts correct answers among priced rows.
- A prediction whose distribution held NaN, a negative, or a value above 1 was
  accepted as long as the entries summed to about 1, and a NaN latency was
  accepted too (#32); each is now refused with the option it concerns. The
  multiclass Brier floor raised `IndexError` on a distribution summing to a
  little under 1, which `Prediction` allows (#31); it now draws from the
  distribution as reported, and one that sums to 1 draws exactly as before.
- An accuracy or AUROC far below its null was reported as INCONCLUSIVE with the
  advice to collect more rows, so an inverted score over 300 rows (AUROC 0.0)
  read as a sample-size problem (#33). Both nulls now carry a 5th percentile,
  and a value below it gets its own verdict: accuracy "worse than chance", AUROC
  "ranks incorrect above correct", each with the likely reason. Values above or
  inside the null read exactly as before.
- The pricing template priced every call at $0, so a copy used unedited let
  any run past `--max-cost-usd` (#28). Its prices are now null and its `as_of`
  a `YYYY-MM-DD` placeholder that the loader refuses, and any entry pricing
  both input and output at 0 is refused unless it says `"free": true`.
- A failure no retry can fix was retried anyway, and each retry was another
  billed call (#26). The runner retried every error but a refusal, including a
  wire-contract error, a checkpoint mismatch, a missing optional dependency and
  a 401, and both SDKs retried again underneath, so one case could make nine
  calls. Now only transport failures are retried (a dropped connection, a
  timeout, 408, 409, 425, 429 and 5xx), `Retry-After` is honoured up to 60
  seconds, the SDKs' own retries are off, and a failed case records the
  attempts it actually made.
- Two workers answering cases with the same text wrote the same cache entry
  through one shared temporary file; on Windows the loser raised
  `PermissionError` and aborted the run, losing every call already paid for
  (#27). Each write now has a temporary file of its own, a write that still
  fails is counted in the cache stats (`write_errors`) instead of raised, and
  cases sharing a key are answered once per run, the rest from the cache.
- An endpoint set through `TYPESAFE_BASE_URL` or `ANTHROPIC_BASE_URL` changed
  which server answered but reached neither the cache key nor the artifact, so
  a self-hosted run and a hosted one shared cache entries (#40). The adapters
  now resolve the endpoint the way their SDKs do, it is part of the key, the
  artifact records it as `endpoint`, and the report names it when it is not the
  vendor's default. Runs against the default endpoint keep their cache keys.
- Every validation message on the site's calculator and planner read
  "[object Object]" (#25). Messages now say what is wrong, mark the field
  invalid, and are tied to it for screen readers; a result is announced as one
  line rather than the whole result block (#53).
- A failure to draw the worked example was reported as the example being
  missing; the page now says which happened. `floor.js` turned a bin count of 0
  into 10 where the Python refuses it, and now refuses it too. The parity
  check holds the worked example's floor to 1e-9, not only to the four
  decimals the report prints (#55).
- A link to a later section of the explainer (`#planner`) stopped short of it,
  because the worked example grows the page after it loads.
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
- CI's guard against a traceback leaking from the quickstart refusal could
  never fail: `grep -qv Traceback` succeeds on any output with one line
  without the word (#49).
- CI ran every pull request twice, because both the `push` and `pull_request`
  triggers fired on a branch pushed to origin.

### Documentation

- The load summary said every JevBench row "is asked as a one-of-n choice",
  which is not true of an adapter that asks yes/no rows as yes/no questions;
  it now says each row is asked as the question type it states, and the
  report's "3 choice asked as failed" reads "3 choice rows failed" (#62).
  **This changes report text, not any number.**
- CONTRIBUTING said "all three" above four commands and named only the test
  jobs as the gate; it now lists every required check and has a section on
  working on the site (#63). PLAN no longer lists finished work as to do (#64).
  `.env.example` no longer claims a `.env` file is read or lists an Ollama
  adapter that does not exist, and the docs describe `generative` as the
  Anthropic Messages API it is (#65). The CLI's help says what the tool is and
  lists each option's choices, and `plumbline adapters` marks an adapter whose
  optional extra is missing (#66). The adapter template applies a label that
  exists, and there is a template for site bugs (#67).

- A new page, [Your own data](docs/datasets.md), gives the row format, what
  the loader refuses and why, and the commands a user needs next: a local
  checkpoint with its pinned revision, the cascade's two costs, and
  `plumbline report` for runs that already happened (#61). A test holds its
  example rows to the loader.
- The README's hosted-vendor command used `--max-cases 40` against 105 rows,
  which refuses rather than truncates, so it sent nothing (#29); it uses
  `--limit 40`, and `--max-cases` says in its help that it is a guard.
- The README said `pip install plumbline` "will not work"; it installs an
  unrelated project of the same name (#60). It now says so and gives pip
  commands that install this one, and the missing-extra error no longer points
  at the PyPI package.

- `docs/example-report.md` is now checked line for line against its recorded
  command on every site build, not only its ECE line, and the README says that
  instead of calling it "unedited".

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
