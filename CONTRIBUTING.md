# Contributing

## New vendors are config, not code

This is the one thing to read before opening a pull request.

plumbline has three transports, and almost every "please support X" is already
one of them. Adding a vendor should not add a file.

- **Anything serving the Jev wire format is a `base_url` for `typesafe_wire`.**
  A self-hosted endpoint, a compatible server in front of an open model, a
  provider that implemented the same shape. It is a config entry. That includes
  the open decision models that serve this wire format: the unchanged adapter
  runs them.
- **Any open-weights checkpoint is a config entry for `local_logits`**, as a
  HuggingFace model id and a pinned revision. Pin the revision. A moving
  checkpoint makes every stored result unreproducible.
- **Any Anthropic model you want as a control arm is a model string for
  `generative`.** It speaks Anthropic's Messages API only, so another provider's
  chat model is a new transport, not a config entry.

A new adapter module is for a genuinely new *transport*, which is rare. If you
are writing one, say in the issue what wire shape it speaks and why none of the
three fit. There is an issue template for exactly this, and it exists to route
the conversation toward config before anyone writes code.

If a vendor is nearly compatible and something small blocks it, that is a bug in
the adapter worth fixing, not a reason for a fourth one.

**A new adapter config must not commit the vendor's published rates.** Name the
tariff page and leave the figures to the operator's own `--pricing` table. Some
vendors' terms make their pricing confidential and override the usual
public-knowledge exclusion, so copying a rate out of a public page into a file
this project publishes is a disclosure by this project. Secret scanning will not
catch it, because a price is not a credential format; see SECURITY.md and the
test named there.

## The rule about figures

**No figure may be rendered without its n and its null.**

This is not a style preference. A calibration number without its sample size and
its floor is not a weaker version of a finding, it is a number that reads like a
finding and is not one, and the entire point of this tool is to not do that. A
pull request that prints a bare metric will be asked to change, however correct
the arithmetic is.

The same applies to refusals. When there is not enough data, print the verdict
and the reason, not a value with a caveat attached.

## Before you open a pull request

All four must pass:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict
```

CI runs these on Python 3.12 and 3.13, on Ubuntu and Windows.

**Tests never need a network connection or an API key.** Every test must pass
without either. If a change needs a live call to be tested, the live call is
mocked and the real one goes in a manual check, recorded in `docs/PLAN.md`. A
test that reaches the network is a test that fails in CI for reasons unrelated
to the change.

## Working on this

`main` is protected by a ruleset. It requires a pull request and a green CI run,
and it blocks force-pushes and deletion. Merges are squash only.

The flow:

1. **Fork** the repository, or branch directly if you have write access.
2. **Branch** off `main`. Name it after the concern, not the ticket.
3. **Commit** as below: one concern per commit, conventional messages.
4. **Open a pull request.** No approval is required, because there is currently
   one maintainer and a rule demanding one would only demand it of them. CI is
   the gate that actually matters.
5. **CI must be green** before merge. The required checks are the four test
   jobs (ruff, ruff format, mypy --strict and pytest, on Python 3.12 and 3.13,
   on Ubuntu and Windows), the quickstart as the README documents it, the prose
   check (no em dashes, no `--` used as a dash), the built wheel, and the site's
   two checks (the floor agrees with the Python; every link, anchor, meta tag
   and policy resolves, and the pages work in a real browser).
6. **Squash on merge.** The branch is deleted automatically afterwards.

**Some checks are advisory and do not gate a merge.** CodeQL and Socket
Security both report on pull requests, and neither is a required check. The
checks listed above are the gate. This is deliberate, not an oversight: a
supply-chain advisory is a judgement call that a human should make, and a
scanner that can block a merge on a false positive ends up being routed around
rather than read.

So read them. A Socket alert on a dependency change is the one worth stopping
for, because it is the case the tooling is actually good at: a package that has
started running install scripts or reaching the network is a real signal even
with no CVE attached. Say in the pull request what you concluded. Clicking past
it silently is the failure mode, and so is panicking at a report that turns out
to be a transitive dependency's changelog.

The maintainer can bypass the ruleset, and does so for typos and documentation
rather than opening a pull request against themselves. That bypass is a
convenience for trivial changes, not a way around CI for real ones.

## Commits

One concern per commit, conventional messages.

```
fix(adapters): request a model the API actually serves
feat(report): print the recalibration verdict and the cascade sentence
docs(methodology): write down what this build measured
```

The body should say why, not what. The diff already says what. If a commit is
fixing something that was wrong, say what the wrong behavior was, because that
is the part nobody can reconstruct later.

## Pricing entries

Every pricing entry carries the source it was read from and the date it was
read. Both are required, in shipped config and in a user-supplied table alike.
An undated price with no source reads in a report exactly like one checked this
morning.

Do not add a figure for a vendor whose terms treat pricing as confidential.
Those ship unpriced, naming the page to read, and the operator supplies the
numbers with `--pricing`. See `docs/pricing.example.json`.

Use `null`, never `0`, where a vendor publishes no price. `null` is the absence
of a claim; `0` is the claim that those tokens are free.

## Datasets

Never commit anything to `datasets/private/`. It is gitignored and it is where
real labeled data goes.

Run artifacts in `results/` are gitignored too. They contain per-case records
from your own data, and on a live run, from a vendor call. Do not paste one into
an issue without reading it first.

## Working on the site

The site at <https://tmhsdigital.github.io/plumbline/> is built from `site/`
and the repository's markdown by `scripts/build_site.py`, and deployed by
`.github/workflows/site.yml` from `main` only. It needs Node 22 or later on
the path (for its built-in WebSocket) and, for the browser check, Chrome.
Nothing is installed by npm.

```
uv run python scripts/build_site.py --out _site
node scripts/check_floor_parity.mjs _site/example-run.json
node scripts/check_site_links.mjs _site
node scripts/check_search.mjs _site
node scripts/smoke_site.mjs _site
```

- `site/floor.js` is a JavaScript port of the Python floor, held to it within
  1e-9. After changing the floor in Python, run
  `uv run python scripts/floor_golden.py` to regenerate
  `site/floor-golden.json`, and commit both; CI refuses a stale fixture.
- `docs/example-report.md` is checked line for line against the command it
  records. After changing any report wording, rerun that command and replace
  everything from `# plumbline report` down; the build refuses otherwise.
- Every page carries a Content-Security-Policy that allows only the site's own
  files, so no inline script, inline style block, or `style=` attribute.
