# Security

## Reporting an issue

Report vulnerabilities through GitHub's private vulnerability reporting, under
the Security tab of this repository. That keeps the report private until there
is a fix.

Please do not open a public issue for a security problem.

## How plumbline handles credentials

- **API keys are read from environment variables only.** There is no key field
  in any config file plumbline reads, and no command line flag that takes one.
- **A key is never written to an artifact or a log line.** The runner redacts
  any config key whose name looks like a credential before anything reaches
  disk, matching on `key`, `token`, `secret`, `password`, `credential` and
  `authorization`. The matching is deliberately crude: it errs toward redacting
  a harmless field rather than letting a secret through.
- **A key is never part of a cache key.** Cache keys are written to disk and are
  derived from what changes the answer. A credential authenticates the caller;
  it does not change the answer.
- `.env` is gitignored. `.env.example` is the template and holds no values.

## What the automated controls cover, and what they do not

This repository has secret scanning and push protection enabled. Both work by
recognising **credential formats**: a token that looks like a token gets caught,
and push protection rejects the commit before it lands. That is the right
control for a leaked key and it is the reason a key has never reached a commit
here.

It is worth being precise about what that leaves.

The closest thing to a disclosure this project has actually had was not a
credential. It was a **vendor's price**, quoted from their documentation into a
shipped config file, under a customer agreement that makes pricing information
confidential and expressly overrides the usual public-knowledge exclusion. No
scanner recognises a price. `0.042` is a float. Push protection would have
passed it through without a murmur, and did, until it was caught by reading.

Contractual confidentiality is not a pattern-matching problem, so it does not
get a pattern-matching control. What guards it here is design and a test:

- **Design.** plumbline ships no pricing figures for a vendor whose terms treat
  them as confidential. The shipped entry names the tariff page and prices
  nothing, and the operator supplies the numbers themselves through
  `--pricing`, in their own working copy. Reading a published page and writing
  the number down is the operator's act, not this project's.
- **A test.** `tests/test_cost_and_latency.py::test_no_shipped_entry_states_a_figure_for_the_confidential_vendor`
  asserts that no shipped entry for that vendor carries a numeric price, in
  either a price field or its source string. It fails if a number is put back
  in a price field, and it fails if a price is quoted in the prose around one.
  Both paths were verified by injecting a figure and watching it go red.

If you are reviewing a change that touches pricing, that test is the control.
Do not assume a green secret-scanning badge says anything about it.

## Run artifacts contain your data

This is the part worth pausing on.

A results artifact in `results/` holds per-case records: the text of every case
that was sent, its labels, its gold label, and the full response. On a live run
that includes content that went to a third-party API.

- `results/`, `cache/`, `models/` and `datasets/private/` are all gitignored.
- Treat an artifact the way you would treat the dataset it came from. If the
  dataset is sensitive, the artifact is sensitive.
- Read an artifact before attaching it to an issue, a pull request, or a support
  thread.

## Scope

plumbline is a measurement tool that you run against your own data with your own
credentials. It has no server, no telemetry, and makes no network call except
the ones an adapter you selected makes to the vendor you pointed it at.
