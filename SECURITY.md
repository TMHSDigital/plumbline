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
