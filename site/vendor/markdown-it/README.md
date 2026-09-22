# Vendored: markdown-it

`markdown-it.min.js` is markdown-it 15.0.2's browser build,
`dist/browser/markdown-it.umd.min.js`, taken unmodified from the npm tarball
`https://registry.npmjs.org/markdown-it/-/markdown-it-15.0.2.tgz`. The tarball
matched the registry's stated integrity,
`sha512-q4IGxMv56jCqT4OCRCADBoDP3LO4MhmTXjFbphHPXs4g3j9Xg5RDnxqN8IF/3vIWEU+VCnUq+7JUg/cfy2E6Qw==`.
The file's own sha256 is pinned in `scripts/render_docs.mjs`, which refuses to
run if it changes.

- `LICENSE` is markdown-it's MIT license.
- `THIRD-PARTY-LICENSES` holds the licenses of the packages the browser build
  inlines: mdurl, entities (BSD-2-Clause), uc.micro, linkify-it, and
  punycode.js.

## What it is used for

`scripts/render_docs.mjs` runs it under the CI runner's own Node, at build time,
to turn the repository's markdown docs into the site's doc pages. It is not
shipped to the site: the pages it produces need no script. It was chosen because
it is maintained, has no install step once vendored, escapes raw HTML unless
told otherwise, and refuses `javascript:`, `vbscript:`, `file:`, and non-image
`data:` link targets. The build turns raw HTML on only to check it against an
allowlist of exact tags and fail on anything else.

## Updating it

1. Download the new version's tarball from the registry and check it against
   the `dist.integrity` the registry states for that version.
2. Replace `markdown-it.min.js` with the tarball's
   `dist/browser/markdown-it.umd.min.js`, and `LICENSE` with its `LICENSE`.
   Refresh `THIRD-PARTY-LICENSES` if the bundled dependencies changed.
3. Update the version, the integrity above, and `VENDORED_SHA256` in
   `scripts/render_docs.mjs`, in the same commit.
4. Run `node scripts/render_docs.mjs --self-test` and the site build.
