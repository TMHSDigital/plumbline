// Render the repository's markdown docs to HTML fragments for the site.
//
// Called by scripts/build_site.py, which owns the page template. It runs on the
// runner's own Node with the standard library and one vendored file,
// site/vendor/markdown-it/markdown-it.min.js; nothing is installed.
//
// Input, on stdin, as JSON:
//   { repo, sha, tracked: [paths], docs: [{ source, slug, markdown }] }
// Output, on stdout, as JSON: { slug: { html, headings, sections } }, where
// headings is [{ level, id, text }] for the page's table of contents and
// sections is [{ id, heading, level, text }] (plain text split at h1 to h3)
// for the site's search index.
//
// Rendering happens here, at build time, not in the browser: the pages need no
// script to be read and navigated (the site's one script only adds search, a
// theme toggle, and copy buttons), and every rule below fails the build
// instead of degrading on a reader's screen. The rules:
//
// - Raw HTML in markdown is refused unless it is one of a handful of literal
//   tags (ALLOWED_TAGS). There is no sanitizer to get wrong, only an allowlist
//   of exact strings, so markdown content cannot put a script on the site.
// - A relative link to another hosted doc becomes a link to that doc's page,
//   and its #fragment must name a heading that exists there.
// - A relative link to any other tracked file becomes a GitHub link to that
//   file at the commit being built, so it shows the code the page describes.
// - A relative link to anything else fails the build.
// - An image from another origin is rendered as its alt text, so the site
//   makes no third-party request.
//
//   node scripts/render_docs.mjs --self-test

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const VENDORED = path.join(here, "..", "site", "vendor", "markdown-it", "markdown-it.min.js");

// markdown-it 15.0.2, dist/browser/markdown-it.umd.min.js from the npm tarball
// whose integrity the registry states. site/vendor/markdown-it/README.md says
// how to update it; change this hash in the same commit.
const VENDORED_SHA256 = "635972b985228e8af9f0143647c68616b7a3bb09f6946e7e4a52e43dcf5e7be5";

// Exact tag strings, not patterns: README's collapsible install sections.
const ALLOWED_TAGS = new Set([
  "<details>",
  "<details open>",
  "</details>",
  "<summary>",
  "</summary>",
  "<b>",
  "</b>",
]);

const SCHEME = /^[a-z][a-z0-9+.-]*:/i;

class RenderError extends Error {}

function loadMarkdownIt() {
  const bytes = readFileSync(VENDORED);
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (actual !== VENDORED_SHA256) {
    throw new RenderError(
      `${path.relative(process.cwd(), VENDORED)} has sha256 ${actual}, expected ${VENDORED_SHA256}. ` +
        "Update the pinned hash only together with a deliberate vendor update.",
    );
  }
  return createRequire(import.meta.url)(VENDORED);
}

// GitHub's heading anchors (github-slugger), so a #fragment that works on
// github.com works here too.
function slugger() {
  const seen = new Map();
  return (text) => {
    const base = text.toLowerCase().replace(/[^\p{L}\p{M}\p{N}\p{Pc} -]/gu, "").replace(/ /g, "-");
    let slug = base;
    let count = seen.get(base) ?? 0;
    while (seen.has(slug)) slug = `${base}-${++count}`;
    seen.set(base, count);
    seen.set(slug, 0);
    return slug;
  };
}

function inlineText(token) {
  return (token.children ?? [])
    .filter((child) => child.type === "text" || child.type === "code_inline")
    .map((child) => child.content)
    .join("");
}

// Raw HTML is emitted verbatim, so this validates rather than sanitizes: every
// "<" in it must open one of the exact allowed tags, or the build fails.
function checkHtml(content, where, problems) {
  for (let at = content.indexOf("<"); at !== -1; at = content.indexOf("<", at + 1)) {
    const close = content.indexOf(">", at);
    const tag = close === -1 ? content.slice(at) : content.slice(at, close + 1);
    if (!ALLOWED_TAGS.has(tag.toLowerCase())) {
      problems.push(`${where}: raw HTML ${JSON.stringify(tag)} is not on the allowlist`);
    }
  }
}

// The line a token starts on, for error messages.
function lineOf(token, doc) {
  return token.map ? `${doc.source}:${token.map[0] + 1}` : doc.source;
}

// The plain text of a doc, split at each heading down to h3, for the site's
// search. Text before the first heading belongs to a section with no id.
function sectionsOf(tokens) {
  const sections = [];
  let current = { id: null, heading: "", level: 0, parts: [] };
  const flush = () => {
    const text = current.parts.join(" ").replace(/\s+/g, " ").trim();
    if (current.id !== null || text) sections.push({ id: current.id, heading: current.heading, level: current.level, text });
  };
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token.type === "heading_open" && Number(token.tag.slice(1)) <= 3) {
      flush();
      current = { id: token.attrGet("id"), heading: inlineText(tokens[i + 1]), level: Number(token.tag.slice(1)), parts: [] };
      i += 1; // the heading's own inline token is its title, not its body
    } else if (token.type === "inline") {
      current.parts.push(inlineText(token));
    } else if (token.type === "fence" || token.type === "code_block") {
      current.parts.push(token.content);
    }
  }
  flush();
  return sections;
}

export function render(job) {
  const markdownit = loadMarkdownIt();
  const md = markdownit({ html: true, linkify: false, typographer: false });
  const tracked = new Set(job.tracked);
  const trackedDirs = new Set();
  for (const file of job.tracked) {
    for (let dir = path.posix.dirname(file); dir !== "."; dir = path.posix.dirname(dir)) {
      trackedDirs.add(dir);
    }
  }
  const bySource = new Map(job.docs.map((doc) => [doc.source, doc]));
  const blob = (kind, target) => `https://github.com/${job.repo}/${kind}/${job.sha}/${target}`;

  // A link on each h2 and h3, so a section can be linked to without reading the
  // page source. It is emitted by the renderer, not written into the markdown,
  // so the raw HTML allowlist is untouched.
  md.renderer.rules.heading_close = (tokens, i, options, env, self) => {
    const open = tokens[i - 2];
    const id = open?.type === "heading_open" ? open.attrGet("id") : null;
    const anchor =
      id && (open.tag === "h2" || open.tag === "h3")
        ? ` <a class="anchor" href="#${id}" aria-label="Link to this section">#</a>`
        : "";
    return anchor + self.renderToken(tokens, i, options);
  };

  // First pass: parse everything and collect heading anchors, so links can be
  // checked against the headings of the doc they point into.
  const parsed = new Map();
  for (const doc of job.docs) {
    const tokens = md.parse(doc.markdown, {});
    const slug = slugger();
    const anchors = new Set();
    const headings = [];
    tokens.forEach((token, i) => {
      if (token.type !== "heading_open") return;
      const text = inlineText(tokens[i + 1]);
      const id = slug(text);
      token.attrSet("id", id);
      anchors.add(id);
      headings.push({ level: Number(token.tag.slice(1)), id, text });
    });
    parsed.set(doc.source, { tokens, anchors, headings });
  }

  const problems = [];

  function rewrite(href, doc, where) {
    if (href.startsWith("//") || SCHEME.test(href)) return href;
    const [target, fragment] = href.split("#", 2);
    if (target === "") {
      if (!parsed.get(doc.source).anchors.has(fragment)) {
        problems.push(`${where}: #${fragment} names no heading in ${doc.source}`);
      }
      return href;
    }
    const resolved = target.startsWith("/")
      ? path.posix.normalize(target.slice(1))
      : path.posix.normalize(path.posix.join(path.posix.dirname(doc.source), target));
    const clean = resolved.replace(/\/$/, "");
    const hosted = bySource.get(clean);
    if (hosted) {
      if (fragment !== undefined && !parsed.get(clean).anchors.has(fragment)) {
        problems.push(`${where}: #${fragment} names no heading in ${clean}`);
      }
      return `${hosted.slug}.html${fragment !== undefined ? `#${fragment}` : ""}`;
    }
    const suffix = fragment !== undefined ? `#${fragment}` : "";
    if (tracked.has(clean)) return blob("blob", clean) + suffix;
    if (trackedDirs.has(clean)) return blob("tree", clean) + suffix;
    problems.push(`${where}: ${href} resolves to ${clean}, which is neither a hosted doc nor a tracked file`);
    return href;
  }

  const out = {};
  for (const doc of job.docs) {
    const { tokens, headings } = parsed.get(doc.source);
    for (const token of tokens) {
      const where = lineOf(token, doc);
      // Table alignment arrives as an inline style; the site's CSP allows none.
      if (token.type === "th_open" || token.type === "td_open") {
        const align = /text-align:\s*(left|right|center)/.exec(token.attrGet("style") ?? "");
        if (align) {
          token.attrs = token.attrs.filter(([name]) => name !== "style");
          token.attrJoin("class", `align-${align[1]}`);
        }
      }
      if (token.type === "html_block") checkHtml(token.content, where, problems);
      for (const child of token.children ?? []) {
        if (child.type === "html_inline") checkHtml(child.content, where, problems);
        if (child.type === "link_open") {
          child.attrSet("href", rewrite(child.attrGet("href"), doc, where));
        }
        if (child.type === "image") {
          const src = child.attrGet("src");
          if (src.startsWith("//") || SCHEME.test(src)) {
            // Keep the words, drop the request.
            child.type = "text";
            child.content = child.content || inlineText(child);
            child.children = null;
          } else {
            problems.push(`${where}: image ${src} is not copied to the site; link it instead`);
          }
        }
      }
    }
    out[doc.slug] = {
      html: md.renderer.render(tokens, md.options, {}),
      headings,
      sections: sectionsOf(tokens),
    };
  }

  if (problems.length) throw new RenderError(problems.join("\n"));
  return out;
}

function selfTest() {
  const job = (markdown, extra = []) => ({
    repo: "o/r",
    sha: "abc",
    tracked: ["a.md", "src/x.py", ...extra],
    docs: [
      { source: "a.md", slug: "a", markdown },
      { source: "docs/b.md", slug: "b", markdown: "# B\n\n## Two words\n" },
    ],
  });
  const refuses = (markdown, pattern) => {
    try {
      render(job(markdown));
    } catch (error) {
      if (pattern.test(error.message)) return;
      throw new Error(`refused for the wrong reason: ${error.message}`);
    }
    throw new Error(`accepted what it should refuse: ${JSON.stringify(markdown)}`);
  };
  const html = (markdown) => render(job(markdown)).a.html;
  const expect = (condition, what) => {
    if (!condition) throw new Error(`self-test failed: ${what}`);
  };

  refuses("<script>alert(1)</script>\n", /not on the allowlist/);
  refuses("hi <img src=x onerror=alert(1)>\n", /not on the allowlist/);
  refuses('<details open onclick="x">\n', /not on the allowlist/);
  refuses("<details>\n<scr<b>ipt>alert(1)</script>\n</details>\n", /not on the allowlist/);
  refuses("[x](missing.md)\n", /neither a hosted doc nor a tracked file/);
  refuses("[x](docs/b.md#nope)\n", /names no heading/);
  refuses("# A\n\n[x](#nope)\n", /names no heading/);
  refuses("![x](pic.png)\n", /not copied to the site/);
  expect(!html("[x](javascript:alert(1))\n").includes("href"), "javascript: is linked");
  expect(html("[x](docs/b.md#two-words)\n").includes('href="b.html#two-words"'), "doc link");
  expect(html("[x](src/x.py)\n").includes('href="https://github.com/o/r/blob/abc/src/x.py"'), "blob");
  expect(html("[x](src)\n").includes('href="https://github.com/o/r/tree/abc/src"'), "tree");
  expect(!html("[![CI](https://e.x/b.svg)](https://e.x)\n").includes("<img"), "remote image");
  expect(html("# T\n\n## T\n").includes('id="t-1"'), "duplicate heading anchors");
  expect(html("<details open>\n<summary><b>PS</b></summary>\n\nx\n\n</details>\n").includes("<details open>"), "allowlist");
  expect(html("# T\n\n## Two\n").includes('<h2 id="two">Two <a class="anchor" href="#two"'), "h2 anchor");
  expect(!html("# T\n").includes('class="anchor"'), "no anchor on h1");
  const table = html("| a | b |\n|:-:|--:|\n| 1 | 2 |\n");
  expect(!table.includes("style=") && table.includes('class="align-right"'), "table alignment as a class");
  const out = render(job("# Top\n\nlead `x`\n\n## One\n\nbody\n\n```\ncode\n```\n\n#### Deep\n\nmore\n")).a;
  expect(
    JSON.stringify(out.headings.map((h) => [h.level, h.id])) === '[[1,"top"],[2,"one"],[4,"deep"]]',
    "headings",
  );
  const sections = out.sections.map((s) => [s.id, s.text]);
  expect(JSON.stringify(sections) === '[["top","lead x"],["one","body code Deep more"]]', `sections ${JSON.stringify(sections)}`);
  console.log("render_docs self-test: 20 cases pass");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    if (process.argv.includes("--self-test")) {
      selfTest();
    } else {
      const job = JSON.parse(readFileSync(0, "utf8"));
      process.stdout.write(JSON.stringify(render(job)));
    }
  } catch (error) {
    process.stderr.write(`${error instanceof RenderError ? "" : error.stack + "\n"}${error.message}\n`);
    process.exit(1);
  }
}
