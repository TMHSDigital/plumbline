// Check the site's search. Runs on the runner's own Node with the standard
// library only, like check_floor_parity.mjs.
//
//   node scripts/check_search.mjs            ranking cases only
//   node scripts/check_search.mjs _site      also the built index and the files the scripts load
//
// Ranking: site/search.js is loaded as a module and held to a handful of cases
// that say what a reader should get first. Given the assembled site, every
// entry in search-index.json must point at a page that exists and an id on it,
// and the other files the page scripts fetch or load (the worker, the example
// run) must be there, since check_site_links.mjs only follows tags in HTML.

import { readFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const Search = createRequire(import.meta.url)(path.join(here, "..", "site", "search.js"));

const problems = [];
const expect = (condition, what) => {
  if (!condition) problems.push(what);
};

// ---- ranking ------------------------------------------------------------

const fixture = [
  { t: "Methodology", h: "Methodology", u: "docs/methodology.html", x: "How plumbline measures what it measures." },
  { t: "Methodology", h: "Binning scheme and the ECE floor", u: "docs/methodology.html#binning", x: "Equal width bins, ten by default." },
  { t: "README", h: "Install", u: "docs/readme.html#install", x: "The ECE floor is computed for you. Bins bins bins bins bins bins bins." },
  { t: "Plan", h: "Calibración", u: "docs/plan.html#c", x: "Accents fold away." },
];
const first = (query) => Search.rank(fixture, query, 5)[0]?.entry.u;
const urls = (query) => Search.rank(fixture, query, 5).map((r) => r.entry.u);

expect(first("ece floor") === "docs/methodology.html#binning", "a heading match outranks a body match");
expect(first("bins") === "docs/methodology.html#binning" || first("bins") === "docs/readme.html#install", "bins finds a section");
expect(Search.rank(fixture, "bins", 5)[0].score <= 3 + 5, "body matches are capped per term");
expect(urls("ece install").join() === "docs/readme.html#install", "every term must match");
expect(urls("measur").includes("docs/methodology.html"), "a term matches the start of a word");
expect(!urls("easures").includes("docs/methodology.html"), "a term does not match inside a word");
expect(first("CALIBRACION") === "docs/plan.html#c", "case and accents are ignored");
expect(Search.rank(fixture, "   ", 5).length === 0, "an empty query finds nothing");
expect(Search.rank(fixture, "methodology", 5)[0].entry.u === "docs/methodology.html", "a page's own entry leads its sections on a tie");

const piece = Search.snippet("one two three floor four five", "floor", 160);
const marked = piece.marks.map(([s, e]) => piece.text.slice(s, e));
expect(marked.join() === "floor", `snippet marks the term (got ${JSON.stringify(marked)})`);
const long = `${"word ".repeat(80)}needle ${"word ".repeat(80)}`;
const window_ = Search.snippet(long, "needle", 100);
expect(window_.text.startsWith("… ") && window_.text.endsWith(" …"), "a snippet from the middle says so at both ends");
expect(window_.marks.length === 1 && window_.text.slice(...window_.marks[0]) === "needle", "the mark survives the ellipsis offset");

// ---- the built site -----------------------------------------------------

const site = process.argv[2];
let entries = 0;
if (site) {
  const read = (p) => readFileSync(path.join(site, ...p.split("/")), "utf8");
  const index = JSON.parse(read("search-index.json"));
  entries = index.length;
  expect(Array.isArray(index) && index.length > 20, "the index has entries");
  const pages = new Map();
  for (const entry of index) {
    for (const key of ["t", "h", "u", "x"]) {
      if (typeof entry[key] !== "string") problems.push(`${JSON.stringify(entry).slice(0, 80)}: ${key} is not a string`);
    }
    const url = new URL(entry.u, "https://site.invalid/");
    let page = url.pathname.slice(1);
    if (page === "" || page.endsWith("/")) page += "index.html";
    if (!pages.has(page)) pages.set(page, existsSync(path.join(site, ...page.split("/"))) ? read(page) : null);
    const html = pages.get(page);
    if (html === null) {
      problems.push(`${entry.u}: no page ${page}`);
      continue;
    }
    const id = decodeURIComponent(url.hash.slice(1));
    if (id && !html.includes(`id="${id}"`)) problems.push(`${entry.u}: no id "${id}" on ${page}`);
  }
  // Loaded by script rather than by a tag, so check_site_links.mjs cannot see them.
  for (const file of ["search-index.json", "search.js", "floor-worker.js", "example-run.json", "floor.js"]) {
    expect(existsSync(path.join(site, file)), `${file} is missing from the site`);
  }
}

if (problems.length) {
  console.error(`check_search: ${problems.length} problem(s):\n  ${problems.join("\n  ")}`);
  process.exit(1);
}
console.log(`check_search: ranking cases pass${site ? `; ${entries} index entries point at pages and ids that exist` : ""}.`);
