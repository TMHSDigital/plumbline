// Check the assembled site as a visitor would meet it. Runs on the runner's own
// Node with the standard library only.
//
//   node scripts/check_site_links.mjs _site
//   node scripts/check_site_links.mjs https://tmhsdigital.github.io/plumbline/
//
// Given a directory it checks the build before deploy; given the site's URL it
// checks what is actually being served. Either way, starting from the explainer
// and following every internal link, it fails if:
//
// - an internal link, stylesheet, script, or image does not resolve,
// - a #fragment names no id on the page it points to,
// - a page loads anything (script, stylesheet, image, frame) from another origin,
// - a page lacks its canonical, Open Graph, or Twitter card tags, or its
//   canonical and og:url do not name the page itself,
// - og.png is not a 1200x630 PNG,
// - the 404 page is missing, links anywhere that does not resolve, or (live)
//   a missing path is not answered with it and a 404 status.

import { readFile } from "node:fs/promises";
import path from "node:path";

const SITE_URL = "https://tmhsdigital.github.io/plumbline/";
const SITE_PATH = new URL(SITE_URL).pathname; // "/plumbline/"

const target = process.argv[2];
if (!target) {
  console.error("usage: check_site_links.mjs <site directory | site URL>");
  process.exit(2);
}
const live = /^https?:\/\//.test(target);
const base = live ? (target.endsWith("/") ? target : `${target}/`) : null;

// Fetch one site path ("" is the explainer). Returns { status, text, bytes }.
async function get(sitePath) {
  if (live) {
    const response = await fetch(base + sitePath, { redirect: "follow", cache: "no-store" });
    const bytes = new Uint8Array(await response.arrayBuffer());
    return { status: response.status, bytes, text: new TextDecoder().decode(bytes) };
  }
  try {
    const bytes = new Uint8Array(await readFile(path.join(target, ...sitePath.split("/"))));
    return { status: 200, bytes, text: new TextDecoder().decode(bytes) };
  } catch {
    return { status: 404, bytes: new Uint8Array(), text: "" };
  }
}

// Map a reference found on the page at `from` to a site path, or report it as
// external. `from` is a site path such as "docs/plan.html".
function resolve(reference, from) {
  if (/^(data|mailto):/i.test(reference)) return { skip: true };
  let url;
  if (/^https?:\/\//i.test(reference)) {
    if (!reference.startsWith(SITE_URL)) return { external: reference };
    url = new URL(reference);
  } else if (reference.startsWith("//")) {
    return { external: reference };
  } else {
    url = new URL(reference, SITE_URL + from);
  }
  if (!url.pathname.startsWith(SITE_PATH)) return { error: `${reference} leaves the site's path` };
  let sitePath = decodeURIComponent(url.pathname.slice(SITE_PATH.length));
  if (sitePath === "" || sitePath.endsWith("/")) sitePath += "index.html";
  return { sitePath, fragment: url.hash ? decodeURIComponent(url.hash.slice(1)) : null };
}

// Every tag with a reference, as [tag, attribute, value, rel].
function references(html) {
  const found = [];
  for (const [, tag, attributes] of html.matchAll(/<(a|link|script|img|iframe|source|video|audio)\b([^>]*)>/gi)) {
    const rel = /\brel="([^"]*)"/i.exec(attributes)?.[1] ?? "";
    for (const [, name, value] of attributes.matchAll(/\b(href|src)="([^"]*)"/gi)) {
      found.push([tag.toLowerCase(), name.toLowerCase(), value.replaceAll("&amp;", "&"), rel.toLowerCase()]);
    }
  }
  return found;
}

const ids = (html) => new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
const meta = (html, key, value) =>
  new RegExp(`<meta (?:property|name)="${key}" content="${value}">`).test(html);
const metaValue = (html, key) =>
  new RegExp(`<meta (?:property|name)="${key}" content="([^"]*)">`).exec(html)?.[1] ?? null;

const problems = [];
const pages = new Map(); // site path -> html, for fragment checks
const checked = new Set();

// The canonical URL a page should declare.
const canonicalFor = (sitePath) =>
  SITE_URL + (sitePath.endsWith("index.html") ? sitePath.slice(0, -"index.html".length) : sitePath);

async function page(sitePath) {
  if (!pages.has(sitePath)) {
    const { status, text } = await get(sitePath);
    pages.set(sitePath, status === 200 ? text : null);
  }
  return pages.get(sitePath);
}

function checkSocial(sitePath, html) {
  const want = canonicalFor(sitePath);
  const canonical = /<link rel="canonical" href="([^"]*)">/.exec(html)?.[1];
  if (canonical !== want) problems.push(`${sitePath}: canonical is ${canonical ?? "missing"}, expected ${want}`);
  if (metaValue(html, "og:url") !== want) problems.push(`${sitePath}: og:url does not name the page`);
  for (const key of ["og:title", "og:description", "og:image:alt", "twitter:title", "twitter:description"]) {
    if (!metaValue(html, key)) problems.push(`${sitePath}: ${key} is missing or empty`);
  }
  for (const [key, value] of [
    ["og:image", `${SITE_URL}og.png`],
    ["og:image:width", "1200"],
    ["og:image:height", "630"],
    ["twitter:card", "summary_large_image"],
    ["twitter:image", `${SITE_URL}og.png`],
  ]) {
    if (!meta(html, key, value)) problems.push(`${sitePath}: ${key} is not ${value}`);
  }
}

// Check one page's references; queue the internal HTML pages it links to.
async function crawl(sitePath, { social = true, servedAs = sitePath } = {}) {
  if (checked.has(sitePath)) return;
  checked.add(sitePath);
  const html = await page(sitePath);
  if (html === null) {
    problems.push(`${sitePath}: not found`);
    return;
  }
  if (social) checkSocial(sitePath, html);
  const queue = [];
  for (const [tag, , value, rel] of references(html)) {
    const where = `${servedAs}: <${tag}> ${value}`;
    if (value.startsWith("#")) {
      if (!ids(html).has(decodeURIComponent(value.slice(1)))) problems.push(`${where}: no such id on this page`);
      continue;
    }
    const loads = tag !== "a" && !(tag === "link" && /\bcanonical\b/.test(rel));
    const resolved = resolve(value, servedAs);
    if (resolved.skip) continue;
    if (resolved.error) {
      problems.push(`${where}: ${resolved.error}`);
      continue;
    }
    if (resolved.external) {
      if (loads) problems.push(`${where}: loads from another origin`);
      continue;
    }
    const { sitePath: to, fragment } = resolved;
    if (to.endsWith(".html")) {
      const linked = await page(to);
      if (linked === null) problems.push(`${where}: does not resolve`);
      else {
        if (fragment && !ids(linked).has(fragment)) problems.push(`${where}: no id "${fragment}" on ${to}`);
        queue.push(to);
      }
    } else {
      const { status } = await get(to);
      if (status !== 200) problems.push(`${where}: does not resolve (${status})`);
    }
  }
  for (const next of queue) await crawl(next);
}

await crawl("index.html");

// og.png: a real PNG of the size the tags claim.
{
  const { status, bytes } = await get("og.png");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const png = status === 200 && bytes.length > 24 && view.getUint32(0) === 0x89504e47;
  if (!png) problems.push("og.png: missing or not a PNG");
  else if (view.getUint32(16) !== 1200 || view.getUint32(20) !== 630) {
    problems.push(`og.png: ${view.getUint32(16)}x${view.getUint32(20)}, expected 1200x630`);
  }
}

// The 404 page, checked as if served at a deep missing path.
{
  const missing = `no-such-page-${Date.now()}/deeper/still.html`;
  if (live) {
    const { status, text } = await get(missing);
    if (status !== 404) problems.push(`${missing}: status ${status}, expected 404`);
    if (!text.includes("<h1>Not found</h1>")) problems.push(`${missing}: not answered with the 404 page`);
  }
  await crawl("404.html", { social: false, servedAs: missing });
}

const htmlPages = [...checked].length;
if (problems.length) {
  console.error(`${problems.length} problem(s) on ${live ? base : target}:\n  ${problems.join("\n  ")}`);
  process.exit(1);
}
console.log(`${htmlPages} pages checked on ${live ? base : target}: every link, anchor, and meta tag resolves; nothing loads from another origin.`);
