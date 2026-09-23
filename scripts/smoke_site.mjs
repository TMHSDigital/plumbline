// Drive the assembled site in a real browser and fail if a reader would meet a
// broken page. Runs on the runner's own Node (22 or later, for its built-in
// WebSocket) and the Chrome the runner already has; nothing is installed. The
// browser is driven over the Chrome DevTools Protocol directly.
//
//   node scripts/smoke_site.mjs _site
//
// Set CHROME to the browser's path if it is not in a usual place.
//
// The site is served at /plumbline/, as Pages serves it, and the check fails on
// any console error, uncaught exception, or Content-Security-Policy violation,
// on any page, and if any of these does not hold:
//
// - the explainer opens with its result card, and the worked example derives
//   the report's line in the browser;
// - an invalid input says what is wrong (not "[object Object]") and marks the
//   field invalid;
// - the calculator runs, and writes its inputs into the address;
// - a link carrying those inputs runs the calculator as the page opens;
// - search, opened from the header on a doc page, finds a section of that doc.

import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";

const root = process.argv[2];
if (!root || !existsSync(path.join(root, "index.html"))) {
  console.error("usage: smoke_site.mjs <assembled site directory>");
  process.exit(2);
}
if (typeof WebSocket !== "function") {
  console.error(`smoke_site: Node ${process.version} has no built-in WebSocket; Node 22 or later is needed`);
  process.exit(2);
}

// ---- serve the site where Pages serves it --------------------------------

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".md": "text/markdown; charset=utf-8",
};
const PREFIX = "/plumbline/";

const server = createServer((request, response) => {
  const url = new URL(request.url, "http://localhost");
  if (!url.pathname.startsWith(PREFIX)) {
    response.writeHead(404).end();
    return;
  }
  let file = path.join(root, ...decodeURIComponent(url.pathname.slice(PREFIX.length)).split("/"));
  if (!path.resolve(file).startsWith(path.resolve(root))) {
    response.writeHead(403).end();
    return;
  }
  if (existsSync(file) && statSync(file).isDirectory()) file = path.join(file, "index.html");
  if (!existsSync(file)) {
    response.writeHead(404, { "content-type": TYPES[".html"] }).end(readFileSync(path.join(root, "404.html")));
    return;
  }
  response.writeHead(200, { "content-type": TYPES[path.extname(file)] ?? "application/octet-stream" });
  response.end(readFileSync(file));
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const site = `${origin}${PREFIX}`;

// ---- start the browser ----------------------------------------------------

function findChrome() {
  const candidates = [
    process.env.CHROME,
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  ];
  return candidates.find((candidate) => candidate && existsSync(candidate));
}
const chromePath = findChrome();
if (!chromePath) {
  console.error("smoke_site: no Chrome found; set CHROME to its path");
  process.exit(2);
}

// A free port, chosen here rather than by Chrome, so the DevTools address is
// known without reading Chrome's stderr (which a cold start on a CI runner has
// been seen to leave empty for longer than any sensible wait).
async function freePort() {
  const probe = createServer();
  await new Promise((resolve) => probe.listen(0, "127.0.0.1", resolve));
  const { port } = probe.address();
  await new Promise((resolve) => probe.close(resolve));
  return port;
}

// Starts Chrome and waits until its DevTools endpoint answers. A cold start is
// slow and occasionally stalls, so each attempt waits up to a minute and a
// stalled Chrome is killed and started again, three times at most.
async function launchChrome() {
  let last = "";
  for (let attempt = 1; attempt <= 3; attempt++) {
    const port = await freePort();
    const profile = mkdtempSync(path.join(tmpdir(), "plumbline-smoke-"));
    const child = spawn(chromePath, [
      "--headless=new",
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profile}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-gpu",
      "--disable-dev-shm-usage",
      "--window-size=1280,900",
      "about:blank",
    ], { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    let exited = null;
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("exit", (code) => { exited = code; });
    const deadline = Date.now() + 60000;
    while (Date.now() < deadline && exited === null) {
      try {
        const response = await fetch(`http://127.0.0.1:${port}/json/version`);
        if (response.ok) {
          const { webSocketDebuggerUrl } = await response.json();
          return { child, profile, browserUrl: webSocketDebuggerUrl };
        }
      } catch {
        // Not listening yet.
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    last = exited === null ? "did not answer within 60s" : `exited with ${exited}`;
    console.log(`  Chrome attempt ${attempt} ${last}; ${attempt < 3 ? "starting another" : "giving up"}`);
    child.kill();
    try {
      rmSync(profile, { recursive: true, force: true });
    } catch {
      // Held briefly after exit; the OS cleans temp.
    }
    if (attempt === 3) throw new Error(`Chrome ${last}:\n${stderr}`);
  }
}

const { child: chrome, profile, browserUrl } = await launchChrome();

// ---- a minimal DevTools client ---------------------------------------------

const socket = new WebSocket(browserUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});
let nextId = 0;
const waiting = new Map();
const listeners = [];
socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id !== undefined && waiting.has(message.id)) {
    const { resolve, reject } = waiting.get(message.id);
    waiting.delete(message.id);
    if (message.error) reject(new Error(`${message.error.message} (${message.error.code})`));
    else resolve(message.result);
  } else if (message.method) {
    for (const listener of listeners) listener(message);
  }
});
function send(method, params = {}, sessionId) {
  const id = ++nextId;
  socket.send(JSON.stringify({ id, method, params, sessionId }));
  return new Promise((resolve, reject) => waiting.set(id, { resolve, reject }));
}

const { targetId } = await send("Target.createTarget", { url: "about:blank" });
const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
const page = (method, params) => send(method, params, sessionId);

// Everything a reader's console would show as a problem.
const problems = [];
let where = "about:blank";
listeners.push(({ method, params, sessionId: from }) => {
  if (from !== sessionId) return;
  if (method === "Runtime.exceptionThrown") {
    const details = params.exceptionDetails;
    problems.push(`${where}: uncaught ${details.exception?.description ?? details.text}`);
  } else if (method === "Runtime.consoleAPICalled" && (params.type === "error" || params.type === "assert")) {
    problems.push(`${where}: console.${params.type}: ${params.args.map((a) => a.value ?? a.description).join(" ")}`);
  } else if (method === "Log.entryAdded" && params.entry.level === "error") {
    problems.push(`${where}: ${params.entry.source}: ${params.entry.text}`);
  }
});
await page("Runtime.enable");
await page("Log.enable");
await page("Page.enable");
// Recorded on every page before its own scripts run, so a violation the
// console might word differently still counts.
await page("Page.addScriptToEvaluateOnNewDocument", {
  source: `window.__violations = [];
    document.addEventListener("securitypolicyviolation", (e) =>
      window.__violations.push(e.violatedDirective + " " + (e.blockedURI || "inline")));`,
});

async function evaluate(expression) {
  const { result, exceptionDetails } = await page("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (exceptionDetails) throw new Error(`${expression.slice(0, 80)}: ${exceptionDetails.exception?.description ?? exceptionDetails.text}`);
  return result.value;
}

// Polls a condition in the page until it holds.
async function until(condition, what, timeout = 30000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await evaluate(`Boolean(${condition})`)) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`timed out waiting for ${what}`);
}

// Loads a page and waits for its load event. A navigation to the address the
// page already has is a same-document jump with no load event, so each check
// opens an address of its own; the timeout turns a mistake there into a
// failure rather than a hang.
async function open(sitePath) {
  where = sitePath || "index.html";
  let listener;
  const loaded = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${where} did not finish loading`)), 20000);
    listener = ({ method, sessionId: from }) => {
      if (method === "Page.loadEventFired" && from === sessionId) {
        clearTimeout(timer);
        resolve();
      }
    };
    listeners.push(listener);
  });
  try {
    await page("Page.navigate", { url: site + sitePath });
    await loaded;
  } finally {
    listeners.splice(listeners.indexOf(listener), 1);
  }
}

async function violations() {
  const found = await evaluate("window.__violations || []");
  for (const v of found) problems.push(`${where}: CSP violation: ${v}`);
}

const failures = [];
async function check(name, body) {
  try {
    await body();
    console.log(`  ok    ${name}`);
  } catch (error) {
    failures.push(`${name}: ${error.message}`);
    console.log(`  FAIL  ${name}: ${error.message}`);
  }
}
const expect = (condition, what) => {
  if (!condition) throw new Error(what);
};
const $ = (id) => `document.getElementById(${JSON.stringify(id)})`;

// ---- the checks ------------------------------------------------------------

console.log(`smoke_site: ${chromePath}\n  on ${site}`);

await check("the explainer opens with its result card and derives the example", async () => {
  await open("");
  expect(await evaluate("!!document.querySelector('.hero-card .gauge')"), "no result card");
  await until(`!${$("ex-steps")}.hidden`, "the worked example to be drawn");
  const match = await evaluate(`${$("ex-match")}.textContent`);
  expect(match.startsWith("Derived here, and identical"), `the derived line does not match: ${match}`);
  expect(await evaluate("!document.querySelector('.search-open').hidden"), "site.js did not run: search is still hidden");
  await violations();
});

await check("an invalid input says what is wrong and marks the field", async () => {
  await evaluate(`${$("n")}.value = "0"; ${$("calc-go")}.click();`);
  const status = await evaluate(`${$("calc-status")}.textContent`);
  expect(status === "Rows must be a whole number from 1 to 100,000.", `status reads ${JSON.stringify(status)}`);
  expect((await evaluate(`${$("n")}.getAttribute("aria-invalid")`)) === "true", "the field is not marked invalid");
});

await check("the calculator runs and keeps its inputs in the address", async () => {
  await evaluate(`${$("n")}.value = "200"; ${$("measured")}.value = "0.2"; ${$("calc-go")}.click();`);
  await until(`!${$("calc-result")}.hidden && !${$("calc-go")}.disabled`, "a calculator result");
  const status = await evaluate(`${$("calc-status")}.textContent`);
  expect(status.startsWith("Floor mean "), `status reads ${JSON.stringify(status)}`);
  const search = await evaluate("location.search");
  expect(search.includes("n=200") && search.includes("measured=0.2"), `address is ${search}`);
  await violations();
});

await check("a link carrying the inputs runs the calculator as the page opens", async () => {
  // Not the address the previous check left behind (see open()).
  await open("?n=150&bins=10&accuracy=0.8&measured=0.25#calculator");
  await until(`!${$("calc-result")}.hidden`, "the linked result");
  const verdict = await evaluate(`${$("calc-verdict")}.textContent`);
  expect(verdict.includes("ECE 0.2500 over 150 rows"), `verdict reads ${JSON.stringify(verdict.slice(0, 80))}`);
  await violations();
});

await check("search on a doc page finds a section of it", async () => {
  await open("docs/methodology.html");
  await evaluate("document.querySelector('.search-open').click()");
  await until("document.querySelector('dialog.search')?.open", "the search dialog");
  await evaluate(`(() => { const input = document.querySelector('dialog.search input');
    input.value = 'temperature'; input.dispatchEvent(new Event('input')); })()`);
  await until("document.querySelectorAll('.search-results a').length > 0", "search results");
  const first = await evaluate("document.querySelector('.search-results a').href");
  expect(first.includes("/docs/methodology.html#"), `the first result is ${first}`);
  await violations();
});

await check("every doc page loads without an error", async () => {
  const index = JSON.parse(readFileSync(path.join(root, "search-index.json"), "utf8"));
  const pages = [...new Set(index.map((entry) => entry.u.split("#")[0]).filter((u) => u.startsWith("docs/")))];
  for (const sitePath of [...pages, "docs/", "404.html"]) {
    await open(sitePath);
    await violations();
  }
});

// ---- report ----------------------------------------------------------------

socket.close();
chrome.kill();
server.close();
await new Promise((resolve) => chrome.once("exit", resolve).once("error", resolve));
try {
  rmSync(profile, { recursive: true, force: true });
} catch {
  // Chrome can hold the profile a moment after it exits; the OS cleans temp.
}

const all = [...failures, ...problems];
if (all.length) {
  console.error(`\nsmoke_site: ${all.length} problem(s):\n  ${all.join("\n  ")}`);
  process.exit(1);
}
console.log("smoke_site: every flow works, with no console error and no CSP violation.");
