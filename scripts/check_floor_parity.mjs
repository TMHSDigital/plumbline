// Holds site/floor.js to the golden values the Python exported.
//
// Runs on Node's standard library alone, with no package install:
//
//     node scripts/check_floor_parity.mjs [_site/example-run.json]
//
// Given a built site's example-run.json, it also derives the example report's
// ECE line from the example's rows and requires it to match the report exactly.
//
// Exits non-zero on any disagreement larger than the fixture's tolerance, on any
// verdict sentence that differs by a character, and on a PCG64 starting state
// that differs from numpy's. The site deploy depends on this passing.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const floor = createRequire(import.meta.url)(join(root, "site", "floor.js"));
const fixture = JSON.parse(readFileSync(join(root, "site", "floor-golden.json"), "utf8"));
const tolerance = fixture.tolerance;

const failures = [];

for (const [seed, expected] of Object.entries(fixture.pcg64_initial_state)) {
  const actual = floor._internal.seedStates[seed];
  if (!actual || actual.state !== expected.state || actual.inc !== expected.inc) {
    failures.push(`PCG64 state for seed ${seed} differs from numpy's`);
  }
}

let worst = 0;
for (const c of fixture.cases) {
  const label = `n=${c.n} bins=${c.n_bins} accuracy=${c.accuracy}`;
  const started = Date.now();
  const band = floor.syntheticFloor(c.n, c.n_bins, c.accuracy);
  const elapsed = Date.now() - started;

  for (const key of ["mean", "p95"]) {
    const gap = Math.abs(band[key] - c[key]);
    worst = Math.max(worst, gap);
    if (!(gap <= tolerance)) {
      failures.push(`${label}: ${key} ${band[key]} against Python ${c[key]} (off by ${gap})`);
    }
  }
  for (const [measured, expected] of Object.entries(c.statements)) {
    const actual = floor.statement(Number(measured), band);
    if (actual !== expected) {
      failures.push(`${label}: verdict for ECE ${measured} differs\n  js:     ${actual}\n  python: ${expected}`);
    }
  }
  console.log(`${label.padEnd(36)} mean ${band.mean.toFixed(6)}  p95 ${band.p95.toFixed(6)}  ${elapsed}ms`);
}

// Pasted predictions: the page parses the text, then scores it as a report
// does. Each case must parse to the rows the Python scored and print the same
// line, character for character.
for (const c of fixture.pasted) {
  const label = `pasted, ${c.label}`;
  const parsed = floor.parsePredictions(c.text);
  if (parsed.errors.length) {
    failures.push(`${label}: refused: ${JSON.stringify(parsed.errors.slice(0, 2))}`);
    continue;
  }
  if (parsed.rows !== c.n) failures.push(`${label}: read ${parsed.rows} rows, the Python scored ${c.n}`);
  const measured = floor.ece(parsed.probabilities, parsed.correct, c.n_bins);
  const band = floor.calibrationFloor(parsed.probabilities, c.n_bins);
  for (const [key, actual] of [["ece", measured], ["mean", band.mean], ["p95", band.p95]]) {
    const gap = Math.abs(actual - c[key]);
    worst = Math.max(worst, gap);
    if (!(gap <= tolerance)) failures.push(`${label}: ${key} ${actual} against Python ${c[key]} (off by ${gap})`);
  }
  const line = floor.statement(measured, band);
  if (line !== c.statement) failures.push(`${label}: verdict differs\n  js:     ${line}\n  python: ${c.statement}`);
}
console.log(`${fixture.pasted.length} pasted texts parsed and scored`);

// What the parser refuses, and the line it names. There is no Python to hold
// these to: the report never reads pasted text. They pin the page's own rules.
const refusals = [
  ["0.5,1\n1.2,0", [2, "the probability 1.2 is outside 0 to 1"]],
  ["probability,outcome\n0.5,1\nabc,1", [3, 'the probability "abc" is not a number']],
  ["NaN,1\n0.5,1", [1, 'the probability "NaN" is not a number']],
  ["0.5,2", [1, 'the outcome "2" is not 0 or 1']],
  ["0.5\n0.5,1", [1, "expected two values, a probability and an outcome, and found 1"]],
  ["0.5,1,0.3", [1, "expected two values, a probability and an outcome, and found 3"]],
  ["", [null, "there are no rows to score. Paste one row per line: a probability, then 0 or 1"]],
  ["probability,outcome\n\n", [null, "there are no rows to score. Paste one row per line: a probability, then 0 or 1"]],
];
for (const [text, [line, message]] of refusals) {
  const { errors } = floor.parsePredictions(text);
  const got = errors[0];
  if (!got || got.line !== line || got.message !== message) {
    failures.push(`parser: ${JSON.stringify(text)} gave ${JSON.stringify(got)}, expected line ${line}: ${message}`);
  }
}
const capped = floor.parsePredictions("0.5,1\n0.5,0\n0.5,1", { maxRows: 2 }).errors;
if (!(capped.length === 1 && capped[0].line === null && capped[0].message.startsWith("3 rows is more than this page scores"))) {
  failures.push(`parser: a paste over the row cap gave ${JSON.stringify(capped)}`);
}
const lenient = floor.parsePredictions(' 0.5, 1,\n\n"0.25"\tfalse \n1e-1 TRUE\n');
if (lenient.errors.length || lenient.probabilities.join() !== "0.5,0.25,0.1" || lenient.correct.join() !== "true,false,true") {
  failures.push(`parser: lenient input read as ${JSON.stringify(lenient)}`);
}
console.log(`${refusals.length + 2} parser cases checked`);

// Every figure on the page is printed through fixed4, which must round as
// Python's "%.4f" does, including on and beside a tie.
const formats = Object.entries(fixture.fixed4);
const misformatted = formats.filter(([value, expected]) => floor.fixed4(Number(value)) !== expected);
for (const [value, expected] of misformatted.slice(0, 5)) {
  failures.push(`fixed4(${value}) is ${floor.fixed4(Number(value))}, Python's %.4f prints ${expected}`);
}
if (misformatted.length > 5) failures.push(`...and ${misformatted.length - 5} more fixed4 disagreements`);
console.log(`${formats.length} values formatted, ${misformatted.length} differ from Python's %.4f`);

console.log(`\n${fixture.cases.length} cases, largest difference ${worst.toExponential(2)}, tolerance ${tolerance}`);

// The worked example, when a built site is given: the page derives the example
// report's ECE line from the 105 rows, and it must come out character for
// character as the report prints it.
const examplePath = process.argv[2];
if (examplePath) {
  const example = JSON.parse(readFileSync(examplePath, "utf8"));
  if (example.adapter !== "mock") failures.push(`example rows come from ${example.adapter}, not the mock`);
  const measured = floor.ece(example.probabilities, example.correct, example.n_bins);
  const band = floor.calibrationFloor(example.probabilities, example.n_bins);
  const derived = floor.statement(measured, band);
  if (derived !== example.report.ece_line) {
    failures.push(`worked example: derived line differs from docs/example-report.md\n  js:     ${derived}\n  report: ${example.report.ece_line}`);
  }
  const summary = floor.syntheticFloor(example.probabilities.length, example.n_bins, example.python.accuracy);
  for (const key of ["mean", "p95"]) {
    const gap = Math.abs(summary[key] - example.python.summary_floor[key]);
    if (!(gap <= tolerance)) failures.push(`worked example: summary floor ${key} off by ${gap}`);
    // The line above agrees to 4 decimals; this holds the floor from the rows
    // to the same 1e-9 as every other case.
    const real = Math.abs(band[key] - example.python.calibration_floor[key]);
    if (!(real <= tolerance)) failures.push(`worked example: calibration floor ${key} off by ${real}`);
  }
  console.log(`worked example: ${derived.slice(0, 110)}...`);
}
if (failures.length) {
  console.error(`\n${failures.length} disagreement(s) with the Python:\n`);
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log("floor.js agrees with the Python.");
